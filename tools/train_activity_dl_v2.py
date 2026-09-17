from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
INPUT = ROOT/"datasets/processed/bits2/activity_raw_windows.npz"
OUT = ROOT/"models/activity_dl_v2"
CLASSES = ["RESTING","SITTING","WALKING","RUNNING"]

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_data(path):
    d=np.load(path, allow_pickle=True)
    def pick(names):
        return next((k for k in names if k in d), None)
    xk,yk,gk=pick(["X","x","windows","data"]),pick(["y","labels","activity","target"]),pick(["groups","subjects","subject_id","group"])
    if not all((xk,yk,gk)): raise KeyError(f"Need X/y/groups (aliases accepted). Keys={list(d.keys())}")
    X=np.asarray(d[xk],dtype=np.float32); yr=np.asarray(d[yk]); groups=np.asarray(d[gk])
    if X.ndim!=3: raise ValueError(f"Expected 3-D X, got {X.shape}")
    # Canonical contract: (N,3,40). Explicitly handle both layouts.
    if X.shape[1:]==(3,40): pass
    elif X.shape[1:]==(40,3): X=np.transpose(X,(0,2,1))
    else: raise ValueError(f"Expected windows (3,40) or (40,3), got {X.shape[1:]}")
    if len(X)!=len(yr) or len(X)!=len(groups): raise ValueError("X/y/groups length mismatch")
    if np.issubdtype(yr.dtype,np.number): y=yr.astype(int)
    else:
        mp={c:i for i,c in enumerate(CLASSES)}
        try: y=np.array([mp[str(v).upper()] for v in yr],dtype=np.int64)
        except KeyError as e: raise ValueError(f"Unknown label {e.args[0]}")
    if sorted(np.unique(y).tolist())!=list(range(4)): raise ValueError(f"Labels must be 0..3, got {np.unique(y)}")
    return X,y,groups

def fit_std(X):
    if X.ndim!=3 or X.shape[1]!=3: raise ValueError(f"fit_std expects (N,3,T), got {X.shape}")
    # Correct 3-D reduction; no invalid transpose.
    m=X.mean(axis=(0,2),dtype=np.float64).astype(np.float32)
    s=np.maximum(X.std(axis=(0,2),dtype=np.float64),1e-6).astype(np.float32)
    return m,s

def apply_std(X,m,s): return ((X-m[None,:,None])/s[None,:,None]).astype(np.float32)

class Block(nn.Module):
    def __init__(self,a,b,d):
        super().__init__(); p=2*d
        self.f=nn.Sequential(nn.Conv1d(a,b,5,padding=p,dilation=d),nn.BatchNorm1d(b),nn.GELU(),
                             nn.Dropout(.12),nn.Conv1d(b,b,5,padding=p,dilation=d),nn.BatchNorm1d(b),nn.GELU(),nn.Dropout(.12))
        self.skip=nn.Conv1d(a,b,1) if a!=b else nn.Identity()
    def forward(self,x): return self.f(x)+self.skip(x)

class TCN(nn.Module):
    def __init__(self):
        super().__init__(); self.f=nn.Sequential(Block(3,32,1),Block(32,48,2),Block(48,64,4),Block(64,80,8))
        self.h=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(80,48),nn.GELU(),nn.Dropout(.18),nn.Linear(48,4))
    def forward(self,x): return self.h(self.f(x))

class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__(); self.c=nn.Sequential(nn.Conv1d(3,32,5,padding=2),nn.BatchNorm1d(32),nn.GELU(),
            nn.Conv1d(32,48,5,padding=2),nn.BatchNorm1d(48),nn.GELU(),nn.MaxPool1d(2),nn.Dropout(.1),
            nn.Conv1d(48,64,3,padding=1),nn.BatchNorm1d(64),nn.GELU())
        self.g=nn.GRU(64,48,batch_first=True); self.h=nn.Sequential(nn.Linear(48,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,4))
    def forward(self,x):
        z=self.c(x).transpose(1,2); z,_=self.g(z); return self.h(z[:,-1])

def make_model(name): return TCN() if name=="tcn" else CNNGRU()

def train(Xtr,ytr,Xv,yv,name,seedv,epochs,bs,patience,lr):
    seed(seedv); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=make_model(name).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3); lossfn=nn.CrossEntropyLoss(label_smoothing=.03)
    dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=bs,shuffle=True)
    vx=torch.from_numpy(Xv).to(dev); vy=torch.from_numpy(yv)
    best=-1; stale=0; state=None; be=0
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(set_to_none=True)
            loss=lossfn(model(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
        model.eval()
        with torch.no_grad(): pred=model(vx).argmax(1).cpu().numpy()
        score=f1_score(yv,pred,average="macro",zero_division=0)
        if score>best+1e-5:
            best=float(score); be=ep; stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(state); model.eval()
    return model,be,best

def calc(y,p):
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "per_class":classification_report(y,p,target_names=CLASSES,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(y,p).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=INPUT)
    ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--model",choices=["tcn","cnn_gru"],default="tcn")
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--patience",type=int,default=10); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--lr",type=float,default=1e-3)
    a=ap.parse_args(); seed(a.seed); a.out.mkdir(parents=True,exist_ok=True); start=time.time()
    X,y,g=load_data(a.input)
    m,s=fit_std(X[:min(8,len(X))]); assert m.shape==(3,) and s.shape==(3,)
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=a.seed); yt=[]; yp=[]; folds=[]
    for fold,(tri,tei) in enumerate(outer.split(X,y,g),1):
        inner=StratifiedGroupKFold(4,shuffle=True,random_state=a.seed+fold)
        ri,vi=next(inner.split(X[tri],y[tri],g[tri])); tr=tri[ri]; va=tri[vi]
        m,s=fit_std(X[tr]); xtr=apply_std(X[tr],m,s); xv=apply_std(X[va],m,s); xt=apply_std(X[tei],m,s)
        model,ep,best=train(xtr,y[tr],xv,y[va],a.model,a.seed+fold,a.epochs,a.batch_size,a.patience,a.lr)
        dev=next(model.parameters()).device
        with torch.no_grad(): p=model(torch.from_numpy(xt).to(dev)).argmax(1).cpu().numpy()
        fm=calc(y[tei],p); fm.update(fold=fold,best_epoch=ep,inner_best_macro_f1=best,test_subjects=sorted({str(v) for v in g[tei]}))
        folds.append(fm); yt.extend(y[tei]); yp.extend(p)
        print(f"Fold {fold}: acc={fm['accuracy']:.4f} bal_acc={fm['balanced_accuracy']:.4f} macro_f1={fm['macro_f1']:.4f} epoch={ep}")
    pooled=calc(np.array(yt),np.array(yp))
    # Deployment model: fit after evaluation, with grouped validation only for early stopping.
    m,s=fit_std(X); XA=apply_std(X,m,s); fin=StratifiedGroupKFold(5,shuffle=True,random_state=a.seed+1000)
    ri,vi=next(fin.split(XA,y,g))
    model,ep,best=train(XA[ri],y[ri],XA[vi],y[vi],a.model,a.seed+1000,a.epochs,a.batch_size,a.patience,a.lr)
    torch.save({"model_name":a.model,"classes":CLASSES,"input_shape":[3,40],"mean":m.tolist(),"std":s.tolist(),"state_dict":model.state_dict()},a.out/"activity_dl_v2_model.pt")
    report={"experiment":"bits2_activity_dl_v2","model":a.model,"input":str(a.input),"window_shape":list(X.shape[1:]),"samples":len(X),"subjects":len(np.unique(g)),
            "classes":CLASSES,"method":{"outer":"5-fold StratifiedGroupKFold by subject","normalization":"training-only per-channel mean/std","early_stopping":"inner grouped validation","outer_test_used_for_selection":False},
            "outer_fold_metrics":folds,"pooled_outer_metrics":pooled,"final_deployment":{"best_epoch":ep,"inner_best_macro_f1":best,"model_path":str(a.out/"activity_dl_v2_model.pt")},
            "seed":a.seed,"elapsed_seconds":time.time()-start}
    (a.out/"activity_dl_v2_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\nPOOLED: accuracy={pooled['accuracy']:.4%} balanced={pooled['balanced_accuracy']:.4%} macro_f1={pooled['macro_f1']:.4%}")
    print(f"Report: {a.out/'activity_dl_v2_report.json'}")

if __name__=="__main__": main()
