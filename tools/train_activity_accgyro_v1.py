from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/"datasets/processed/bits2/activity_accgyro_windows.npz"
OUT=ROOT/"models/activity_accgyro_v1"
CLASSES=["RESTING","SITTING","WALKING","RUNNING"]

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load(p):
    d=np.load(p,allow_pickle=True)
    for k in ["X","y","groups"]:
        if k not in d: raise KeyError(f"NPZ missing required key '{k}'. Keys={list(d.keys())}")
    X=np.asarray(d["X"],np.float32); y=np.asarray(d["y"],np.int64); g=np.asarray(d["groups"]).astype(str)
    if X.ndim!=3 or X.shape[1]!=6: raise ValueError(f"Expected X=(N,6,T), got {X.shape}")
    if X.shape[2]!=40: raise ValueError(f"Expected 40 samples/window, got {X.shape}")
    if len(X)!=len(y) or len(X)!=len(g): raise ValueError("X/y/groups length mismatch")
    if sorted(np.unique(y).tolist())!=[0,1,2,3]: raise ValueError(f"Expected labels 0..3, got {np.unique(y)}")
    return X,y,g

def fit_std(X):
    if X.ndim!=3 or X.shape[1]!=6: raise ValueError(f"fit_std expected (N,6,T), got {X.shape}")
    m=X.mean(axis=(0,2),dtype=np.float64).astype(np.float32)
    s=np.maximum(X.std(axis=(0,2),dtype=np.float64),1e-6).astype(np.float32)
    return m,s

def apply_std(X,m,s): return ((X-m[None,:,None])/s[None,:,None]).astype(np.float32)

class IMUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv1d(6,32,5,padding=2),nn.BatchNorm1d(32),nn.GELU(),
            nn.Conv1d(32,48,5,padding=2),nn.BatchNorm1d(48),nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48,64,3,padding=1),nn.BatchNorm1d(64),nn.GELU(),
            nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.GELU(),
            nn.Dropout(.10))
        self.h=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(64,48),nn.GELU(),nn.Dropout(.18),nn.Linear(48,4))
    def forward(self,x): return self.h(self.f(x))

def train(Xtr,ytr,Xv,yv,seedv,epochs,bs,patience,lr):
    seed(seedv); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=IMUCNN().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    loss=nn.CrossEntropyLoss(label_smoothing=.03)
    dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=bs,shuffle=True)
    vx=torch.from_numpy(Xv).to(dev); best=-1; stale=0; state=None; best_ep=0
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(set_to_none=True)
            z=loss(model(xb),yb); z.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
        model.eval()
        with torch.no_grad(): p=model(vx).argmax(1).cpu().numpy()
        score=f1_score(yv,p,average="macro",zero_division=0)
        if score>best+1e-5:
            best=float(score); best_ep=ep; stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=patience: break
    if state is None: raise RuntimeError("Early-stopping state was never recorded")
    model.load_state_dict(state); model.eval()
    return model,best_ep,best

def metric(y,p):
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "per_class":classification_report(y,p,target_names=CLASSES,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(y,p).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=DEFAULT); ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--patience",type=int,default=10); ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); seed(a.seed); a.out.mkdir(parents=True,exist_ok=True); start=time.time()
    X,y,g=load(a.input)
    # Preflight catches the exact class of tensor-axis mistakes before CV.
    m,s=fit_std(X[:min(8,len(X))]); z=apply_std(X[:min(8,len(X))],m,s)
    assert z.shape==X[:min(8,len(X))].shape and m.shape==(6,) and s.shape==(6,)
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=a.seed)
    yt=[];yp=[];folds=[]
    for fold,(tri,tei) in enumerate(outer.split(X,y,g),1):
        inner=StratifiedGroupKFold(4,shuffle=True,random_state=a.seed+fold)
        ri,vi=next(inner.split(X[tri],y[tri],g[tri])); tr=tri[ri]; va=tri[vi]
        m,s=fit_std(X[tr]); xtr=apply_std(X[tr],m,s); xv=apply_std(X[va],m,s); xt=apply_std(X[tei],m,s)
        model,ep,best=train(xtr,y[tr],xv,y[va],a.seed+fold,a.epochs,a.batch_size,a.patience,a.lr)
        dev=next(model.parameters()).device
        with torch.no_grad(): p=model(torch.from_numpy(xt).to(dev)).argmax(1).cpu().numpy()
        fm=metric(y[tei],p); fm.update(fold=fold,best_epoch=ep,inner_best_macro_f1=best,test_subjects=sorted(set(g[tei])))
        folds.append(fm); yt.extend(y[tei]);yp.extend(p)
        print(f"Fold {fold}: acc={fm['accuracy']:.4f} bal_acc={fm['balanced_accuracy']:.4f} macro_f1={fm['macro_f1']:.4f} epoch={ep}")
    pooled=metric(np.asarray(yt),np.asarray(yp))
    report={"experiment":"bits2_activity_accgyro_v1","model":"tiny_cnn_6ch","input":str(a.input),"window_shape":list(X.shape[1:]),
            "samples":len(X),"subjects":len(np.unique(g)),"classes":CLASSES,
            "method":{"outer":"5-fold StratifiedGroupKFold by subject","normalization":"training-only per-channel mean/std",
                      "early_stopping":"inner grouped validation","outer_test_used_for_selection":False},
            "channels":["ax","ay","az","gx","gy","gz"],"outer_fold_metrics":folds,"pooled_outer_metrics":pooled,
            "seed":a.seed,"elapsed_seconds":time.time()-start}
    (a.out/"activity_accgyro_v1_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\\nPOOLED: accuracy={pooled['accuracy']:.4%} balanced={pooled['balanced_accuracy']:.4%} macro_f1={pooled['macro_f1']:.4%}")
    print("Report:",a.out/"activity_accgyro_v1_report.json")

if __name__=="__main__": main()
