from __future__ import annotations
import sys, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/"datasets/processed/bits2/activity_accgyro_windows.npz"
CLASSES=["RESTING","SITTING","WALKING","RUNNING"]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_npz(path):
    d=np.load(path,allow_pickle=True)
    required={"X","y","groups"}
    missing=required-set(d.files)
    if missing: raise KeyError(f"NPZ missing {sorted(missing)}; available={d.files}")
    X=np.asarray(d["X"],dtype=np.float32)
    y=np.asarray(d["y"],dtype=np.int64)
    g=np.asarray(d["groups"]).astype(str)
    if X.ndim!=3: raise ValueError(f"Expected 3-D X=(N,C,T), got {X.shape}")
    if X.shape[1]!=6: raise ValueError(f"Expected 6 channels ACC+GYRO, got {X.shape}")
    if X.shape[2]!=40: raise ValueError(f"Expected 40 samples/window, got {X.shape}")
    if not (len(X)==len(y)==len(g)): raise ValueError(f"Length mismatch X={len(X)}, y={len(y)}, groups={len(g)}")
    vals=sorted(np.unique(y).tolist())
    if vals != [0,1,2,3]: raise ValueError(f"Expected labels 0..3, got {vals}")
    return X,y,g

def fit_standardizer(X):
    if X.ndim!=3 or X.shape[1]!=6: raise ValueError(f"Bad training tensor {X.shape}")
    mean=X.mean(axis=(0,2),dtype=np.float64).astype(np.float32)
    std=np.maximum(X.std(axis=(0,2),dtype=np.float64),1e-6).astype(np.float32)
    return mean,std

def standardize(X,mean,std):
    return ((X-mean[None,:,None])/std[None,:,None]).astype(np.float32)

def metrics(y,p):
    return {
        "accuracy":float(accuracy_score(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "per_class":classification_report(y,p,target_names=CLASSES,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(y,p).tolist()
    }

def evaluate_model(model,X,y,device):
    model.eval()
    with torch.no_grad():
        p=model(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()
    return metrics(y,p)

def make_loaders(X,y,bs):
    return DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),batch_size=bs,shuffle=True)

class LSTMClassifier(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=.20):
        super().__init__()
        self.lstm=nn.LSTM(input_size=6,hidden_size=hidden,num_layers=layers,
                          batch_first=True,dropout=dropout if layers>1 else 0.0,
                          bidirectional=True)
        self.head=nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2,64),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(64,4))
    def forward(self,x):
        # Explicit conversion: (N,C,T) -> (N,T,C), no ambiguous transpose.
        x=x.permute(0,2,1).contiguous()
        z,_=self.lstm(x)
        return self.head(z[:,-1,:])

class TransformerClassifier(nn.Module):
    def __init__(self, d_model=64, heads=4, layers=2, ff=128, dropout=.15):
        super().__init__()
        self.proj=nn.Linear(6,d_model)
        self.pos=nn.Parameter(torch.zeros(1,40,d_model))
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=heads,
            dim_feedforward=ff,dropout=dropout,activation="gelu",
            batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc,num_layers=layers)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Sequential(nn.Linear(d_model,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,4))
    def forward(self,x):
        x=x.permute(0,2,1).contiguous()
        z=self.proj(x)+self.pos
        z=self.enc(z)
        z=self.norm(z.mean(dim=1))
        return self.head(z)

def train_one(model,Xtr,ytr,Xv,yv,seed,epochs,bs,patience,lr,weight_decay):
    seed_all(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    criterion=nn.CrossEntropyLoss(label_smoothing=.03)
    loader=make_loaders(Xtr,ytr,bs)
    vx=torch.from_numpy(Xv).to(device)
    best=-1.0; stale=0; best_state=None; best_epoch=0
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss=criterion(model(xb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0)
            opt.step()
        model.eval()
        with torch.no_grad(): p=model(vx).argmax(1).cpu().numpy()
        score=f1_score(yv,p,average="macro",zero_division=0)
        if score>best+1e-5:
            best=float(score); best_epoch=ep; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=patience: break
    if best_state is None: raise RuntimeError("No validation checkpoint was produced.")
    model.load_state_dict(best_state)
    return model,best_epoch,best

def run_cv(model_name,input_path,out_dir,epochs,patience,bs,lr,weight_decay,seed):
    import json,time
    start=time.time(); seed_all(seed)
    X,y,g=load_npz(input_path)
    # Shape preflight before expensive CV.
    mean,std=fit_standardizer(X[:min(8,len(X))])
    probe=standardize(X[:min(8,len(X))],mean,std)
    assert probe.shape==X[:min(8,len(X))].shape
    if model_name=="lstm":
        factory=lambda:LSTMClassifier()
    elif model_name=="transformer":
        factory=lambda:TransformerClassifier()
    else: raise ValueError(model_name)

    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    folds=[]; yt=[];yp=[]
    for fold,(tri,tei) in enumerate(outer.split(X,y,g),1):
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed+fold)
        ri,vi=next(inner.split(X[tri],y[tri],g[tri]))
        tr=tri[ri]; va=tri[vi]

        mean,std=fit_standardizer(X[tr])
        Xtr=standardize(X[tr],mean,std)
        Xv=standardize(X[va],mean,std)
        Xt=standardize(X[tei],mean,std)

        model,ep,best=train_one(factory(),Xtr,y[tr],Xv,y[va],
                                 seed+fold,epochs,bs,patience,lr,weight_decay)
        device=next(model.parameters()).device
        fm=evaluate_model(model,Xt,y[tei],device)
        fm.update(fold=fold,best_epoch=ep,inner_best_macro_f1=best,
                  test_subjects=sorted(set(g[tei])))
        folds.append(fm); yt.extend(y[tei]); yp.extend(
            np.asarray(evaluate_predictions(model,Xt,device),dtype=np.int64))
        print(f"Fold {fold}: acc={fm['accuracy']:.4f} bal_acc={fm['balanced_accuracy']:.4f} macro_f1={fm['macro_f1']:.4f} epoch={ep}")

    pooled=metrics(np.asarray(yt),np.asarray(yp))
    out_dir.mkdir(parents=True,exist_ok=True)
    report={
        "experiment":f"bits2_activity_accgyro_{model_name}_v2",
        "model":model_name,
        "input":str(input_path),
        "window_shape":list(X.shape[1:]),
        "samples":int(len(X)),
        "subjects":int(len(np.unique(g))),
        "classes":CLASSES,
        "channels":["ax","ay","az","gx","gy","gz"],
        "method":{"outer":"5-fold StratifiedGroupKFold by subject",
                  "normalization":"training-only per-channel mean/std",
                  "early_stopping":"inner grouped validation",
                  "outer_test_used_for_selection":False},
        "outer_fold_metrics":folds,
        "pooled_outer_metrics":pooled,
        "seed":seed,
        "elapsed_seconds":time.time()-start
    }
    path=out_dir/f"activity_accgyro_{model_name}_v2_report.json"
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\nPOOLED: accuracy={pooled['accuracy']:.4%} balanced={pooled['balanced_accuracy']:.4%} macro_f1={pooled['macro_f1']:.4%}")
    print("Report:",path)

def evaluate_predictions(model,X,device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=DEFAULT)
    ap.add_argument("--out",type=Path,default=ROOT/"models/activity_accgyro_v2")
    ap.add_argument("--epochs",type=int,default=70)
    ap.add_argument("--patience",type=int,default=12)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--model",choices=["lstm","transformer"],required=True)
    a=ap.parse_args()
    run_cv(a.model,a.input,a.out,a.epochs,a.patience,a.batch_size,a.lr,a.weight_decay,a.seed)
