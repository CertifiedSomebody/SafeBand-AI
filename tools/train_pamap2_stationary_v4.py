"""
SafeBand AI — PAMAP2 Stationary V4 FINAL MODEL SUITE

Models:
  1. RBF-SVM on 94 handcrafted features
  2. ANN/MLP on the same 94 features
  3. Tiny 1D CNN on raw 6x200 windows
  4. BiLSTM on raw sequences
  5. BiGRU on raw sequences

Fair protocol:
  - identical 5-fold StratifiedGroupKFold by subject
  - seed 42
  - outer test subjects are never used for selection
  - SVM/ANN use inner grouped CV
  - neural models use an inner grouped validation split for early stopping
  - normalization is fitted only on outer training data
  - Subject 5 is retained
"""
from pathlib import Path
import sys, argparse, json, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from tools.pamap2_stationary_v4_common import load_dataset,handcrafted_features,standardize_features,seed_everything,CLASSES,DATA,OUT

def metrics(y,p,g,fold,selected=None,inner=None):
    return {
      "fold":fold,
      "accuracy":float(accuracy_score(y,p)),
      "balanced_accuracy":float(balanced_accuracy_score(y,p)),
      "macro_f1":float(f1_score(y,p,average="macro")),
      "per_class_f1":f1_score(y,p,average=None,labels=[0,1,2]).tolist(),
      "confusion_matrix":confusion_matrix(y,p,labels=[0,1,2]).tolist(),
      "test_subjects":sorted(np.unique(g).tolist()),
      "test_samples":int(len(y)),
      "selected":selected,
      "inner_best_macro_f1":None if inner is None else float(inner)
    }

def pooled(yt,yp):
    return {
      "accuracy":float(accuracy_score(yt,yp)),
      "balanced_accuracy":float(balanced_accuracy_score(yt,yp)),
      "macro_f1":float(f1_score(yt,yp,average="macro")),
      "per_class_f1":f1_score(yt,yp,average=None,labels=[0,1,2]).tolist(),
      "confusion_matrix":confusion_matrix(yt,yp,labels=[0,1,2]).tolist()
    }

def svm_run(F,y,g,folds,seed):
    rows=[]; yt=[]; yp=[]
    grid={"svc__C":[0.5,1,2,4,8],"svc__gamma":["scale",.001,.003,.01,.03,.1]}
    for k,(tr,te) in enumerate(folds,1):
        pipe=Pipeline([("scale",StandardScaler()),("svc",SVC(kernel="rbf",class_weight="balanced"))])
        cv=StratifiedGroupKFold(4,shuffle=True,random_state=seed+k)
        splits=list(cv.split(F[tr],y[tr],g[tr]))
        gs=GridSearchCV(pipe,grid,scoring="f1_macro",cv=splits,n_jobs=-1)
        gs.fit(F[tr],y[tr]); p=gs.predict(F[te])
        yt.extend(y[te]);yp.extend(p);rows.append(metrics(y[te],p,g[te],k,gs.best_params_,gs.best_score_))
    return rows,np.asarray(yt),np.asarray(yp)

def ann_run(F,y,g,folds,seed):
    rows=[];yt=[];yp=[]; configs=[(64,),(128,64),(128,64,32)]
    for k,(tr,te) in enumerate(folds,1):
        Xtr,Xte=standardize_features(F[tr],F[te])
        cv=StratifiedGroupKFold(4,shuffle=True,random_state=seed+k)
        best=(-1,None)
        for cfg in configs:
            vals=[]
            for a,b in cv.split(Xtr,y[tr],g[tr]):
                m=MLPClassifier(hidden_layer_sizes=cfg,solver="adam",activation="relu",
                    alpha=1e-4,batch_size=64,learning_rate_init=1e-3,max_iter=250,
                    early_stopping=True,validation_fraction=.15,n_iter_no_change=15,
                    random_state=seed+k)
                m.fit(Xtr[a],y[tr][a])
                vals.append(f1_score(y[tr][b],m.predict(Xtr[b]),average="macro"))
            s=float(np.mean(vals))
            if s>best[0]: best=(s,cfg)
        m=MLPClassifier(hidden_layer_sizes=best[1],solver="adam",activation="relu",
            alpha=1e-4,batch_size=64,learning_rate_init=1e-3,max_iter=300,
            early_stopping=True,validation_fraction=.15,n_iter_no_change=20,random_state=seed+k)
        m.fit(Xtr,y[tr]);p=m.predict(Xte)
        yt.extend(y[te]);yp.extend(p);rows.append(metrics(y[te],p,g[te],k,{"hidden_layers":best[1]},best[0]))
    return rows,np.asarray(yt),np.asarray(yp)

def neural_run(X,y,g,folds,kind,epochs,seed):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader,TensorDataset

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.body=nn.Sequential(
              nn.Conv1d(6,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),
              nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),
              nn.AdaptiveAvgPool1d(1))
            self.fc=nn.Linear(64,3)
        def forward(self,x): return self.fc(self.body(x).squeeze(-1))

    class RNN(nn.Module):
        def __init__(self,cell):
            super().__init__()
            C=nn.LSTM if cell=="lstm" else nn.GRU
            self.r=C(6,64,num_layers=2,batch_first=True,bidirectional=True,dropout=.2)
            self.fc=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,3))
        def forward(self,x):
            o,_=self.r(x.permute(0,2,1).contiguous())
            return self.fc(o[:,-1,:])

    def train_model(model,Xtr,ytr,Xva,yva,max_epochs,seed):
        torch.manual_seed(seed);np.random.seed(seed)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        dl=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),128,shuffle=True)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
        loss=nn.CrossEntropyLoss()
        best=-1;state=None;wait=0;best_ep=1
        for ep in range(1,max_epochs+1):
            model.train()
            for xb,yb in dl:
                xb,yb=xb.to(device),yb.to(device);opt.zero_grad()
                loss(model(xb),yb).backward();opt.step()
            model.eval();pred=[]
            with torch.no_grad():
                for i in range(0,len(Xva),256):
                    pred.extend(model(torch.tensor(Xva[i:i+256]).to(device)).argmax(1).cpu().numpy())
            s=f1_score(yva,np.asarray(pred),average="macro")
            if s>best:
                best=s;best_ep=ep;wait=0
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: wait+=1
            if wait>=10: break
        model.load_state_dict(state)
        return model,best,best_ep,device

    rows=[];yt=[];yp=[]
    for k,(tr,te) in enumerate(folds,1):
        inner=StratifiedGroupKFold(4,shuffle=True,random_state=seed+k)
        a,b=next(inner.split(X[tr],y[tr],g[tr]))
        mu=X[tr][a].mean(axis=(0,2),keepdims=True);sd=X[tr][a].std(axis=(0,2),keepdims=True)
        sd=np.where(sd<1e-6,1,sd)
        Xtr=((X[tr][a]-mu)/sd).astype(np.float32)
        Xva=((X[tr][b]-mu)/sd).astype(np.float32)
        if kind=="cnn": m=CNN()
        else: m=RNN(kind)
        m,best,best_ep,device=train_model(m,Xtr,y[tr][a],Xva,y[tr][b],epochs,seed+k)
        # Final refit on ALL outer training subjects for exactly the selected epoch count.
        Xall=((X[tr]-mu)/sd).astype(np.float32)
        Xtest=((X[te]-mu)/sd).astype(np.float32)
        final=CNN() if kind=="cnn" else RNN(kind)
        final=final.to(device)
        dl=DataLoader(TensorDataset(torch.tensor(Xall),torch.tensor(y[tr])),128,shuffle=True)
        opt=torch.optim.Adam(final.parameters(),lr=1e-3,weight_decay=1e-4);loss=nn.CrossEntropyLoss()
        for _ in range(best_ep):
            final.train()
            for xb,yb in dl:
                xb,yb=xb.to(device),yb.to(device);opt.zero_grad();loss(final(xb),yb).backward();opt.step()
        final.eval();p=[]
        with torch.no_grad():
            for i in range(0,len(Xtest),256):
                p.extend(final(torch.tensor(Xtest[i:i+256]).to(device)).argmax(1).cpu().numpy())
        p=np.asarray(p);yt.extend(y[te]);yp.extend(p)
        rows.append(metrics(y[te],p,g[te],k,{"best_epoch":best_ep,"inner_macro_f1":best},best))
    return rows,np.asarray(yt),np.asarray(yp)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=DATA)
    ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--epochs",type=int,default=35)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    seed_everything(args.seed)
    X,y,g=load_dataset(args.data)
    subjects=sorted(np.unique(g),key=lambda s:(len(s),s))
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=args.seed)
    folds=list(outer.split(X,y,g))
    out=args.out if args.out.is_absolute() else ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    F=handcrafted_features(X)
    np.save(out/"v4_features.npy",F)
    results={}
    jobs=[
      ("rbf_svm",lambda:svm_run(F,y,g,folds,args.seed)),
      ("ann_mlp",lambda:ann_run(F,y,g,folds,args.seed)),
      ("cnn",lambda:neural_run(X,y,g,folds,"cnn",args.epochs,args.seed)),
      ("bilstm",lambda:neural_run(X,y,g,folds,"lstm",args.epochs,args.seed)),
      ("bigru",lambda:neural_run(X,y,g,folds,"gru",args.epochs,args.seed))
    ]
    t=time.time()
    for name,job in jobs:
        print(f"\n===== {name.upper()} =====",flush=True)
        rows,yt,yp=job()
        results[name]={"pooled":pooled(yt,yp),"folds":rows}
        q=results[name]["pooled"]
        print(f"Accuracy {q['accuracy']*100:.2f}% | Balanced {q['balanced_accuracy']*100:.2f}% | Macro-F1 {q['macro_f1']*100:.2f}%",flush=True)
        print("Class F1:",[round(v,4) for v in q["per_class_f1"]],flush=True)
    report={
      "experiment":"pamap2_stationary_model_suite_v4",
      "seed":args.seed,"samples":len(y),"subjects":subjects,"classes":CLASSES,
      "input_shape":list(X.shape),"feature_count":int(F.shape[1]),
      "validation":{"outer":"5-fold StratifiedGroupKFold by subject",
                    "inner":"4-fold grouped selection / early stopping",
                    "outer_test_used_for_selection":False,
                    "normalization":"training-only"},
      "models":results,"elapsed_seconds":time.time()-t,
      "notes":["Subject 5 retained.","Same outer folds for every model.",
               "No raw/processed dataset modification.","Compare pooled metrics AND fold stability."]}
    path=out/"model_suite_v4_report.json";path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\nREPORT: {path}")
if __name__=="__main__": main()
