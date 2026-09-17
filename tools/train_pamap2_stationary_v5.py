from pathlib import Path
import sys, json, time, argparse
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from pamap2_stationary_v5_common import *

def metrics(y,p):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,labels=range(N_CLASSES),zero_division=0)
    return {
        "accuracy": float(accuracy_score(y,p)),
        "balanced_accuracy": float(balanced_accuracy_score(y,p)),
        "macro_f1": float(f1_score(y,p,average="macro")),
        "per_class": [{"class":CLASSES[i],"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f1[i])} for i in range(N_CLASSES)],
        "confusion_matrix": confusion_matrix(y,p,labels=range(N_CLASSES)).tolist(),
    }

def svm_run(F,y,g,folds):
    rows=[]; yp=np.full_like(y,-1)
    for k,(tr,te) in enumerate(folds,1):
        inner=grouped_inner_splits(F[tr],y[tr],g[tr],4)
        pipe=Pipeline([("scale",StandardScaler()),("svc",SVC(kernel="rbf",class_weight="balanced",random_state=SEED))])
        grid={"svc__C":[0.5,1,2,4,8],"svc__gamma":["scale",0.001,0.01,0.1]}
        gs=GridSearchCV(pipe,grid,scoring="f1_macro",cv=inner,n_jobs=-1,refit=True)
        gs.fit(F[tr],y[tr])
        yp[te]=gs.predict(F[te])
        rows.append({"fold":k,"test_groups":sorted(set(g[te].tolist())),"best_params":gs.best_params_,"inner_macro_f1":float(gs.best_score_)})
        print(f"Fold {k}: {metrics(y[te],yp[te])['macro_f1']:.4f} | {gs.best_params_}")
    return rows,yp

def mlp_run(F,y,g,folds):
    rows=[]; yp=np.full_like(y,-1)
    configs=[(64,),(128,64),(128,64,32)]
    for k,(tr,te) in enumerate(folds,1):
        inner=grouped_inner_splits(F[tr],y[tr],g[tr],4)
        scores=[]
        for cfg in configs:
            vals=[]
            for itr,iva in inner:
                model=Pipeline([("scale",StandardScaler()),("mlp",MLPClassifier(hidden_layer_sizes=cfg,activation="relu",solver="adam",alpha=1e-4,max_iter=250,early_stopping=False,random_state=SEED))])
                model.fit(F[tr][itr],y[tr][itr])
                vals.append(f1_score(y[tr][iva],model.predict(F[tr][iva]),average="macro"))
            scores.append(float(np.mean(vals)))
        best_idx=int(np.argmax(scores)); cfg=configs[best_idx]
        # Select epoch count without touching outer test: deterministic grouped inner CV.
        epoch_grid=[50,100,150,200]
        ep_scores=[]
        for ep in epoch_grid:
            vals=[]
            for itr,iva in inner:
                model=Pipeline([("scale",StandardScaler()),("mlp",MLPClassifier(hidden_layer_sizes=cfg,activation="relu",solver="adam",alpha=1e-4,max_iter=ep,early_stopping=False,random_state=SEED))])
                model.fit(F[tr][itr],y[tr][itr])
                vals.append(f1_score(y[tr][iva],model.predict(F[tr][iva]),average="macro"))
            ep_scores.append(float(np.mean(vals)))
        ep=epoch_grid[int(np.argmax(ep_scores))]
        final=Pipeline([("scale",StandardScaler()),("mlp",MLPClassifier(hidden_layer_sizes=cfg,activation="relu",solver="adam",alpha=1e-4,max_iter=ep,early_stopping=False,random_state=SEED))])
        final.fit(F[tr],y[tr]); yp[te]=final.predict(F[te])
        rows.append({"fold":k,"test_groups":sorted(set(g[te].tolist())),"hidden_layers":list(cfg),"inner_config_scores":scores,"selected_max_iter":ep,"inner_epoch_scores":ep_scores})
        print(f"Fold {k}: {metrics(y[te],yp[te])['macro_f1']:.4f} | MLP {cfg}, max_iter={ep}")
    return rows,yp

def torch_models():
    import torch
    import torch.nn as nn
    class CNN(nn.Module):
        def __init__(self):
            super().__init__(); self.net=nn.Sequential(nn.Conv1d(6,32,7,padding=3),nn.ReLU(),nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.fc=nn.Linear(64,3)
        def forward(self,x): return self.fc(self.net(x).squeeze(-1))
    class RNN(nn.Module):
        def __init__(self,kind):
            super().__init__(); R=getattr(nn,kind); self.r=R(6,48,batch_first=True,bidirectional=True); self.fc=nn.Linear(96,3)
        def forward(self,x): _,h=self.r(x); return self.fc(torch.cat([h[-2],h[-1]],1))
    return CNN,RNN

def neural_run(X,y,g,folds,kind,device,seed=SEED):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    CNN,RNN=torch_models()
    yp=np.full_like(y,-1); rows=[]
    torch.manual_seed(seed)
    for k,(tr,te) in enumerate(folds,1):
        inner=grouped_inner_splits(X[tr],y[tr],g[tr],4)
        candidates=[("small",16),("medium",32)]
        score_rows=[]
        for cname,epochs in candidates:
            vals=[]
            for itr,iva in inner:
                Xitr,Xiva,_,mean,std=standardize_train(X[tr][itr],X[tr][iva],None)
                model=CNN() if kind=="CNN" else RNN(kind)
                model.to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
                lossfn=torch.nn.CrossEntropyLoss()
                ds=TensorDataset(torch.from_numpy(Xitr),torch.from_numpy(y[tr][itr]))
                dl=DataLoader(ds,batch_size=128,shuffle=True)
                model.train()
                for _ in range(epochs):
                    for xb,yb in dl:
                        xb=xb.to(device); yb=yb.to(device)
                        if kind!="CNN": xb=xb.permute(0,2,1)
                        opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    xv=torch.from_numpy(Xiva).to(device)
                    if kind!="CNN": xv=xv.permute(0,2,1)
                    p=model(xv).argmax(1).cpu().numpy()
                vals.append(f1_score(y[tr][iva],p,average="macro"))
            score_rows.append({"candidate":cname,"epochs":epochs,"inner_macro_f1":float(np.mean(vals))})
        best=max(score_rows,key=lambda z:z["inner_macro_f1"])
        Xtr,Xte,_,mean,std=standardize_train(X[tr],X[te],None)
        model=CNN() if kind=="CNN" else RNN(kind); model.to(device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); lossfn=torch.nn.CrossEntropyLoss()
        dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(y[tr])),batch_size=128,shuffle=True)
        model.train()
        for _ in range(best["epochs"]):
            for xb,yb in dl:
                xb=xb.to(device); yb=yb.to(device)
                if kind!="CNN": xb=xb.permute(0,2,1)
                opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xt=torch.from_numpy(Xte).to(device)
            if kind!="CNN": xt=xt.permute(0,2,1)
            p=model(xt).argmax(1).cpu().numpy()
        yp[te]=p
        rows.append({"fold":k,"test_groups":sorted(set(g[te].tolist())),"selected_epochs":best["epochs"],"inner_candidates":score_rows})
        print(f"Fold {k}: {metrics(y[te],p)['macro_f1']:.4f} | {kind}, epochs={best['epochs']}")
    return rows,yp

def main():
    ap=argparse.ArgumentParser(description="SafeBand AI PAMAP2 Stationary V5 model-family benchmark.")
    ap.add_argument("--data",default=str(ROOT/"datasets/processed/pamap2/stationary_accgyro_windows.npz"))
    ap.add_argument("--out",default=str(ROOT/"models/pamap2_stationary_v5"))
    args=ap.parse_args()
    np.random.seed(SEED)
    X,y,g=load_data(args.data); folds=outer_splits(X,y,g)
    print(f"Dataset: {len(y)} windows | shape={X.shape} | subjects={sorted(set(g))}")
    F=window_features(X); print(f"Feature matrix: {F.shape}")
    results={}; timings={}
    jobs=[("RBF_SVM",lambda:svm_run(F,y,g,folds)),("ANN_MLP",lambda:mlp_run(F,y,g,folds))]
    for name,job in jobs:
        print(f"\n===== {name} ====="); t=time.perf_counter(); fr,p=job(); timings[name]=time.perf_counter()-t; results[name]={"folds":fr,"pooled":metrics(y,p)}
    try:
        import torch
        device="cuda" if torch.cuda.is_available() else "cpu"
        for name,kind in [("Tiny_CNN","CNN"),("BiLSTM","LSTM"),("BiGRU","GRU")]:
            print(f"\n===== {name} ====="); t=time.perf_counter(); fr,p=neural_run(X,y,g,folds,kind,device); timings[name]=time.perf_counter()-t; results[name]={"folds":fr,"pooled":metrics(y,p),"device":device}
    except ImportError:
        raise RuntimeError("PyTorch is required for Tiny_CNN/BiLSTM/BiGRU. Install it in the active venv.")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    report={"experiment":"pamap2_stationary_model_suite_v5","seed":SEED,"dataset":str(Path(args.data)),"samples":int(len(y)),"subjects":sorted(set(g.tolist())),"classes":CLASSES,"input_shape":list(X.shape),"feature_count":int(F.shape[1]),"validation":"5-fold StratifiedGroupKFold by PeopleId; all hyperparameter/epoch selection uses 4-fold grouped inner CV; outer test untouched; normalization fitted only on outer training data.","models":results,"timing_seconds":timings}
    (out/"model_suite_v5_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\n===== POOLED COMPARISON =====")
    for n,r in results.items():
        m=r["pooled"]; print(f"{n:12s} acc={m['accuracy']*100:.2f}% bal={m['balanced_accuracy']*100:.2f}% macroF1={m['macro_f1']*100:.2f}%")
    print(f"\nSaved: {out/'model_suite_v5_report.json'}")

if __name__=="__main__": main()
