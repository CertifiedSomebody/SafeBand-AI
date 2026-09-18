from pathlib import Path
import sys, argparse, json, random
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

LABELS=["breath","cough","crying","laugh","screaming","sneeze","yawn"]

class SmallLogMelCNN(nn.Module):
    def __init__(self,nc=7):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(.10),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(.15),
            nn.Conv2d(64,96,3,padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(128,nc))
    def forward(self,x): return self.fc(self.net(x))

def metric_dict(y,p):
    return {
        "accuracy":float(accuracy_score(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,labels=list(range(7)),average="macro")),
        "confusion_matrix":confusion_matrix(y,p,labels=list(range(7))).tolist()
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="datasets/processed/nonspeech7k/logmel_v3.npz")
    ap.add_argument("--out",default="reports/nonspeech7k_audio_v3")
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--patience",type=int,default=6)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    d=np.load(ROOT/args.input,allow_pickle=False)
    X=d["X"].astype(np.float32); y=d["y"].astype(np.int64); groups=d["groups"].astype(str)
    if X.shape!=(6283,1,64,128): raise RuntimeError(f"Unexpected cache shape: {X.shape}")
    if len(set(groups))!=1899: raise RuntimeError("Unexpected group count")
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    report={"protocol":"5-fold StratifiedGroupKFold grouped by file_id",
            "samples":len(y),"groups":len(set(groups)),"input_shape":list(X.shape[1:]),
            "model":"SmallLogMelCNN","seed":args.seed,"device":device,"folds":[]}
    print(f"[INFO] device: {device}")
    for fold,(outer_tr,outer_te) in enumerate(cv.split(X,y,groups),1):
        # Validation split is grouped and derived only from outer training data.
        gss=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=args.seed+fold)
        rel_tr,rel_val=next(gss.split(outer_tr,y[outer_tr],groups[outer_tr]))
        train_idx=outer_tr[rel_tr]; val_idx=outer_tr[rel_val]
        mean=X[train_idx].mean(dtype=np.float64)
        std=X[train_idx].std(dtype=np.float64)+1e-6
        Xt=((X[train_idx]-mean)/std).astype(np.float32)
        Xv=((X[val_idx]-mean)/std).astype(np.float32)
        Xe=((X[outer_te]-mean)/std).astype(np.float32)
        weights=compute_class_weight(class_weight="balanced",classes=np.arange(7),y=y[train_idx]).astype(np.float32)
        model=SmallLogMelCNN().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
        loader=DataLoader(TensorDataset(torch.from_numpy(Xt),torch.from_numpy(y[train_idx])),
                          batch_size=args.batch_size,shuffle=True,num_workers=0)
        best=-1.0; bad=0; best_epoch=0; best_state=None
        for epoch in range(1,args.epochs+1):
            model.train()
            for xb,yb in loader:
                xb,yb=xb.to(device),yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss=loss_fn(model(xb),yb)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vp=model(torch.from_numpy(Xv).to(device)).argmax(1).cpu().numpy()
            vf=f1_score(y[val_idx],vp,labels=list(range(7)),average="macro")
            if vf>best:
                best=float(vf); best_epoch=epoch; bad=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad+=1
                if bad>=args.patience: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pred=model(torch.from_numpy(Xe).to(device)).argmax(1).cpu().numpy()
        m=metric_dict(y[outer_te],pred)
        m.update({"fold":fold,"best_epoch":best_epoch,"val_macro_f1":best,
                  "test_samples":int(len(outer_te))})
        report["folds"].append(m)
        print(f"[FOLD {fold}] acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f} epoch={best_epoch}")
    report["mean_accuracy"]=float(np.mean([f["accuracy"] for f in report["folds"]]))
    report["mean_balanced_accuracy"]=float(np.mean([f["balanced_accuracy"] for f in report["folds"]]))
    report["mean_macro_f1"]=float(np.mean([f["macro_f1"] for f in report["folds"]]))
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    (out/"cnn_v3_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["mean_accuracy","mean_balanced_accuracy","mean_macro_f1"]},indent=2))
    print(f"[PASS] report -> {out/'cnn_v3_report.json'}")

if __name__=="__main__": main()
