from pathlib import Path
import sys, argparse, json, random
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA = REPO_ROOT / "datasets" / "processed" / "pamap2" / "stationary_accgyro_windows.npz"
DEFAULT_OUT = REPO_ROOT / "models" / "pamap2_stationary_v1"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class TinyCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, n_classes))
    def forward(self, x):
        return self.head(self.net(x))

def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y,p)),
        "balanced_accuracy": float(balanced_accuracy_score(y,p)),
        "macro_f1": float(f1_score(y,p,average="macro")),
        "per_class_f1": f1_score(y,p,average=None).tolist(),
        "confusion_matrix": confusion_matrix(y,p).tolist(),
    }

def standardize(train, val):
    mu = train.mean(axis=(0,2), keepdims=True)
    sd = train.std(axis=(0,2), keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (train-mu)/sd, (val-mu)/sd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    seed_all(args.seed)

    d=np.load(args.data, allow_pickle=True)
    for k in ("X","y","groups"):
        if k not in d: raise RuntimeError(f"Missing NPZ key: {k}")
    X=d["X"].astype(np.float32); y=d["y"].astype(np.int64); groups=d["groups"].astype(str)
    if X.ndim != 3 or X.shape[1] != 6:
        raise RuntimeError(f"Expected X=(N,6,T), got {X.shape}")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    all_y=[]; all_p=[]; folds=[]

    for fold,(tr,te) in enumerate(cv.split(X,y,groups),1):
        # Inner subject-grouped validation for early stopping.
        inner=StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed+fold)
        itr,iva=next(inner.split(X[tr], y[tr], groups[tr]))
        tr_idx, va_idx = tr[itr], tr[iva]

        Xtr,Xva=standardize(X[tr_idx],X[va_idx])
        Xt=(X[te]-X[tr_idx].mean(axis=(0,2),keepdims=True))
        sd=X[tr_idx].std(axis=(0,2),keepdims=True); sd=np.where(sd<1e-6,1.0,sd)
        Xt=Xt/sd

        model=TinyCNN().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        loss_fn=nn.CrossEntropyLoss()
        best_state=None; best=-1; stale=0

        tx=torch.tensor(Xtr); ty=torch.tensor(y[tr_idx])
        vx=torch.tensor(Xva); vy=torch.tensor(y[va_idx])
        for ep in range(1,args.epochs+1):
            model.train()
            perm=torch.randperm(len(tx))
            for i in range(0,len(tx),128):
                b=perm[i:i+128]
                xb=tx[b].to(device); yb=ty[b].to(device)
                opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred=model(vx.to(device)).argmax(1).cpu().numpy()
            score=f1_score(y[va_idx],pred,average="macro")
            if score>best:
                best=score; stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_ep=ep
            else:
                stale+=1
                if stale>=args.patience: break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred=model(torch.tensor(Xt).to(device)).argmax(1).cpu().numpy()
        m=metrics(y[te],pred)
        m["fold"]=fold; m["best_epoch"]=best_ep; m["inner_best_macro_f1"]=float(best)
        folds.append(m); all_y.extend(y[te]); all_p.extend(pred)

        print(f"Fold {fold}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f} epoch={best_ep}")

    pooled=metrics(np.asarray(all_y),np.asarray(all_p))
    report={"experiment":"pamap2_stationary_accgyro_v1","model":"tiny_cnn","data":str(args.data),
            "window_shape":list(X.shape[1:]),"samples":int(len(X)),"subjects":int(len(set(groups))),
            "classes":["LYING","SITTING","STANDING"],"channels":["ax","ay","az","gx","gy","gz"],
            "method":{"outer":"5-fold StratifiedGroupKFold by subject","normalization":"training-only per-channel","early_stopping":"inner grouped validation","outer_test_used_for_selection":False},
            "outer_fold_metrics":folds,"pooled_outer_metrics":pooled,"seed":args.seed}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"pamap2_stationary_v1_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nPOOLED:", json.dumps(pooled,indent=2))
    print("Report:", args.out/"pamap2_stationary_v1_report.json")

if __name__=="__main__":
    main()
