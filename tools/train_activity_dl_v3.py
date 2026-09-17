from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "datasets/processed/bits2/activity_raw_windows.npz"
FEAT = ROOT / "datasets/processed/bits2/activity_windows.csv"
OUT = ROOT / "models/activity_dl_v3"
CLASSES = ["RESTING", "SITTING", "WALKING", "RUNNING"]

# The locked handcrafted ACC contract from the existing BITS2 activity pipeline.
FEATURES = [
    "ax_mean","ax_std","ax_min","ax_max","ax_rms",
    "ay_mean","ay_std","ay_min","ay_max","ay_rms",
    "az_mean","az_std","az_min","az_max","az_rms",
    "acc_mag_mean","acc_mag_std","acc_mag_min","acc_mag_max","acc_mag_rms",
    "acc_mag_p10","acc_mag_p25","acc_mag_p50","acc_mag_p75","acc_mag_p90",
    "acc_sma","acc_range","acc_jerk_mean","acc_jerk_std","acc_jerk_max",
    "acc_diff_energy","acc_zero_crossings",
    "acc_x_y_corr","acc_x_z_corr","acc_y_z_corr",
    "acc_fft_low","acc_fft_mid","acc_fft_high",
]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_raw(path):
    d = np.load(path, allow_pickle=True)
    def pick(names):
        return next((k for k in names if k in d), None)
    xk = pick(["X","x","windows","data"])
    yk = pick(["y","labels","activity","target"])
    gk = pick(["groups","subjects","subject_id","group"])
    if not all((xk, yk, gk)):
        raise KeyError(f"Need X/y/groups. Keys={list(d.keys())}")
    X = np.asarray(d[xk], dtype=np.float32)
    yr = np.asarray(d[yk])
    groups = np.asarray(d[gk])
    if X.ndim != 3:
        raise ValueError(f"Expected 3-D raw windows, got {X.shape}")
    if X.shape[1:] == (3,40):
        pass
    elif X.shape[1:] == (40,3):
        X = np.transpose(X, (0,2,1))
    else:
        raise ValueError(f"Expected (N,3,40) or (N,40,3), got {X.shape}")
    if np.issubdtype(yr.dtype, np.number):
        y = yr.astype(np.int64)
    else:
        mp = {c:i for i,c in enumerate(CLASSES)}
        y = np.array([mp[str(v).upper()] for v in yr], dtype=np.int64)
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("Raw X/y/groups length mismatch")
    if sorted(np.unique(y).tolist()) != list(range(4)):
        raise ValueError(f"Expected labels 0..3, got {np.unique(y)}")
    return X, y, groups

def load_features(path, expected_n):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing handcrafted features: {missing}")
    label_col = next((c for c in ["label","activity","activity_label","class","target"] if c in df.columns), None)
    group_col = next((c for c in ["subject_id","subject","group","participant"] if c in df.columns), None)
    if label_col is None or group_col is None:
        raise ValueError(f"Could not identify label/group columns. Columns={list(df.columns)}")
    # The raw NPZ contract excludes FALL windows; the shared CSV may contain
    # those rows after the four activity classes, so keep the aligned subset.
    if label_col == "activity_label":
        df = df[df[label_col].astype(str).str.upper() != "FALL"].reset_index(drop=True)
    X = df[FEATURES].to_numpy(np.float32)
    yr = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()
    mp = {c:i for i,c in enumerate(CLASSES)}
    y = np.array([mp[str(v).upper()] if not isinstance(v,(int,np.integer)) else int(v) for v in yr], dtype=np.int64)
    if len(X) != expected_n:
        raise ValueError(
            f"Raw NPZ has {expected_n} rows but handcrafted CSV has {len(X)}. "
            "The two artifacts must be row-aligned."
        )
    if not np.array_equal(y, None):
        pass
    return X, y, groups

def fit_std(X):
    if X.ndim != 3 or X.shape[1] != 3:
        raise ValueError(f"Expected (N,3,T), got {X.shape}")
    m = X.mean(axis=(0,2), dtype=np.float64).astype(np.float32)
    s = np.maximum(X.std(axis=(0,2), dtype=np.float64), 1e-6).astype(np.float32)
    return m,s

def apply_std(X,m,s):
    return ((X-m[None,:,None])/s[None,:,None]).astype(np.float32)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3,32,5,padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32,48,5,padding=2), nn.BatchNorm1d(48), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48,64,3,padding=1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64,64,3,padding=1), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64,48), nn.GELU(), nn.Dropout(.18),
            nn.Linear(48,4)
        )
    def forward(self,x):
        return self.head(self.net(x))

def train_cnn(Xtr,ytr,Xv,yv,seed,epochs=50,patience=9,bs=64,lr=1e-3):
    seed_everything(seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TinyCNN().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    loss=nn.CrossEntropyLoss(label_smoothing=.03)
    dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=bs,shuffle=True)
    vx=torch.from_numpy(Xv).to(dev)
    best=-1; stale=0; state=None; best_ep=0
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev)
            opt.zero_grad(set_to_none=True)
            z=loss(model(xb),yb); z.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
        model.eval()
        with torch.no_grad():
            p=torch.softmax(model(vx),dim=1).cpu().numpy()
        score=f1_score(yv,p.argmax(1),average="macro",zero_division=0)
        if score>best+1e-5:
            best=float(score); best_ep=ep; stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(state); model.eval()
    return model,best_ep,best

def cnn_probs(model,X):
    dev=next(model.parameters()).device
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X).to(dev)),dim=1).cpu().numpy()

def metric(y,p):
    return {
        "accuracy":float(accuracy_score(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "per_class":classification_report(y,p,target_names=CLASSES,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(y,p).tolist()
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",type=Path,default=RAW)
    ap.add_argument("--features",type=Path,default=FEAT)
    ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--blend-grid",type=str,default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    a=ap.parse_args()
    seed_everything(a.seed); a.out.mkdir(parents=True,exist_ok=True); start=time.time()

    Xraw,y,g=load_raw(a.raw)
    Xfeat,yf,gf=load_features(a.features,len(Xraw))
    if not np.array_equal(y,yf):
        raise ValueError("Raw and handcrafted labels are not row-aligned.")
    if not np.array_equal(g.astype(str),gf.astype(str)):
        raise ValueError("Raw and handcrafted subject IDs are not row-aligned.")

    outer=StratifiedGroupKFold(5,shuffle=True,random_state=a.seed)
    weights=[float(x) for x in a.blend_grid.split(",")]
    pooled={k:[] for k in ["cnn","rf","blend"]}
    folds=[]

    for fold,(tri,tei) in enumerate(outer.split(Xraw,y,g),1):
        inner=StratifiedGroupKFold(4,shuffle=True,random_state=a.seed+fold)
        ri,vi=next(inner.split(Xraw[tri],y[tri],g[tri]))
        tr=tri[ri]; va=tri[vi]

        m,s=fit_std(Xraw[tr])
        xr_tr=apply_std(Xraw[tr],m,s); xr_va=apply_std(Xraw[va],m,s); xr_te=apply_std(Xraw[tei],m,s)
        cnn,ep,inner_cnn=train_cnn(xr_tr,y[tr],xr_va,y[va],a.seed+fold,a.epochs)

        # Handcrafted model is also selected only from training subjects.
        rf=ExtraTreesClassifier(
            n_estimators=500,max_features="sqrt",min_samples_leaf=2,
            class_weight="balanced",random_state=a.seed+fold,n_jobs=-1
        )
        rf.fit(Xfeat[tr],y[tr])
        pc_tr=cnn_probs(cnn,xr_va)
        pr_va=rf.predict_proba(Xfeat[va])
        # Select blend weight on inner validation only.
        best_w=None; best_score=-1
        for w in weights:
            pb=w*pc_tr+(1-w)*pr_va
            sc=f1_score(y[va],pb.argmax(1),average="macro",zero_division=0)
            if sc>best_score:
                best_score=float(sc); best_w=w

        pc=cnn_probs(cnn,xr_te)
        pr=rf.predict_proba(Xfeat[tei])
        pc_pred=pc.argmax(1); pr_pred=pr.argmax(1)
        pb=best_w*pc+(1-best_w)*pr
        pb_pred=pb.argmax(1)

        fm={"fold":fold,"best_epoch":ep,"inner_cnn_macro_f1":inner_cnn,
            "inner_blend_weight_cnn":best_w,
            "cnn":metric(y[tei],pc_pred),"rf":metric(y[tei],pr_pred),"blend":metric(y[tei],pb_pred),
            "test_subjects":sorted({str(v) for v in g[tei]})}
        folds.append(fm)
        for k,p in [("cnn",pc_pred),("rf",pr_pred),("blend",pb_pred)]:
            pooled[k][0:] = pooled[k]  # explicit no-op for readability
        pooled.setdefault("yt",[]).extend(y[tei].tolist())
        for k,p in [("cnn",pc_pred),("rf",pr_pred),("blend",pb_pred)]:
            pooled[k].extend(p.tolist())
        print(f"Fold {fold}: CNN={fm['cnn']['macro_f1']:.4f} RF={fm['rf']['macro_f1']:.4f} BLEND={fm['blend']['macro_f1']:.4f} w_cnn={best_w:.1f}")

    yt=np.array(pooled.pop("yt"))
    pooled_metrics={k:metric(yt,np.array(v)) for k,v in pooled.items()}

    report={
        "experiment":"bits2_activity_dl_v3_fusion",
        "purpose":"Complementary fusion of raw temporal ACC CNN and locked handcrafted ACC model",
        "samples":int(len(Xraw)),
        "subjects":int(len(np.unique(g))),
        "classes":CLASSES,
        "input_raw_shape":list(Xraw.shape[1:]),
        "handcrafted_features":FEATURES,
        "method":{
            "outer":"5-fold StratifiedGroupKFold by subject",
            "normalization":"CNN training-only per-channel mean/std",
            "blend_weight_selection":"inner grouped validation only",
            "outer_test_used_for_selection":False,
            "fusion":"probability-level weighted average"
        },
        "folds":folds,
        "pooled_outer_metrics":pooled_metrics,
        "seed":a.seed,
        "elapsed_seconds":time.time()-start
    }
    (a.out/"activity_dl_v3_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\n=== POOLED OUTER ===")
    for k,v in pooled_metrics.items():
        print(f"{k.upper():6s}: acc={v['accuracy']:.4%} bal={v['balanced_accuracy']:.4%} macro_f1={v['macro_f1']:.4%}")
    print(f"Report: {a.out/'activity_dl_v3_report.json'}")

if __name__=="__main__":
    main()
