from pathlib import Path
import sys, argparse, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

def safe_corr(a,b):
    sa,sb=np.std(a),np.std(b)
    if sa==0 or sb==0: return 0.0
    v=np.corrcoef(a,b)[0,1]
    return 0.0 if not np.isfinite(v) else float(v)

def features(X, hz=50.0):
    n, c, w = X.shape
    if c != 3: raise ValueError(f"Expected ACC shape (N,3,W), got {X.shape}")
    out=np.empty((n,38),dtype=np.float32)
    for i in range(n):
        ax,ay,az=X[i]
        mag=np.sqrt(ax*ax+ay*ay+az*az)
        vals=[]
        for a in (ax,ay,az):
            vals += [a.mean(),a.std(),a.min(),a.max(),np.sqrt(np.mean(a*a))]
        vals += [mag.mean(),mag.std(),mag.min(),mag.max(),np.sqrt(np.mean(mag*mag))]
        vals += list(np.percentile(mag,[10,25,50,75,90]))
        vals += [np.mean(np.abs(ax))+np.mean(np.abs(ay))+np.mean(np.abs(az)),
                 mag.max()-mag.min()]
        j=np.diff(mag)*hz
        vals += [np.mean(np.abs(j)),np.std(j),np.max(np.abs(j))]
        d=np.diff(mag)
        vals += [np.sum(d*d), np.sum(np.signbit(d[1:]) != np.signbit(d[:-1]))]
        vals += [safe_corr(ax,ay),safe_corr(ax,az),safe_corr(ay,az)]
        f=np.fft.rfftfreq(w,1/hz); p=np.abs(np.fft.rfft(mag-mag.mean()))**2
        bands=[(f>=0.3)&(f<3),(f>=3)&(f<8),(f>=8)&(f<20)]
        vals += [float(p[b].sum()) for b in bands]
        out[i]=vals
    return out

def eval_model(name, model, Xtr,ytr,Xte,yte,classes):
    t=time.time(); model.fit(Xtr,ytr); pred=model.predict(Xte)
    return {"accuracy":float(accuracy_score(yte,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(yte,pred)),
            "macro_f1":float(f1_score(yte,pred,average="macro")),
            "classes":classes,
            "elapsed_s":time.time()-t}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",required=True); ap.add_argument("--manifest",required=True)
    ap.add_argument("--out",required=True); args=ap.parse_args()
    z=np.load(args.npz); X=z["X"]; m=pd.read_csv(args.manifest)
    if len(X)!=len(m): raise ValueError("NPZ/manifest row mismatch")
    y=m["safeband_activity"].astype(str).to_numpy()
    fold=m["fold"].astype(str).to_numpy()
    tr=fold!="test"; te=fold=="test"
    print("Extracting 38 ACC features...")
    F=features(X)
    if not np.isfinite(F).all(): raise ValueError("Non-finite handcrafted features detected")
    classes=sorted(np.unique(y))
    models={
      "logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced")),
      "random_forest":RandomForestClassifier(n_estimators=300,n_jobs=-1,class_weight="balanced_subsample",random_state=42),
      "extra_trees":ExtraTreesClassifier(n_estimators=300,n_jobs=-1,class_weight="balanced",random_state=42),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=250,learning_rate=.08,max_leaf_nodes=31,l2_regularization=1.0,random_state=42),
    }
    results={"protocol":"AICOS-HAR SafeBand taxonomy ACC-only V6",
             "classes":classes,"train_windows":int(tr.sum()),"test_windows":int(te.sum()),
             "feature_count":38,"models":{}}
    for name,model in models.items():
        print(f"Training {name}...")
        results["models"][name]=eval_model(name,model,F[tr],y[tr],F[te],y[te],classes)
        r=results["models"][name]
        print(f"  acc={r['accuracy']:.4f} bal={r['balanced_accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"Report: {out/'report.json'}")

if __name__=="__main__": main()
