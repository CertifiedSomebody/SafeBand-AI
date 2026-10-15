
"""AICOS-HAR official-fold classical benchmark.

Input: manifest + NPZ created by build_aicos_har_windows_v1.py.
Uses acquisition-level official folds, so windows from one acquisition never
cross train/test. Normalization is fit on training windows only.

Models: ExtraTrees, RandomForest, HistGradientBoosting, RBF-SVM.
Features: 38-feature SafeBand ACC contract used in prior activity work.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

FEATURES=["ax_mean","ax_std","ax_min","ax_max","ax_rms",
"ay_mean","ay_std","ay_min","ay_max","ay_rms",
"az_mean","az_std","az_min","az_max","az_rms",
"acc_mag_mean","acc_mag_std","acc_mag_min","acc_mag_max","acc_mag_rms",
"acc_mag_p10","acc_mag_p25","acc_mag_p50","acc_mag_p75","acc_mag_p90",
"acc_sma","acc_range","acc_jerk_mean","acc_jerk_std","acc_jerk_max",
"acc_diff_energy","acc_zero_crossings","acc_x_y_corr","acc_x_z_corr",
"acc_y_z_corr","acc_fft_low","acc_fft_mid","acc_fft_high"]

def feats(X,fs=50):
    out=[]
    for w in X:
        ax,ay,az=w
        m=np.sqrt((w*w).sum(0))
        d=np.diff(w,axis=1)
        magd=np.diff(m)
        def q(a,p): return np.percentile(a,p)
        f=[]
        for a in (ax,ay,az):
            f += [a.mean(),a.std(),a.min(),a.max(),np.sqrt(np.mean(a*a))]
        f += [m.mean(),m.std(),m.min(),m.max(),np.sqrt(np.mean(m*m)),
              q(m,10),q(m,25),q(m,50),q(m,75),q(m,90),
              np.mean(np.abs(w)),m.max()-m.min(),
              np.mean(np.abs(magd)) if len(magd) else 0,
              np.std(magd) if len(magd) else 0,
              np.max(np.abs(magd)) if len(magd) else 0,
              np.sum(d*d),np.sum(np.diff(np.sign(m-np.mean(m)))!=0)]
        for a,b in ((ax,ay),(ax,az),(ay,az)):
            f.append(np.corrcoef(a,b)[0,1] if np.std(a)>0 and np.std(b)>0 else 0)
        spec=np.abs(np.fft.rfft(m-np.mean(m)))**2
        freq=np.fft.rfftfreq(len(m),1/fs)
        for lo,hi in ((0,2),(2,5),(5,10)):
            f.append(spec[(freq>=lo)&(freq<hi)].sum())
        out.append(f)
    return np.nan_to_num(np.asarray(out,dtype=np.float32))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",required=True); ap.add_argument("--manifest",required=True)
    ap.add_argument("--out",default="models/aicos_har/classical_v1")
    args=ap.parse_args()
    X=np.load(args.npz)["X"]; rows=list(csv.DictReader(open(args.manifest,encoding="utf-8-sig")))
    y=np.array([r["activity"] for r in rows]); fold=np.array([r["fold"] for r in rows])
    F=feats(X)
    test_mask=np.char.lower(fold.astype(str))=="test"
    train_mask=~test_mask
    classes=sorted(np.unique(y))
    models={
      "extra_trees":ExtraTreesClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1),
      "random_forest":RandomForestClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=250,random_state=42),
      "rbf_svm":Pipeline([("scale",StandardScaler()),("svc",SVC(C=4,gamma="scale",class_weight="balanced"))])
    }
    results={}
    for name,model in models.items():
        model.fit(F[train_mask],y[train_mask])
        pred=model.predict(F[test_mask])
        results[name]={
          "accuracy":accuracy_score(y[test_mask],pred),
          "balanced_accuracy":balanced_accuracy_score(y[test_mask],pred),
          "macro_f1":f1_score(y[test_mask],pred,average="macro"),
          "classes":classes,
          "confusion_matrix":confusion_matrix(y[test_mask],pred,labels=classes).tolist()
        }
        print(name,results[name]["accuracy"],results[name]["balanced_accuracy"],results[name]["macro_f1"])
        out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
        import joblib
        joblib.dump(model,out/f"{name}.joblib")
    (Path(args.out)/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")

if __name__=="__main__": main()
