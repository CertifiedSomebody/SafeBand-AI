"""AICOS-HAR classical ACC benchmark using the official Fold=test holdout.

This is a benchmark, not test-set-driven tuning: every model is trained on
non-test folds and evaluated once on the official test fold. No model is
selected from the test score.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
import joblib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def feats(X,fs=50):
    out=[]
    for w in X:
        ax,ay,az=w;m=np.sqrt((w*w).sum(0));d=np.diff(w,axis=1);magd=np.diff(m)
        f=[]
        for a in (ax,ay,az):f += [a.mean(),a.std(),a.min(),a.max(),np.sqrt(np.mean(a*a))]
        f += [m.mean(),m.std(),m.min(),m.max(),np.sqrt(np.mean(m*m)),*np.percentile(m,[10,25,50,75,90]),np.mean(np.abs(w)),m.max()-m.min(),np.mean(np.abs(magd)),np.std(magd),np.max(np.abs(magd)),np.sum(d*d),np.sum(np.diff(np.sign(m-np.mean(m)))!=0)]
        for a,b in ((ax,ay),(ax,az),(ay,az)):f.append(np.corrcoef(a,b)[0,1] if np.std(a)>0 and np.std(b)>0 else 0)
        spec=np.abs(np.fft.rfft(m-np.mean(m)))**2;freq=np.fft.rfftfreq(len(m),1/fs)
        for lo,hi in ((0,2),(2,5),(5,10)):f.append(spec[(freq>=lo)&(freq<hi)].sum())
        out.append(f)
    return np.nan_to_num(np.asarray(out,dtype=np.float32))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--npz",required=True);ap.add_argument("--manifest",required=True);ap.add_argument("--out",default="models/aicos_har/classical_v4");args=ap.parse_args()
    X=np.load(args.npz)["X"];rows=list(csv.DictReader(open(args.manifest,encoding="utf-8-sig")));y=np.array([r["activity"] for r in rows]);fold=np.char.lower(np.array([r["fold"] for r in rows],dtype=str));F=feats(X)
    te=fold=="test";tr=~te
    if not te.any():raise SystemExit("No official Fold=test samples found.")
    classes=sorted(np.unique(y));models={
      "extra_trees":ExtraTreesClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1),
      "random_forest":RandomForestClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=250,random_state=42),
      "rbf_svm":Pipeline([("scale",StandardScaler()),("svc",SVC(C=4,gamma="scale",class_weight="balanced"))])}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);results={}
    for name,model in models.items():
        model.fit(F[tr],y[tr]);pred=model.predict(F[te])
        results[name]={"accuracy":accuracy_score(y[te],pred),"balanced_accuracy":balanced_accuracy_score(y[te],pred),"macro_f1":f1_score(y[te],pred,average="macro"),"classes":classes,"confusion_matrix":confusion_matrix(y[te],pred,labels=classes).tolist()}
        joblib.dump(model,out/f"{name}.joblib");print(name,results[name])
    (out/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"Official test samples: {te.sum()}")
    print(f"Training samples: {tr.sum()}")
if __name__=="__main__":main()
