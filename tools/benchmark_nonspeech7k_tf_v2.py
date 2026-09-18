from pathlib import Path
import sys, argparse, json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="datasets/processed/nonspeech7k/train_tf_features_v2.csv")
    ap.add_argument("--out",default="reports/nonspeech7k_audio_v2/benchmark_report.json")
    args=ap.parse_args()
    df=pd.read_csv(ROOT/args.input,dtype={"file_id":"string"})
    if not {"file_id","label"}.issubset(df.columns): raise RuntimeError("Feature CSV missing file_id/label.")
    df["file_id"]=df["file_id"].astype("string").str.strip()
    labels=["breath","cough","crying","laugh","screaming","sneeze","yawn"]
    X=df.drop(columns=["file_id","label","audio_path"],errors="ignore").replace([np.inf,-np.inf],np.nan)
    if X.shape[1] < 10: raise RuntimeError(f"Only {X.shape[1]} numeric features found.")
    X=X.select_dtypes(include=[np.number])
    y=df.label.to_numpy(); groups=df.file_id.to_numpy()
    models={
      "logreg":make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",random_state=42)),
      "extra_trees":make_pipeline(SimpleImputer(),ExtraTreesClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=1)),
      "random_forest":make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=1)),
      "hist_gradient_boosting":make_pipeline(SimpleImputer(),HistGradientBoostingClassifier(max_iter=250,random_state=42))
    }
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    report={"protocol":"5-fold StratifiedGroupKFold grouped by file_id","features":int(X.shape[1]),"models":{}}
    for name,model in models.items():
        pred=np.empty(len(y),dtype=object)
        for fold,(tr,te) in enumerate(cv.split(X,y,groups),1):
            model.fit(X.iloc[tr],y[tr]); pred[te]=model.predict(X.iloc[te])
        report["models"][name]={
          "accuracy":float(accuracy_score(y,pred)),
          "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
          "macro_f1":float(f1_score(y,pred,labels=labels,average="macro")),
          "confusion_matrix":confusion_matrix(y,pred,labels=labels).tolist()
        }
        print(name, report["models"][name]["accuracy"], report["models"][name]["balanced_accuracy"], report["models"][name]["macro_f1"])
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"[PASS] report -> {out}")
if __name__=="__main__": main()
