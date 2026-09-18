import argparse, json, warnings
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from tools.common_nonspeech7k import CLASSES

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="datasets/processed/nonspeech7k/train_features_v1.csv")
    ap.add_argument("--out-dir",default="reports/nonspeech7k_audio_v1")
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    required={"file_id","label"}
    if not required.issubset(df.columns): raise ValueError(f"Missing {required-set(df.columns)}")
    feat=[c for c in df.columns if c not in {"file_id","label","path"}]
    X=df[feat].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
    y=df.label.astype(str).to_numpy(); groups=df.file_id.astype(str).to_numpy()
    models={
      "logreg":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LogisticRegression(max_iter=1500,class_weight="balanced"))]),
      "extra_trees":Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1))]),
      "random_forest":Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1))]),
      "hist_gradient_boosting":Pipeline([("imp",SimpleImputer(strategy="median")),("m",HistGradientBoostingClassifier(max_iter=250,l2_regularization=1.0,random_state=42))])
    }
    splits=list(GroupKFold(n_splits=5).split(X,y,groups))
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    summary={}
    for name,model in models.items():
        folds=[]; yp=np.empty(len(y),dtype=object)
        for k,(tr,te) in enumerate(splits,1):
            model.fit(X[tr],y[tr]); pred=model.predict(X[te]); yp[te]=pred
            folds.append({"fold":k,"accuracy":accuracy_score(y[te],pred),"balanced_accuracy":balanced_accuracy_score(y[te],pred),"macro_f1":f1_score(y[te],pred,labels=CLASSES,average="macro",zero_division=0)})
        summary[name]={"pooled":{"accuracy":accuracy_score(y,yp),"balanced_accuracy":balanced_accuracy_score(y,yp),"macro_f1":f1_score(y,yp,labels=CLASSES,average="macro",zero_division=0),"confusion_matrix":confusion_matrix(y,yp,labels=CLASSES).tolist()},"folds":folds,"features":feat}
    (out/"benchmark_report.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v["pooled"] for k,v in summary.items()},indent=2))
if __name__=="__main__": main()
