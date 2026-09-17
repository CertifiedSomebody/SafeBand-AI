"""Train/evaluate high-accuracy FORTH-TRACE wrist activity models.

Subject-independent protocol. Validation selects the model; the test subjects
are untouched until final evaluation. Multiple window sizes can be benchmarked.
Models: ExtraTrees, RandomForest, HistGradientBoosting, XGBoost (if installed),
RBF-SVM. Feature extraction uses ACC+GYRO+MAG and never uses participant ID.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
try:
 from xgboost import XGBClassifier
except Exception: XGBClassifier=None
from ai.forth_trace_features import CHANNELS, extract_window_features

FS=51.2

def load_windows(path):
    df=pd.read_csv(path)
    required=["participant_id","activity_label","window_samples"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    excluded={"participant_id","start_row","end_row","window_samples","sample_rate_hz","label_purity","activity_label"}
    feature_cols=[c for c in df.columns if c not in excluded]
    X=df[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    y=df.activity_label.astype(str).reset_index(drop=True)
    g=df.participant_id.astype(str).reset_index(drop=True)
    meta=df[["participant_id","label_purity","window_samples"]].reset_index(drop=True)
    return X,y,g,meta

def split_groups(X,y,g,seed):
    classes=set(y.unique())
    for k in range(100):
        s=seed+k
        a=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=s)
        trv,te=next(a.split(X,y,g)); trv_g=g.iloc[trv]
        b=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=s+1000)
        tr,va=next(b.split(X.iloc[trv],y.iloc[trv],trv_g)); tr=trv[tr]; va=trv[va]
        if all(set(y.iloc[idx].unique())==classes for idx in (tr,va,te)):
            return tr,va,te
    raise RuntimeError("Could not find a subject-independent split containing every class")

def metrics(y,p):
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average='macro',zero_division=0))}

def models(seed, n_classes):
    d={
      "extra_trees":ExtraTreesClassifier(n_estimators=600,max_features='sqrt',min_samples_leaf=1,class_weight='balanced',random_state=seed,n_jobs=-1),
      "random_forest":RandomForestClassifier(n_estimators=600,max_features='sqrt',min_samples_leaf=1,class_weight='balanced',random_state=seed,n_jobs=-1),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=350,learning_rate=.06,max_leaf_nodes=31,l2_regularization=.2,random_state=seed),
      "rbf_svm":Pipeline([('scale',StandardScaler()),('model',SVC(C=8.0,gamma='scale',class_weight='balanced',probability=False,random_state=seed))]),
    }
    if XGBClassifier is not None:
      d['xgboost']=XGBClassifier(n_estimators=700,max_depth=7,learning_rate=.05,subsample=.85,colsample_bytree=.85,min_child_weight=2,reg_lambda=1.0,objective='multi:softprob',num_class=n_classes,eval_metric='mlogloss',tree_method='hist',random_state=seed,n_jobs=-1)
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--out-dir',default='models/forth_trace_activity'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--include-xgboost',action='store_true')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    X,y,g,meta=load_windows(args.input); tr,va,te=split_groups(X,y,g,args.seed)
    results={}; fitted={}
    label_encoder=LabelEncoder().fit(y)
    y_enc=pd.Series(label_encoder.transform(y),index=y.index)
    for name,m in models(args.seed,len(y.unique())).items():
      if name=='xgboost' and not args.include_xgboost: continue
      if name=='xgboost':
        m.fit(X.iloc[tr],y_enc.iloc[tr]); p=label_encoder.inverse_transform(m.predict(X.iloc[va]).astype(int))
      else:
        m.fit(X.iloc[tr],y.iloc[tr]); p=m.predict(X.iloc[va])
      results[name]=metrics(y.iloc[va],p); fitted[name]=m
      print(f"{name:24s} val acc={results[name]['accuracy']:.4f} bal={results[name]['balanced_accuracy']:.4f} macroF1={results[name]['macro_f1']:.4f}")
    best_name=max(results,key=lambda n:(results[n]['macro_f1'],results[n]['balanced_accuracy'],results[n]['accuracy']))
    best=fitted[best_name]; trainval=np.concatenate([tr,va])
    if best_name=='xgboost':
      best.fit(X.iloc[trainval],y_enc.iloc[trainval]); pred=label_encoder.inverse_transform(best.predict(X.iloc[te]).astype(int))
    else:
      best.fit(X.iloc[trainval],y.iloc[trainval]); pred=best.predict(X.iloc[te])
    test=metrics(y.iloc[te],pred); labels=sorted(y.unique()); rep=classification_report(y.iloc[te],pred,labels=labels,output_dict=True,zero_division=0); cm=confusion_matrix(y.iloc[te],pred,labels=labels).tolist()
    payload={'model':best,'feature_columns':X.columns.tolist(),'model_name':f'SafeBand FORTH-TRACE Left-Wrist Activity Model','model_version':'forth_trace_9dof_v1','sample_rate_hz':FS,'window_samples':int(meta.window_samples.iloc[0]),'task':Path(args.input).stem,'label_classes':labels,'selection_metric':'macro_f1, then balanced_accuracy, then accuracy','xgb_label_encoder': label_encoder if best_name=='xgboost' else None}
    joblib.dump(payload,out/'activity_model.joblib')
    report={'dataset':str(Path(args.input)),'sensor':'dev1 left wrist','sample_rate_hz':FS,'subjects':sorted(g.unique()),'feature_count':X.shape[1],'validation_models':results,'selected_model':best_name,'test':test,'classification_report':rep,'confusion_matrix':cm,'labels':labels,'split':{'train_subjects':sorted(g.iloc[tr].unique()),'validation_subjects':sorted(g.iloc[va].unique()),'test_subjects':sorted(g.iloc[te].unique())}}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame({'subject':g.iloc[te].values,'y_true':y.iloc[te].values,'y_pred':pred}).to_csv(out/'test_predictions.csv',index=False)
    print(f"\nSELECTED: {best_name}\nTEST: {test}\nSaved to {out}")
    print(classification_report(y.iloc[te],pred,labels=labels,zero_division=0))

if __name__=='__main__': main()
