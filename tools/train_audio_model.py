"""Train the optional SafeBand INMP441 audio-event model from WAV files.

Expected input layout:

datasets/raw/audio/
    SCREAM/
        subject01/*.wav
        subject02/*.wav
    SHOUT/
        subject01/*.wav
    NORMAL/
        subject01/*.wav

The immediate parent directory of a WAV file is used as subject when it is
not itself the class directory. If a dataset has no subject directories, the
filename must contain a subject token such as s01 or subject01.

A subject/group-disjoint 60/20/20 split is enforced. The script refuses a
split that loses a class, because that makes the held-out evaluation
misleading.
"""
from __future__ import annotations
import argparse,json,re,sys,wave
from pathlib import Path
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
from ai.audio_features import FEATURE_COLUMNS,extract_audio_features

def read_wav(path:Path):
    with wave.open(str(path),"rb") as w:
        ch,sw,fr,n=w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes()
        if sw not in (2,): raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        raw=w.readframes(n)
    x=np.frombuffer(raw,dtype="<i2").astype(np.float32)/32768.0
    if ch>1: x=x.reshape(-1,ch).mean(axis=1)
    return x,float(fr)

def subject_for(path:Path,class_dir:Path)->str:
    # Prefer explicit subject directory immediately above the file.
    parent=path.parent.name
    if parent != class_dir.name:
        return parent
    m=re.search(r"(?:subject|sub|s)[_-]?(\d+)",path.stem,re.I)
    return f"s{m.group(1)}" if m else path.stem

def window_rows(path,label,subject,window_sec,shift_sec,relative_root):
    x,fs=read_wav(path); w=max(8,int(round(window_sec*fs))); step=max(1,int(round(shift_sec*fs)))
    rows=[]
    for start in range(0,max(0,len(x)-w+1),step):
        feat=extract_audio_features(x[start:start+w],fs)
        rows.append({"label":label,"subject":subject,"file":str(path.relative_to(relative_root)),
                     "start_sample":start,"sample_rate_hz":fs,**feat})
    return rows

def split_subjects(df,seed):
    labels=set(df.label)
    for attempt in range(100):
        s=seed+attempt
        a=GroupShuffleSplit(1,test_size=.20,random_state=s)
        trv_i,te_i=next(a.split(df,groups=df.subject))
        trv,te=df.iloc[trv_i],df.iloc[te_i]
        b=GroupShuffleSplit(1,test_size=.25,random_state=s+1000)
        tr_i,va_i=next(b.split(trv,groups=trv.subject))
        tr,va=trv.iloc[tr_i],trv.iloc[va_i]
        if all(set(x.label)==labels for x in (tr,va,te)):
            return tr,va,te
    raise RuntimeError("Could not find a subject-disjoint 60/20/20 split containing every audio class. "
                       "Use more subjects per class or adjust the dataset.")

def metrics(y,p,labels):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,average="macro",zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_precision":float(pr),"macro_recall":float(rc),"macro_f1":float(f1),
            "labels":labels,"confusion_matrix":confusion_matrix(y,p,labels=labels).tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,help="Root WAV directory organized by class.")
    ap.add_argument("--model-out",default=str(ROOT/"models"/"audio_event_v1"/"audio_event_model.joblib"))
    ap.add_argument("--report-out",default=str(ROOT/"models"/"audio_event_v1"/"report.json"))
    ap.add_argument("--window-sec",type=float,default=2.0)
    ap.add_argument("--shift-sec",type=float,default=1.0)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    base=Path(args.input)
    files=sorted(base.rglob("*.wav"))
    if not files: raise FileNotFoundError(f"No WAV files found under {base}")
    rows=[]
    class_dirs=sorted({p.parent.parent if p.parent.name.lower().startswith(("subject","sub","s")) else p.parent for p in files})
    # Class is the first directory below input for the documented layout.
    for p in files:
        rel=p.relative_to(base)
        label=rel.parts[0].upper() if len(rel.parts)>=2 else p.parent.name.upper()
        class_dir=base/rel.parts[0] if len(rel.parts)>=2 else p.parent
        subject=subject_for(p,class_dir)
        rows.extend(window_rows(p,label,subject,args.window_sec,args.shift_sec,base))
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No audio windows were generated.")
    labels=sorted(df.label.unique())
    if len(labels)<2: raise RuntimeError("Need at least two audio classes.")
    tr,va,te=split_subjects(df,args.seed)
    Xtr,Xv,Xte=tr[FEATURE_COLUMNS],va[FEATURE_COLUMNS],te[FEATURE_COLUMNS]
    ytr,yv,yte=tr.label,va.label,te.label
    models={
      "logistic":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=args.seed))]),
      "rf":Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",min_samples_leaf=2,n_jobs=-1,random_state=args.seed))]),
      "extra_trees":Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesClassifier(n_estimators=400,class_weight="balanced",min_samples_leaf=2,n_jobs=-1,random_state=args.seed))]),
      "hgb":Pipeline([("imp",SimpleImputer(strategy="median")),("m",HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=args.seed))])
    }
    validation={}
    for name,m in models.items():
        m.fit(Xtr,ytr); validation[name]=metrics(yv,m.predict(Xv),labels)
    selected=max(validation,key=lambda n:(validation[n]["macro_f1"],validation[n]["balanced_accuracy"]))
    model=models[selected]; dev=pd.concat([tr,va],ignore_index=True); model.fit(dev[FEATURE_COLUMNS],dev.label)
    pred=model.predict(Xte); test=metrics(yte,pred,labels)
    out=Path(args.model_out); out.parent.mkdir(parents=True,exist_ok=True)
    pred_out=out.parent/"test_predictions.csv"
    pd.DataFrame({"subject":te.subject.to_numpy(),"file":te.file.to_numpy(),"label":yte.to_numpy(),"prediction":pred}).to_csv(pred_out,index=False)
    joblib.dump({"model":model,"feature_columns":FEATURE_COLUMNS,"model_name":"SafeBand Audio Event Model",
                 "model_version":"audio_event_v1","label_classes":labels,"window_sec":args.window_sec,
                 "source":"WAV PCM audio","notes":"Labels are dataset annotations; loudness alone is not a distress label."},out)
    report={"version":"SafeBand Audio Event V1","dataset_root":str(base.resolve()),"wav_files":len(files),
            "windows":len(df),"subjects":int(df.subject.nunique()),"classes":labels,"feature_count":len(FEATURE_COLUMNS),
            "window_sec":args.window_sec,"shift_sec":args.shift_sec,
            "split":{"method":"grouped_subject_60_20_20","train_subjects":sorted(tr.subject.unique()),
                     "validation_subjects":sorted(va.subject.unique()),"test_subjects":sorted(te.subject.unique()),"subject_overlap":False},
            "validation":validation,"selected_model":selected,"held_out_test":test,"artifact":str(out),"test_predictions":str(pred_out)}
    rp=Path(args.report_out); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
