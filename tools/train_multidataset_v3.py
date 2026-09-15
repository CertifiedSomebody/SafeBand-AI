
#!/usr/bin/env python3
"""SafeBand Multi-Dataset V3: physically motivated raw-signal augmentation benchmark.

Augmentation is applied ONLY to training windows, BEFORE 84-feature extraction.
Validation and test are read from the already prepared feature CSVs and remain
untouched. The V2 subject split is reproduced exactly for comparability.

Recipes:
  baseline              V2 training data only
  orientation           + one random small 3-D rotation per training window
  scale_noise            + magnitude scaling and low-level sensor noise
  orientation_scale_noise
                         + rotation, magnitude scaling, low-level noise
  temporal_jitter        + one nearby window start per recording
  combined              + three independent augmented copies: orientation,
                           scale_noise, temporal_jitter

The script never augments validation/test and never uses test data for selection.
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, time, zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.v3_features import FEATURES

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--bits2",required=True,help="Prepared BITS-2 V6 feature CSV")
    p.add_argument("--sisfall",required=True,help="Prepared SisFall V6 feature CSV")
    p.add_argument("--bits2-raw",required=True,help="Raw BITS-2 ZIP")
    p.add_argument("--sisfall-raw",required=True,help="Raw SisFall ZIP")
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--models",default="logistic,rf,et,hgb")
    p.add_argument("--recipes",default="baseline,orientation,scale_noise,orientation_scale_noise,temporal_jitter,combined")
    p.add_argument("--max-aug-windows",type=int,default=0,help="0=all; debugging only")
    return p.parse_args()

def col(df,cands):
    low={str(c).lower():c for c in df.columns}
    for c in cands:
        if c.lower() in low:return low[c.lower()]
    return None

def subject_key(v):
    s=str(v); m=re.search(r"(S[AE]\d+|user\d+)",s,re.I)
    return m.group(1).upper() if m else s

def load_df(path,dataset):
    df=pd.read_csv(path)
    sc=col(df,["subject_id","subject","user_id","user"])
    lc=col(df,["is_fall","target","y","label","recording_type"])
    rc=col(df,["recording_id","source_file","recording","file"])
    if not sc or not lc or not rc: raise ValueError(f"{dataset}: missing subject/label/recording column")
    missing=[c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError(f"{dataset}: missing {len(missing)} V6 features: {missing}")
    y=df[lc]
    if pd.api.types.is_numeric_dtype(y): yy=pd.to_numeric(y,errors="coerce")
    else: yy=y.astype(str).str.strip().str.upper().map({"FALL":1,"NON_FALL":0,"NONFALL":0,"NON-FALL":0,"TRUE":1,"FALSE":0,"1":1,"0":0})
    if yy.isna().any(): raise ValueError(f"{dataset}: invalid labels in {lc}")
    out=df.copy(); out["__subject"]=out[sc].map(subject_key); out["__recording"]=out[rc].astype(str)
    out["__y"]=yy.astype(int); out["__dataset"]=dataset
    return out

def split_subjects(df,seed,test_size=.20,val_size=.20):
    rng=np.random.RandomState(seed)
    stats=df.groupby("__subject")["__y"].agg(["sum","count"])
    fall=np.array(stats.index[stats["sum"]>0]); non=np.array(stats.index[stats["sum"]==0])
    rng.shuffle(fall); rng.shuffle(non)
    def one(a):
        n=len(a); nt=max(1,round(n*test_size)) if n>=3 else 0
        nv=max(1,round(n*val_size)) if n-nt>=3 else 0
        return a[nt+nv:],a[nt:nt+nv],a[:nt]
    ft,fv,fte=one(fall); nt,nv,nte=one(non)
    tr,va,te=set(ft)|set(nt),set(fv)|set(nv),set(fte)|set(nte)
    if tr&va or tr&te or va&te: raise AssertionError("subject leakage")
    return tr,va,te

def pipe(kind,seed):
    if kind=="logistic": return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed))])
    if kind=="rf": return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=seed))])
    if kind=="et": return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=seed))])
    if kind=="hgb": return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=seed))])
    raise ValueError(kind)

def score(y,p,t=.5):
    pred=(p>=t).astype(int); cm=confusion_matrix(y,pred,labels=[0,1])
    return {"threshold":t,"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
      "f1":float(f1_score(y,pred,zero_division=0)),"confusion_matrix":cm.tolist(),"n":int(len(y)),
      "positives":int(y.sum()),"predicted_positives":int(pred.sum()),
      "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
      "false_positive_rate":float(cm[0,1]/max(1,cm[0,0]+cm[0,1]))}

def rotation_matrix(rng,max_deg=12.0):
    q=rng.normal(size=4); q=q/np.linalg.norm(q)
    # Restrict rotation angle while keeping a uniformly random axis.
    axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
    angle=rng.uniform(-math.radians(max_deg),math.radians(max_deg))
    x,y,z=axis; c=math.cos(angle); s=math.sin(angle); C=1-c
    return np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],
                     [y*x*C+z*s,c+y*y*C,y*z*C-x*s],
                     [z*x*C-y*s,z*y*C+x*s,c+z*z*C]])

def augment_signal(w,recipe,rng):
    x=np.asarray(w,dtype=float).copy()
    if recipe in ("orientation","orientation_scale_noise"):
        x=x@rotation_matrix(rng).T
    if recipe in ("scale_noise","orientation_scale_noise"):
        x*=rng.uniform(.90,1.10)
        sigma=max(float(np.std(x)),1e-6)*rng.uniform(.005,.015)
        x+=rng.normal(0,sigma,size=x.shape)
    return x

def temporal_shift(w,raw,start,rng):
    lo=max(0,start-rng.randint(1,4)); hi=min(len(raw)-60,start+rng.randint(1,4)+1)
    if hi<lo:return np.asarray(w,dtype=float)
    st=int(rng.randint(lo,hi+1)); return np.asarray(raw[st:st+60],dtype=float)

def parse_bits2_acc(z,name):
    rows=[]
    with z.open(name) as fh:
        for line in fh:
            try:
                parts=line.decode("utf-8",errors="ignore").strip().split(",")
                if len(parts)>=6 and parts[-1].strip().lower()=="acc":
                    rows.append((float(parts[1]),float(parts[2]),float(parts[3])))
            except (ValueError,IndexError): pass
    return np.asarray(rows,dtype=float)

PAT=re.compile(r"^(D|F)(\d{2})_(SA|SE)(\d{2})_R(\d{2})\.txt$",re.I)
def parse_sisfall(z,name):
    with z.open(name) as fh: raw=fh.read()
    text=raw.replace(b";\r\n",b",\n").replace(b";\n",b",\n").rstrip(b",\r\n")
    arr=np.fromstring(text,sep=",",dtype=np.int32)
    rem=arr.size%9
    if rem==1: arr=arr[:-1]
    elif rem: raise ValueError(f"{name}: parsed remainder {rem}")
    if arr.size<9:return np.empty((0,3))
    arr=arr.reshape(-1,9)
    xyz=arr[:,:3].astype(float)*((2*16)/(2**13)) # ADXL345 -> g
    n=(len(xyz)//10)*10
    return xyz[:n].reshape(-1,10,3).mean(axis=1)

def raw_recordings(bits_zip,sis_zip,subjects):
    """Yield (dataset, subject, source, label, samples20Hz)."""
    with zipfile.ZipFile(bits_zip) as zb:
        for name in zb.namelist():
            base=os.path.basename(name)
            m=re.match(r"^user(\d+)_((?:adl)|(?:fall))(\d+)\.csv$",base,re.I)
            if not m: continue
            sid=m.group(1); sub="USER"+sid
            if sub not in subjects: continue
            label=1 if m.group(2).lower()=="fall" else 0
            x=parse_bits2_acc(zb,name)
            if len(x)>=60: yield "BITS2",sub,name,label,x
    with zipfile.ZipFile(sis_zip) as zs:
        for name in zs.namelist():
            base=os.path.basename(name); m=PAT.match(base)
            if not m: continue
            typ,num,prefix,sid,trial=m.groups(); sub=prefix.upper()+sid
            if sub not in subjects: continue
            x=parse_sisfall(zs,name)
            if len(x)>=60: yield "SisFall",sub,name,1 if typ.upper()=="F" else 0,x

def make_augmented(bits_raw,sis_raw,train_subjects,recipes,seed,max_aug):
    rng=np.random.RandomState(seed); rows={r:[] for r in recipes if r!="baseline"}
    total=0
    # Each original training window is represented once in the processed CSV.
    # We create only synthetic copies here; originals stay in the V2 CSV.
    for ds,sub,source,label,raw in raw_recordings(bits_raw,sis_raw,train_subjects):
        starts=range(0,len(raw)-59,30)
        for st in starts:
            w=raw[st:st+60]
            variants=[]
            for recipe in rows:
                if recipe=="orientation": variants=[("orientation",augment_signal(w,"orientation",rng))]
                elif recipe=="scale_noise": variants=[("scale_noise",augment_signal(w,"scale_noise",rng))]
                elif recipe=="orientation_scale_noise": variants=[("orientation_scale_noise",augment_signal(w,"orientation_scale_noise",rng))]
                elif recipe=="temporal_jitter": variants=[("temporal_jitter",temporal_shift(w,raw,st,rng))]
                elif recipe=="combined":
                    variants=[("orientation",augment_signal(w,"orientation",rng)),
                              ("scale_noise",augment_signal(w,"scale_noise",rng)),
                              ("temporal_jitter",temporal_shift(w,raw,st,rng))]
                for tag,signal in variants:
                    feat=__import__("ai.v3_features",fromlist=["extract_features"]).extract_features(signal)
                    feat.update({"__dataset":ds,"__subject":sub,"__recording":source,"__y":label,
                                 "__augmentation":tag,"__source_start":st})
                    rows[recipe].append(feat); total+=1
                    if max_aug and total>=max_aug:return rows
    return rows

def main():
    a=args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/"cache").mkdir(exist_ok=True)
    models=[x.strip() for x in a.models.split(",") if x.strip()]
    recipes=[x.strip() for x in a.recipes.split(",") if x.strip()]
    if "baseline" not in recipes: recipes.insert(0,"baseline")
    b=load_df(a.bits2,"BITS2"); s=load_df(a.sisfall,"SisFall")
    bs=split_subjects(b,a.seed); ss=split_subjects(s,a.seed)
    bt=b[b.__subject.isin(bs[0])].copy(); bv=b[b.__subject.isin(bs[1])].copy(); bte=b[b.__subject.isin(bs[2])].copy()
    st=s[s.__subject.isin(ss[0])].copy(); sv=s[s.__subject.isin(ss[1])].copy(); ste=s[s.__subject.isin(ss[2])].copy()
    print(f"[Preflight] BITS2 rows={len(b)} subjects={b['__subject'].nunique()} | SisFall rows={len(s)} subjects={s['__subject'].nunique()}")
    print(f"[Preflight] train/val/test BITS2={len(bt)}/{len(bv)}/{len(bte)} SisFall={len(st)}/{len(sv)}/{len(ste)}")
    print(f"[Preflight] features={len(FEATURES)} window=60@20Hz step=30")
    all_train_subjects=set(bt.__subject)|set(st.__subject)
    print(f"[Preflight] raw augmentation subjects={len(all_train_subjects)}")

    # Parse raw training windows once, then cache each recipe's synthetic features.
    aug={}
    nonbase=[r for r in recipes if r!="baseline"]
    if nonbase:
        t0=time.perf_counter(); aug=make_augmented(a.bits2_raw,a.sisfall_raw,all_train_subjects,nonbase,a.seed,a.max_aug_windows)
        print(f"[Augment] generated in {time.perf_counter()-t0:.1f}s")
        for r,items in aug.items():
            pd.DataFrame(items).to_csv(out/"cache"/f"{r}_train_augmented.csv",index=False)
            print(f"[Augment] {r}: {len(items)} synthetic windows")

    results={"version":"multidataset_v3_augmentation","seed":a.seed,
      "schema":{"feature_count":84,"window_contract":"60 samples @ 20Hz, step 30","split_unit":"subject","features":FEATURES},
      "recipes":recipes,"models":models,"splits":{},
      "selection":{"metric":"mean(BITS2 validation F1, SisFall validation F1)","test_used_for_selection":False},
      "validation":{},"test":{}}
    for ds,df,sp in [("BITS2",b,bs),("SisFall",s,ss)]:
        results["splits"][ds]={}
        for name,ids in zip(("train","val","test"),sp):
            z=df[df.__subject.isin(ids)]
            results["splits"][ds][name]={"subjects":sorted(map(str,ids)),"rows":len(z),"falls":int(z.__y.sum()),"nonfalls":int((z.__y==0).sum())}

    best=None
    train_frames={"BITS2":bt,"SisFall":st}
    val_frames={"BITS2":bv,"SisFall":sv}
    for recipe in recipes:
        if recipe=="baseline":
            tr=pd.concat([bt,st],ignore_index=True)
        else:
            # Original train windows + synthetic copies; equal dataset weight is
            # enforced through sample_weight exactly as V2.
            extra=pd.DataFrame(aug[recipe])
            tr=pd.concat([bt,st,extra],ignore_index=True)
        results["validation"][recipe]={}
        nb=int((tr["__dataset"]=="BITS2").sum()); ns=int((tr["__dataset"]=="SisFall").sum())
        weights=np.where(tr["__dataset"].to_numpy()=="BITS2",.5/nb,.5/ns)
        weights*=len(tr)
        print(f"\n[Recipe] {recipe}: rows={len(tr)}")
        for kind in models:
            m=pipe(kind,a.seed); t0=time.perf_counter()
            m.fit(tr[FEATURES],tr["__y"].to_numpy(),model__sample_weight=weights)
            pb=m.predict_proba(bv[FEATURES])[:,list(m.classes_).index(1)]
            ps=m.predict_proba(sv[FEATURES])[:,list(m.classes_).index(1)]
            mb=score(bv["__y"].to_numpy(),pb); ms=score(sv["__y"].to_numpy(),ps)
            macro={k:(mb[k]+ms[k])/2 for k in ["accuracy","balanced_accuracy","precision","recall","f1","roc_auc"]}
            results["validation"][recipe][kind]={"BITS2":mb,"SisFall":ms,"macro":macro,"fit_seconds":time.perf_counter()-t0,"train_rows":len(tr)}
            key=(macro["f1"],macro["balanced_accuracy"])
            if best is None or key>best["key"]: best={"key":key,"recipe":recipe,"model":kind}
            print(f"  {kind}: BITS2 F1={mb['f1']:.4f} | SisFall F1={ms['f1']:.4f} | macro F1={macro['f1']:.4f}")

    # Lock recipe/model, refit on train+validation, and evaluate test exactly once.
    fb=pd.concat([bt,bv],ignore_index=True); fs=pd.concat([st,sv],ignore_index=True)
    if best["recipe"]=="baseline": final=pd.concat([fb,fs],ignore_index=True)
    else: final=pd.concat([fb,fs,pd.DataFrame(aug[best["recipe"]])],ignore_index=True)
    nb=int((final["__dataset"]=="BITS2").sum()); ns=int((final["__dataset"]=="SisFall").sum())
    weights=np.where(final["__dataset"].to_numpy()=="BITS2",.5/nb,.5/ns)*len(final)
    fm=pipe(best["model"],a.seed); t0=time.perf_counter(); fm.fit(final[FEATURES],final["__y"].to_numpy(),model__sample_weight=weights)
    test=pd.concat([bte,ste],ignore_index=True)
    p=fm.predict_proba(test[FEATURES])[:,list(fm.classes_).index(1)]
    pb=p[test["__dataset"].to_numpy()=="BITS2"]; ps=p[test["__dataset"].to_numpy()=="SisFall"]
    tb=score(bte["__y"].to_numpy(),pb); ts=score(ste["__y"].to_numpy(),ps); pooled=score(test["__y"].to_numpy(),p)
    results["selected"]={"recipe":best["recipe"],"model":best["model"],"selection_metric":results["selection"]["metric"]}
    results["test"]={"BITS2":tb,"SisFall":ts,
      "macro_dataset_average":{k:(tb[k]+ts[k])/2 for k in ["accuracy","balanced_accuracy","precision","recall","f1","roc_auc"]},
      "pooled":pooled}
    results["final_fit_seconds"]=time.perf_counter()-t0
    results["augmentation_policy"]={
      "orientation_max_degrees":12.0,"magnitude_scale":[.90,1.10],
      "noise_std_fraction":[.005,.015],"temporal_jitter_max_samples":3,
      "validation_test_augmented":False,"augmentation_before_feature_extraction":True}
    artifact={"model":fm,"feature_columns":FEATURES,"selected_recipe":best["recipe"],"selected_model":best["model"],
              "seed":a.seed,"window_contract":"60 samples @ 20Hz, step 30","training_datasets":["BITS2","SisFall"],
              "augmentation_policy":results["augmentation_policy"]}
    joblib.dump(artifact,out/"multidataset_v3_augmented_fall_model.joblib")
    (out/"multidataset_v3_augmented_fall_report.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("\n=== FINAL V3 ==="); print("Selected recipe:",best["recipe"]); print("Selected model:",best["model"])
    print("BITS-2 TEST:",json.dumps(tb,indent=2)); print("SisFall TEST:",json.dumps(ts,indent=2))
    print("DATASET-MACRO:",json.dumps(results["test"]["macro_dataset_average"],indent=2))
    print("POOLED:",json.dumps(pooled,indent=2))
    print("Saved:",out/"multidataset_v3_augmented_fall_model.joblib")
    print("Saved:",out/"multidataset_v3_augmented_fall_report.json")

if __name__=="__main__": main()
