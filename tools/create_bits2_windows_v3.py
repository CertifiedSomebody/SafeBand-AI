"""
SafeBand AI — BITS-2 Window Builder v3.

Generates separate datasets:
  --task activity : 4-class activity windows
  --task fall     : FALL plus carefully defined NON_FALL controls

The fall dataset uses the same accelerometer source but adds event-oriented
features. Windows never cross source recordings.
"""

from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from ai.bits2_features import extract_motion_features,FEATURE_COLUMNS
from ai.stationary_features import stationary_features,STATIONARY_FEATURES
from ai.fall_event_features import fall_event_features,FALL_EVENT_FEATURES

ACTIVITY_MAP={
  1:"WALKING",2:"WALKING",3:"RUNNING",
  9:"SITTING",10:"SITTING",11:"SITTING",
  13:"RESTING",14:"RESTING",15:"RESTING",16:"RESTING",
}
ACTIVITY_ADL={1,2,3,9,10,11,13,14,15,16}

def make_label(rt,lid,task):
    if task=="fall":
        return "FALL" if rt=="fall" else ("NON_FALL" if lid in ACTIVITY_ADL else "")
    return ACTIVITY_MAP.get(lid,"") if rt=="adl" else ""

def build(args):
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    cols=["window_id","subject_id","source_file","recording_type","label_id",
          "label","start_sample","end_sample","window_samples",
          *FEATURE_COLUMNS]
    if args.task=="activity": cols += STATIONARY_FEATURES
    else: cols += FALL_EVENT_FEATURES

    total=0; cur=None; samples=[]; meta={}
    def flush():
        nonlocal total,samples,meta
        label=make_label(meta["recording_type"],int(meta["label_id"]),args.task)
        if not label or len(samples)<args.window_samples: samples=[]; return
        for st in range(0,len(samples)-args.window_samples+1,args.stride):
            w=samples[st:st+args.window_samples]
            f=extract_motion_features(w)
            f.update(stationary_features(w) if args.task=="activity" else fall_event_features(w))
            row={"window_id":f'{meta["subject_id"]}_{Path(meta["source_file"]).stem}_{st}',
                 "subject_id":meta["subject_id"],"source_file":meta["source_file"],
                 "recording_type":meta["recording_type"],"label_id":meta["label_id"],
                 "label":label,"start_sample":st,"end_sample":st+args.window_samples-1,
                 "window_samples":args.window_samples,**f}
            writer.writerow(row); total+=1
        samples=[]

    with open(args.input,newline="",encoding="utf-8") as fi,open(out,"w",newline="",encoding="utf-8") as fo:
        reader=csv.DictReader(fi); writer=csv.DictWriter(fo,fieldnames=cols); writer.writeheader()
        for r in reader:
            if r.get("sensor_type")!="acc": continue
            key=r["source_file"]
            if cur is not None and key!=cur: flush()
            if key!=cur:
                cur=key; meta={k:r[k] for k in ["subject_id","source_file","recording_type","label_id"]}
            try: samples.append((float(r["x"]),float(r["y"]),float(r["z"])))
            except (TypeError,ValueError): pass
        if cur is not None: flush()
    print(json.dumps({"task":args.task,"window_samples":args.window_samples,
                      "stride":args.stride,"windows_written":total,"output":str(out)},indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--task",choices=["activity","fall"],required=True)
    ap.add_argument("--window-samples",type=int,required=True)
    ap.add_argument("--overlap",type=float,default=.5)
    args=ap.parse_args(); args.stride=max(1,int(round(args.window_samples*(1-args.overlap))))
    build(args)
