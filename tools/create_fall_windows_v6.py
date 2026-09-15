from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai.bits2_features import extract_motion_features, FEATURE_COLUMNS
from ai.fall_event_features import fall_event_features, FALL_EVENT_FEATURES
from ai.fall_v6_features import extract_fall_v6, V6_FEATURES

ACTIVITY_ADL={1,2,3,9,10,11,13,14,15,16}

def build(args):
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
 cols=["window_id","subject_id","source_file","recording_type","label_id","label","start_sample","end_sample","window_samples",*FEATURE_COLUMNS,*FALL_EVENT_FEATURES,*V6_FEATURES]
 total=0; cur=None; samples=[]; meta={}
 def flush():
  nonlocal total,samples,meta
  if not meta or len(samples)<args.window_samples: samples=[]; return
  label="FALL" if meta["recording_type"]=="fall" else ("NON_FALL" if int(meta["label_id"]) in ACTIVITY_ADL else "")
  if not label: samples=[]; return
  for st in range(0,len(samples)-args.window_samples+1,args.stride):
   w=samples[st:st+args.window_samples]; f=extract_motion_features(w); f.update(fall_event_features(w)); f.update(extract_fall_v6(w))
   writer.writerow({"window_id":f'{meta["subject_id"]}_{Path(meta["source_file"]).stem}_{st}',"subject_id":meta["subject_id"],"source_file":meta["source_file"],"recording_type":meta["recording_type"],"label_id":meta["label_id"],"label":label,"start_sample":st,"end_sample":st+args.window_samples-1,"window_samples":args.window_samples,**f}); total+=1
  samples=[]
 with open(args.input,newline="",encoding="utf-8") as fi,open(out,"w",newline="",encoding="utf-8") as fo:
  reader=csv.DictReader(fi); writer=csv.DictWriter(fo,fieldnames=cols); writer.writeheader()
  for r in reader:
   if r.get("sensor_type")!="acc": continue
   key=r["source_file"]
   if cur is not None and key!=cur: flush()
   if key!=cur: cur=key; meta={k:r[k] for k in ["subject_id","source_file","recording_type","label_id"]}
   try: samples.append((float(r["x"]),float(r["y"]),float(r["z"])))
   except (TypeError,ValueError): pass
  if cur is not None: flush()
 print(json.dumps({"task":"fall","window_samples":args.window_samples,"stride":args.stride,"windows_written":total,"output":str(out)},indent=2))

if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--out",required=True); ap.add_argument("--window-samples",type=int,default=60); ap.add_argument("--overlap",type=float,default=.5)
 a=ap.parse_args(); a.stride=max(1,int(round(a.window_samples*(1-a.overlap)))); build(a)
