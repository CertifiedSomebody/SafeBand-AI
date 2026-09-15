from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai.fall_v6_features import extract_fall_v6, V6_FEATURES


def main():
    ap=argparse.ArgumentParser(description="Create accelerometer windows using the validated V6 feature contract.")
    ap.add_argument("--input",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--window-samples",type=int,default=60); ap.add_argument("--overlap",type=float,default=.5)
    args=ap.parse_args(); stride=max(1,int(round(args.window_samples*(1-args.overlap))))
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    cols=["window_id","subject_id","source_file","recording_type","label_id","label","start_sample","end_sample","window_samples",*V6_FEATURES]
    total=0; cur=None; samples=[]; meta={}
    def flush(writer):
        nonlocal total,samples
        if not meta or len(samples)<args.window_samples: samples=[]; return
        for st in range(0,len(samples)-args.window_samples+1,stride):
            w=samples[st:st+args.window_samples]
            feat=extract_fall_v6(w)
            writer.writerow({"window_id":f'{meta["subject_id"]}_{Path(meta["source_file"]).stem}_{st}',"subject_id":meta["subject_id"],"source_file":meta["source_file"],"recording_type":meta["recording_type"],"label_id":meta.get("label_id",""),"label":meta.get("label",""),"start_sample":st,"end_sample":st+args.window_samples-1,"window_samples":args.window_samples,**feat})
            total+=1
        samples=[]
    with open(args.input,newline="",encoding="utf-8-sig") as fi,open(out,"w",newline="",encoding="utf-8") as fo:
        reader=csv.DictReader(fi); writer=csv.DictWriter(fo,fieldnames=cols); writer.writeheader()
        for r in reader:
            if r.get("sensor_type")!="acc": continue
            key=r.get("source_file","")
            if cur is not None and key!=cur: flush(writer)
            if key!=cur:
                cur=key; meta={k:r.get(k,"") for k in ["subject_id","source_file","recording_type","label_id","label"]}
            try: samples.append((float(r["x"]),float(r["y"]),float(r["z"])))
            except (TypeError,ValueError): pass
        if cur is not None: flush(writer)
    print(json.dumps({"windows_written":total,"window_samples":args.window_samples,"stride":stride,"output":str(out)},indent=2))

if __name__=="__main__": main()
