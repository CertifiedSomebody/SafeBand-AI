#!/usr/bin/env python3
"""Audit BITS-2 HR metadata as MAX30102-domain evidence.

BITS-2 contains device-reported HR at 1 Hz, not raw MAX30102 RED/IR samples.
This tool therefore audits the HR stream and activity coverage but never
pretends it can validate a raw-PPG model.
"""
from __future__ import annotations
import argparse,csv,json,re,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--canonical",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    by=defaultdict(list); subjects=set(); recordings=set()
    with open(a.canonical,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("sensor_type")!="hrt":continue
            try:hr=float(r["x"])
            except (TypeError,ValueError):continue
            if not 30<=hr<=220:continue
            key=(r.get("subject_id",""),r.get("recording_type",""),r.get("label",""));by[key].append(hr);subjects.add(r.get("subject_id"));recordings.add(r.get("source_file"))
    rows=[]
    for (subject,rt,label),vals in sorted(by.items(),key=lambda x:(x[0][0],x[0][1],x[0][2])):
        import numpy as np
        x=np.asarray(vals,float);rows.append({"subject":subject,"recording_type":rt,"label":label,"n_hr_samples":len(x),"mean_hr_bpm":float(x.mean()),"median_hr_bpm":float(np.median(x)),"std_hr_bpm":float(x.std()),"min_hr_bpm":float(x.min()),"max_hr_bpm":float(x.max())})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({"version":"BITS2 MAX30102 HR AUDIT","subjects":len(subjects),"recordings_with_hr":len(recordings),"groups":len(rows),"rows":rows,"limitations":["BITS-2 exposes device-reported HR at 1 Hz, not raw MAX30102 RED/IR waveforms.","Therefore this audit cannot train or validate the SafeBand raw-PPG estimator.","Use it as MAX30102-domain acquisition/context evidence only."]},indent=2),encoding="utf-8")
    print(json.dumps({"subjects":len(subjects),"recordings_with_hr":len(recordings),"groups":len(rows),"out":str(out)},indent=2))
if __name__=="__main__":main()
