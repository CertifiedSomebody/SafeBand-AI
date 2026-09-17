
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse, json, numpy as np, pandas as pd
import wfdb
from ai.walking_features_v5_3 import features

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--root",required=True); p.add_argument("--out",required=True); p.add_argument("--summary-out",required=True)
 p.add_argument("--window-sec",type=float,default=8); p.add_argument("--shift-sec",type=float,default=2)
 a=p.parse_args(); root=Path(a.root); rows=[]; recs=[]
 records=(root/"RECORDS").read_text().splitlines()
 for name in records:
  if "_walk" not in name.lower(): continue
  r=wfdb.rdrecord(str(root/name)); ann=wfdb.rdann(str(root/name),"atr")
  fs=float(r.fs); ppg=r.p_signal[:,1]; acc=r.p_signal[:,5:8]; gyro=r.p_signal[:,2:5]
  peaks=np.asarray(ann.sample,float)
  # target HR at window center from ECG R peaks.
  w=int(round(a.window_sec*fs)); step=int(round(a.shift_sec*fs))
  valid=0
  for start in range(0,len(ppg)-w+1,step):
   end=start+w; center=(start+end)/2
   rr= np.diff(peaks[(peaks>=start)&(peaks<end)])/fs
   target=60/np.median(rr) if len(rr)>=2 else np.nan
   if not np.isfinite(target) or not 40<=target<=220: continue
   f=features(ppg[start:end],acc[start:end],gyro[start:end],fs)
   f.update({"subject":name.split("_")[0],"record":name,"start_sample":start,"target_hr":target})
   rows.append(f); valid+=1
  recs.append({"record":name,"samples":len(ppg),"fs_hz":fs,"windows_valid":valid})
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
 Path(a.summary_out).parent.mkdir(parents=True,exist_ok=True)
 Path(a.summary_out).write_text(json.dumps({"version":"PPG WALKING V5.3 FINAL","window_sec":a.window_sec,"shift_sec":a.shift_sec,"records":recs,"rows":len(rows)},indent=2))
 print(json.dumps({"rows":len(rows),"records":len(recs)},indent=2))
if __name__=="__main__": main()
