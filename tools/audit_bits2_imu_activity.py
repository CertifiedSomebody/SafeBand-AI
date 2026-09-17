from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(description="Audit BITS2 canonical IMU data before ACC+GYRO window construction.")
    ap.add_argument("--canonical",type=Path,default=ROOT/"datasets/processed/bits2/bits2_canonical_long.csv")
    ap.add_argument("--activity",type=Path,default=ROOT/"datasets/processed/bits2/activity_windows.csv")
    a=ap.parse_args()

    required_c=["subject_id","source_file","sensor_type","timestamp_raw","x","y","z"]
    required_a=["subject_id","source_file","activity_label","window_start_sample","window_end_sample","window_samples"]
    c=pd.read_csv(a.canonical,usecols=required_c)
    w=pd.read_csv(a.activity,usecols=required_a)

    print("=== BITS2 IMU AUDIT ===")
    print("Canonical rows:",len(c))
    print("Activity windows:",len(w))
    print("Canonical sensors:",sorted(c["sensor_type"].astype(str).str.lower().unique().tolist()))
    print("Activity labels:",sorted(w["activity_label"].astype(str).str.upper().unique().tolist()))
    print("Canonical subjects:",c["subject_id"].nunique())
    print("Activity subjects:",w["subject_id"].nunique())

    c["sensor_type"]=c["sensor_type"].astype(str).str.lower()
    files=set(c["source_file"].astype(str))
    wf=set(w["source_file"].astype(str))
    missing=sorted(wf-files)
    print("Activity source files missing from canonical:",len(missing))
    if missing: print("First missing:",missing[:10])

    for sensor in ["acc","gyro","mgm","acg"]:
        q=c[c.sensor_type==sensor]
        print(f"{sensor:>4}: rows={len(q):8d}, files={q.source_file.nunique():4d}, subjects={q.subject_id.nunique():3d}")

    # Coverage at recording level for the exact files used by activity windows.
    used=c[c.source_file.isin(wf)]
    for sensor in ["acc","gyro"]:
        q=used[used.sensor_type==sensor]
        print(f"used activity files with {sensor}: {q.source_file.nunique()} / {len(wf)}")

    if missing:
        raise SystemExit("AUDIT FAILED: some activity source_file values are absent from canonical_long.csv")
    print("AUDIT PASSED: required metadata and ACC/GYRO sensor blocks are discoverable.")

if __name__=="__main__": main()
