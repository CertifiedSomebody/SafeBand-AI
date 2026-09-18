
from pathlib import Path
import csv, argparse, collections, math, sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="datasets/synthetic/bno055/bno055_synthetic_v1.csv")
    args=ap.parse_args()
    p=ROOT/args.csv
    counts=collections.Counter(); subjects=set(); sessions=set(); bad=0; rows=0
    prev=None; dt=[]
    with p.open(encoding="utf-8",newline="") as f:
        r=csv.DictReader(f)
        expected=["subject_id","session_id","timestamp_s","activity_label",
                  "accel_x_mps2","accel_y_mps2","accel_z_mps2",
                  "gyro_x_dps","gyro_y_dps","gyro_z_dps",
                  "mag_x_uT","mag_y_uT","mag_z_uT"]
        missing=[x for x in expected if x not in r.fieldnames]
        if missing: raise SystemExit(f"Missing required columns: {missing}")
        for row in r:
            rows+=1; subjects.add(row["subject_id"]); sessions.add((row["subject_id"],row["session_id"]))
            counts[row["activity_label"]]+=1
            vals=[]
            for k,v in row.items():
                if k in ("subject_id","session_id","activity_label"): continue
                try: vals.append(float(v))
                except: bad+=1; continue
                if not math.isfinite(vals[-1]): bad+=1
            key=(row["subject_id"],row["session_id"])
            t=float(row["timestamp_s"])
            if prev and prev[0]==key:
                delta=t-prev[1]
                if delta > 0:
                    dt.append(delta)
            prev=(key,t)
    print(f"File: {p}")
    print(f"Rows: {rows:,}")
    print(f"Subjects: {len(subjects)} | subject-session pairs: {len(sessions)}")
    print(f"Activities: {dict(sorted(counts.items()))}")
    print(f"Non-finite/non-numeric values: {bad}")
    if dt:
        print(f"Median timestep: {sorted(dt)[len(dt)//2]:.6f} s")
        print(f"Approx rate: {1/(sum(dt)/len(dt)):.3f} Hz")
    print("Schema audit: PASS" if bad==0 else "Schema audit: CHECK")
if __name__=="__main__": main()
