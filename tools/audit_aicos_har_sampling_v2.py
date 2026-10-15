
"""AICOS-HAR bounded sampling audit V2.

Reads a bounded number of actual sensor files. Reports timestamp-derived
sampling statistics and value validity. It does not build windows or train.
"""
from pathlib import Path
import argparse, csv, math, statistics, sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_aicos_har_v3 import discover_sensor_files

def read_sensor(path):
    rows=[]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            p=[x.strip() for x in line.strip().split("|")]
            if len(p)<2: continue
            try:
                t=int(float(p[0]))
                vals=[float(x) for x in p[1:]]
            except ValueError:
                continue
            rows.append((t,vals))
    return rows

def stats(rows):
    if len(rows)<3: return None
    ts=[x[0] for x in rows]
    dt=[(b-a)/1e9 for a,b in zip(ts,ts[1:]) if b>a]
    if not dt: return None
    return {
        "rows":len(rows),
        "duration_s":(ts[-1]-ts[0])/1e9,
        "median_hz":1/statistics.median(dt),
        "min_hz":1/max(dt),
        "max_hz":1/min(dt),
        "nonmonotonic":sum(b<=a for a,b in zip(ts,ts[1:])),
        "nonfinite":sum(not math.isfinite(v) for _,vs in rows for v in vs),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--max-acquisitions",type=int,default=40)
    ap.add_argument("--out",default="reports/aicos_har_sampling_v2")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    rows=discover_sensor_files(args.data_dir)
    by=defaultdict(list)
    for r in rows: by[r["acquisition_id"]].append(r)
    ids=sorted(by)[:args.max_acquisitions]

    result=[]
    for aid in ids:
        for r in sorted(by[aid],key=lambda x:x["sensor"]):
            st=stats(read_sensor(r["path"]))
            if st:
                result.append({"acquisition_id":aid,"sensor":r["sensor"],
                               "relative_path":r["relative_path"],**st})
    fields=list(result[0]) if result else ["acquisition_id","sensor","relative_path"]
    with open(out/"sampling_audit.csv","w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(result)

    print("AICOS-HAR SAMPLING AUDIT V2")
    print("="*35)
    print(f"Acquisitions sampled: {len(ids)}")
    print(f"Sensor files read: {len(result)}")
    for s in sorted({r["sensor"] for r in result}):
        x=[r["median_hz"] for r in result if r["sensor"]==s]
        print(f"{s:6}: n={len(x)}, median={statistics.median(x):.3f} Hz, "
              f"min={min(x):.3f}, max={max(x):.3f}")
    print(f"Report: {out/'sampling_audit.csv'}")

if __name__=="__main__": main()
