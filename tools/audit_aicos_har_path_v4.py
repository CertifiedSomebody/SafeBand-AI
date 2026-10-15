from pathlib import Path
import argparse, csv, json, sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_aicos_har_v4 import discover_sensor_files, metadata_to_raw_acquisition_id

def read_metadata(path):
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--inventory",default="")
    ap.add_argument("--out",default="reports/aicos_har_path_reconciliation_v4")
    args=ap.parse_args()
    meta_rows=read_metadata(args.metadata)
    meta_ids={r["AcquisitionID"].strip() for r in meta_rows if r.get("AcquisitionID")}
    raw_rows=discover_sensor_files(args.data_dir)
    raw_meta_ids={r["acquisition_id"] for r in raw_rows}
    raw_ids={r["raw_acquisition_id"] for r in raw_rows}
    expected_raw={metadata_to_raw_acquisition_id(x) for x in meta_ids}
    exact=meta_ids & raw_meta_ids
    missing=sorted(meta_ids-raw_meta_ids)
    raw_only=sorted(raw_meta_ids-meta_ids)
    sensor_counts=Counter(r["sensor"] for r in raw_rows)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    report={
      "metadata_unique":len(meta_ids),"raw_unique_acquisitions":len(raw_meta_ids),
      "raw_sensor_files":len(raw_rows),"exact_metadata_to_raw_matches":len(exact),
      "metadata_without_raw_after_alias":len(missing),"raw_only_after_alias":len(raw_only),
      "sensor_file_counts":dict(sorted(sensor_counts.items())),
      "alias_expected_raw_unique":len(expected_raw),
      "missing_examples":missing[:50],"raw_only_examples":raw_only[:50],
      "lift_elevator_aliases_verified": all(
          metadata_to_raw_acquisition_id(x) in raw_ids for x in meta_ids
          if "/LiftDown_" in x or "/LiftUp_" in x
      )
    }
    (out/"aicos_har_path_reconciliation_v4.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with open(out/"aicos_har_raw_inventory_v4.csv","w",encoding="utf-8",newline="") as f:
        fields=["acquisition_id","raw_acquisition_id","participant_id","activity_trial_id","activity","trial_no","device_position_id","sensor","relative_path"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(raw_rows)
    print("AICOS-HAR PATH RECONCILIATION V4")
    print("="*40)
    print(f"metadata_unique                 : {len(meta_ids)}")
    print(f"raw_unique_acquisitions         : {len(raw_meta_ids)}")
    print(f"raw_sensor_files                : {len(raw_rows)}")
    print(f"metadata_to_raw_matches         : {len(exact)}")
    print(f"metadata_without_raw_after_alias: {len(missing)}")
    print(f"raw_only_after_alias            : {len(raw_only)}")
    print(f"sensor_file_counts              : {dict(sorted(sensor_counts.items()))}")
    print(f"lift/elevator aliases verified  : {report['lift_elevator_aliases_verified']}")
    print(f"Report: {out}")
    if missing or raw_only:
        raise SystemExit("PATH RECONCILIATION FAILED: unresolved acquisition IDs remain.")

if __name__=="__main__": main()
