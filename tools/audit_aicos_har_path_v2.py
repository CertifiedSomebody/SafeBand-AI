"""
AICOS-HAR raw hierarchy audit V2.

Reads filenames/path structure only. It does NOT load sensor samples.
It compares raw acquisition directories against metadata/inventory and
prints representative examples for the 774-path discrepancy.
"""
from pathlib import Path
import argparse, csv, json
from collections import Counter, defaultdict
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_aicos_har_v2 import discover_sensor_files, activity_alias

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", default="reports/aicos_har_path_audit_v2")
    ap.add_argument("--examples", type=int, default=25)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    metadata = read_csv(Path(args.metadata))
    inventory = read_csv(Path(args.inventory))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta_ids = {r["AcquisitionID"].strip() for r in metadata}
    inv_by_id = {r["AcquisitionID"].strip(): r for r in inventory}

    raw_rows = discover_sensor_files(data_dir)
    raw_by_id = defaultdict(list)
    for r in raw_rows:
        raw_by_id[r["acquisition_id"]].append(r)

    raw_ids = set(raw_by_id)
    missing_from_raw = sorted(meta_ids - raw_ids)
    raw_only = sorted(raw_ids - meta_ids)

    # Try conservative normalized key based on participant/activity/trial/position.
    def key_from_id(aid):
        p = aid.split("/")
        if len(p) < 3:
            return None
        activity_trial = p[1]
        if "_" in activity_trial and activity_trial.rsplit("_", 1)[1].isdigit():
            activity, trial = activity_trial.rsplit("_", 1)
        else:
            activity, trial = activity_trial, ""
        return (p[0].strip().lower(), activity_alias(activity), trial, p[2].strip().lower())

    raw_norm = defaultdict(list)
    meta_norm = defaultdict(list)
    for x in raw_ids:
        k = key_from_id(x)
        if k: raw_norm[k].append(x)
    for x in meta_ids:
        k = key_from_id(x)
        if k: meta_norm[k].append(x)

    normalized_pairs = []
    for k, mids in meta_norm.items():
        rids = raw_norm.get(k, [])
        if len(mids) == 1 and len(rids) == 1 and mids[0] != rids[0]:
            normalized_pairs.append((mids[0], rids[0]))

    report = {
        "metadata_unique": len(meta_ids),
        "inventory_unique": len(inv_by_id),
        "raw_acquisition_ids_from_files": len(raw_ids),
        "metadata_not_found_by_current_path_rule": len(missing_from_raw),
        "raw_only_by_current_path_rule": len(raw_only),
        "normalized_nonidentical_pairs": len(normalized_pairs),
        "raw_sensor_files": len(raw_rows),
        "raw_sensor_counts": Counter(r["sensor"] for r in raw_rows),
        "missing_activity_counts": Counter(
            x.split("/", 1)[1].rsplit("_", 1)[0] if "/" in x else "<bad>"
            for x in missing_from_raw
        ),
        "raw_only_activity_counts": Counter(
            x.split("/", 1)[1].rsplit("_", 1)[0] if "/" in x else "<bad>"
            for x in raw_only
        ),
        "sample_missing": missing_from_raw[:args.examples],
        "sample_raw_only": raw_only[:args.examples],
        "sample_normalized_pairs": normalized_pairs[:args.examples],
    }

    # Save representative path evidence.
    rows = []
    for aid in missing_from_raw[:args.examples]:
        rows.append({
            "type": "metadata_not_found",
            "acquisition_id": aid,
            "inventory_raw_found": inv_by_id.get(aid, {}).get("raw_found", ""),
            "inventory_sensors_found": inv_by_id.get(aid, {}).get("sensors_found", ""),
            "actual_raw_files_under_exact_id": len(raw_by_id.get(aid, [])),
            "actual_raw_paths": ";".join(x["relative_path"] for x in raw_by_id.get(aid, [])),
        })
    for aid in raw_only[:args.examples]:
        rows.append({
            "type": "raw_only",
            "acquisition_id": aid,
            "inventory_raw_found": "",
            "inventory_sensors_found": "",
            "actual_raw_files_under_exact_id": len(raw_by_id.get(aid, [])),
            "actual_raw_paths": ";".join(x["relative_path"] for x in raw_by_id.get(aid, [])),
        })

    with open(out / "path_discrepancy_examples.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                           ["type","acquisition_id","inventory_raw_found","inventory_sensors_found",
                            "actual_raw_files_under_exact_id","actual_raw_paths"])
        w.writeheader()
        w.writerows(rows)

    (out / "aicos_har_path_audit_v2.json").write_text(
        json.dumps(report, indent=2, default=lambda x: dict(x)),
        encoding="utf-8"
    )

    print("AICOS-HAR RAW PATH AUDIT V2")
    print("=" * 35)
    print(f"Metadata acquisitions              : {len(meta_ids)}")
    print(f"Raw acquisitions from path rule    : {len(raw_ids)}")
    print(f"Metadata not found by path rule    : {len(missing_from_raw)}")
    print(f"Raw-only by path rule              : {len(raw_only)}")
    print(f"Normalized non-identical pairs     : {len(normalized_pairs)}")
    print()
    print("Missing metadata activity counts:")
    for k,v in Counter(report["missing_activity_counts"]).most_common():
        print(f"  {k}: {v}")
    print()
    print("Raw-only activity counts:")
    for k,v in Counter(report["raw_only_activity_counts"]).most_common():
        print(f"  {k}: {v}")
    print()
    print(f"Examples written to: {out / 'path_discrepancy_examples.csv'}")
    print(f"Full report written to: {out / 'aicos_har_path_audit_v2.json'}")

if __name__ == "__main__":
    main()
