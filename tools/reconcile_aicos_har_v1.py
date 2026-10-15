"""
AICOS-HAR metadata/raw acquisition reconciliation V1.

Purpose:
    Explain why metadata contains 5231 acquisitions while the exact raw-path
    inventory currently matches fewer acquisitions.

This is NOT an ML script.
It does not alter raw data and does not read every sensor sample.

It performs:
1. exact AcquisitionID matching
2. conservative normalization of path components
3. Lift/Elevator activity-name reconciliation
4. case/whitespace normalization
5. detection of genuinely unresolved metadata entries
6. symmetric raw-only entries
7. summary by participant/activity

Usage from repository root:
    python tools\reconcile_aicos_har_v1.py ^
      --data-dir datasets\raw\AICOS-HAR\AICOS-HAR ^
      --metadata datasets\raw\AICOS-HAR\AICOS-HAR\metadata.csv ^
      --inventory datasets\processed\aicos_har\aicos_har_acquisition_inventory.csv ^
      --out reports\aicos_har_reconciliation_v1
"""

from pathlib import Path
import argparse
import csv
import json
import re
from collections import Counter, defaultdict


def norm_text(value):
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def normalize_activity(activity):
    """Conservative activity aliasing only."""
    a = norm_text(activity)

    aliases = {
        "elevatordown": "liftdown",
        "elevator_down": "liftdown",
        "elevator-up": "liftup",
        "elevator_up": "liftup",
        "elevatorup": "liftup",
        "lift_down": "liftdown",
        "lift_up": "liftup",
        "downstairs": "downstairs",
        "upstairs": "upstairs",
        "lyingdown": "lyingdown",
        "lying_down": "lyingdown",
    }
    return aliases.get(a, a)


def parse_acquisition(aid):
    parts = [p.strip() for p in str(aid).split("/") if p.strip()]
    if len(parts) < 3:
        return None

    participant = parts[0]
    activity_trial = parts[1]
    device_position = "/".join(parts[2:])

    # Trial suffix is the final _<integer>.
    m = re.match(r"^(.*)_(\d+)$", activity_trial)
    if m:
        activity = m.group(1)
        trial_no = m.group(2)
    else:
        activity = activity_trial
        trial_no = ""

    return {
        "participant": participant,
        "activity_trial": activity_trial,
        "activity": activity,
        "trial_no": trial_no,
        "device_position": device_position,
    }


def canonical_key(aid, alias_activity=True):
    p = parse_acquisition(aid)
    if p is None:
        return None

    activity = normalize_activity(p["activity"]) if alias_activity else norm_text(p["activity"])
    return (
        norm_text(p["participant"]),
        activity,
        norm_text(p["trial_no"]),
        norm_text(p["device_position"]),
    )


def raw_acquisitions(data_dir):
    found = set()

    for p in data_dir.rglob("*.txt"):
        if p.name.lower() in {"readme.txt", "metadata.txt"}:
            continue

        rel = p.relative_to(data_dir)
        if len(rel.parts) >= 4:
            found.add("/".join(rel.parts[:3]))

    return found


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", default="reports/aicos_har_reconciliation_v1")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    inventory_path = Path(args.inventory).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    metadata = read_csv(metadata_path)
    inventory = read_csv(inventory_path)

    if not metadata:
        raise SystemExit("ERROR: metadata.csv has no rows.")

    if not inventory:
        raise SystemExit("ERROR: acquisition inventory has no rows.")

    acq_col = next(
        (c for c in metadata[0] if c.strip().lower() == "acquisitionid"),
        None,
    )
    if acq_col is None:
        raise SystemExit(
            f"ERROR: metadata has no AcquisitionID column. "
            f"Columns: {list(metadata[0])}"
        )

    inv_acq_col = next(
        (
            c
            for c in inventory[0]
            if c.strip().lower().replace("_", "") == "acquisitionid"
        ),
        None,
    )
    if inv_acq_col is None:
        raise SystemExit(
            f"ERROR: inventory has no acquisition_id column. "
            f"Columns: {list(inventory[0])}"
        )

    meta_ids = [
        str(r.get(acq_col, "")).strip()
        for r in metadata
        if str(r.get(acq_col, "")).strip()
    ]
    inv_ids = [
        str(r.get(inv_acq_col, "")).strip()
        for r in inventory
        if str(r.get(inv_acq_col, "")).strip()
    ]

    meta_set = set(meta_ids)
    inv_set = set(inv_ids)

    exact_meta = meta_set & inv_set
    exact_missing = meta_set - inv_set
    exact_raw_only = inv_set - meta_set

    # Build normalized maps.
    meta_by_key = defaultdict(list)
    raw_by_key = defaultdict(list)

    for aid in meta_ids:
        k = canonical_key(aid, alias_activity=True)
        if k is not None:
            meta_by_key[k].append(aid)

    for aid in inv_ids:
        k = canonical_key(aid, alias_activity=True)
        if k is not None:
            raw_by_key[k].append(aid)

    normalized_resolved = []
    unresolved_metadata = []
    ambiguous_metadata = []

    for aid in sorted(exact_missing):
        k = canonical_key(aid, alias_activity=True)
        candidates = raw_by_key.get(k, [])

        if len(candidates) == 1:
            normalized_resolved.append({
                "metadata_acquisition_id": aid,
                "raw_acquisition_id": candidates[0],
                "resolution": "conservative_normalized_match",
            })
        elif len(candidates) > 1:
            ambiguous_metadata.append({
                "metadata_acquisition_id": aid,
                "raw_candidates": ";".join(sorted(candidates)),
                "candidate_count": len(candidates),
            })
        else:
            unresolved_metadata.append(aid)

    resolved_raw_ids = {
        x["raw_acquisition_id"] for x in normalized_resolved
    }

    # Activity-only diagnostic: identify which activity names account for
    # unresolved metadata/raw differences.
    unresolved_meta_activity = Counter()
    for aid in unresolved_metadata:
        p = parse_acquisition(aid)
        unresolved_meta_activity[p["activity"] if p else "<unparsed>"] += 1

    raw_only_after_resolution = sorted(
        exact_raw_only - resolved_raw_ids
    )

    raw_only_activity = Counter()
    for aid in raw_only_after_resolution:
        p = parse_acquisition(aid)
        raw_only_activity[p["activity"] if p else "<unparsed>"] += 1

    # Participant-level unresolved counts.
    unresolved_by_participant = Counter()
    for aid in unresolved_metadata:
        p = parse_acquisition(aid)
        unresolved_by_participant[p["participant"] if p else "<unparsed>"] += 1

    raw_only_by_participant = Counter()
    for aid in raw_only_after_resolution:
        p = parse_acquisition(aid)
        raw_only_by_participant[p["participant"] if p else "<unparsed>"] += 1

    # Check whether normalized matching creates duplicate collisions.
    duplicate_meta_keys = {
        str(k): v for k, v in meta_by_key.items() if len(v) > 1
    }
    duplicate_raw_keys = {
        str(k): v for k, v in raw_by_key.items() if len(v) > 1
    }

    report = {
        "dataset": "AICOS-HAR",
        "protocol": "reconciliation_v1",
        "metadata_rows": len(metadata),
        "metadata_unique_acquisitions": len(meta_set),
        "inventory_rows": len(inventory),
        "inventory_unique_acquisitions": len(inv_set),
        "exact_match_count": len(exact_meta),
        "exact_metadata_missing_count": len(exact_missing),
        "exact_raw_only_count": len(exact_raw_only),
        "normalized_resolved_count": len(normalized_resolved),
        "ambiguous_normalized_count": len(ambiguous_metadata),
        "unresolved_metadata_count": len(unresolved_metadata),
        "raw_only_after_resolution_count": len(raw_only_after_resolution),
        "normalized_match_rate_of_exact_missing": (
            len(normalized_resolved) / len(exact_missing)
            if exact_missing else 1.0
        ),
        "activity_diagnostics": {
            "unresolved_metadata": unresolved_meta_activity,
            "raw_only_after_resolution": raw_only_activity,
        },
        "participant_diagnostics": {
            "unresolved_metadata": unresolved_by_participant,
            "raw_only_after_resolution": raw_only_by_participant,
        },
        "duplicate_normalized_metadata_keys": duplicate_meta_keys,
        "duplicate_normalized_raw_keys": duplicate_raw_keys,
        "ambiguous_matches": ambiguous_metadata,
        "unresolved_metadata_sample": unresolved_metadata[:500],
        "raw_only_after_resolution_sample": raw_only_after_resolution[:500],
        "raw_data_scanned": False,
    }

    # Full raw scan is only for confirming the inventory itself, not samples.
    # This scans filenames, not sensor contents.
    raw_ids = raw_acquisitions(data_dir)
    report["raw_filename_acquisition_count"] = len(raw_ids)
    report["inventory_vs_raw_filename_set_difference"] = {
        "inventory_not_in_raw_filename_scan": len(inv_set - raw_ids),
        "raw_filename_not_in_inventory": len(raw_ids - inv_set),
    }

    # Resolution table.
    resolution_rows = []

    for aid in sorted(meta_set):
        if aid in exact_meta:
            status = "exact_match"
            raw_match = aid
        else:
            candidates = [
                x["raw_acquisition_id"]
                for x in normalized_resolved
                if x["metadata_acquisition_id"] == aid
            ]
            if len(candidates) == 1:
                status = "normalized_match"
                raw_match = candidates[0]
            elif any(
                x["metadata_acquisition_id"] == aid
                for x in ambiguous_metadata
            ):
                status = "ambiguous_normalized_match"
                raw_match = ""
            else:
                status = "unresolved"
                raw_match = ""

        resolution_rows.append({
            "metadata_acquisition_id": aid,
            "status": status,
            "raw_acquisition_id": raw_match,
        })

    write_csv(
        out / "aicos_har_acquisition_reconciliation.csv",
        resolution_rows,
        ["metadata_acquisition_id", "status", "raw_acquisition_id"],
    )

    # Save concise lists for inspection.
    (out / "unresolved_metadata_ids.txt").write_text(
        "\n".join(unresolved_metadata) + ("\n" if unresolved_metadata else ""),
        encoding="utf-8",
    )

    (out / "raw_only_ids_after_resolution.txt").write_text(
        "\n".join(raw_only_after_resolution) + ("\n" if raw_only_after_resolution else ""),
        encoding="utf-8",
    )

    (out / "normalized_matches.csv").write_text(
        "metadata_acquisition_id,raw_acquisition_id,resolution\n"
        + "\n".join(
            f'"{x["metadata_acquisition_id"]}","{x["raw_acquisition_id"]}","{x["resolution"]}"'
            for x in normalized_resolved
        )
        + ("\n" if normalized_resolved else ""),
        encoding="utf-8",
    )

    (out / "aicos_har_reconciliation_v1.json").write_text(
        json.dumps(report, indent=2, default=lambda x: dict(x)),
        encoding="utf-8",
    )

    print("AICOS-HAR RECONCILIATION V1")
    print("=" * 35)
    print(f"Metadata unique acquisitions : {len(meta_set)}")
    print(f"Inventory unique acquisitions: {len(inv_set)}")
    print(f"Exact matches                : {len(exact_meta)}")
    print(f"Exact metadata missing       : {len(exact_missing)}")
    print(f"Normalized resolved          : {len(normalized_resolved)}")
    print(f"Ambiguous normalized         : {len(ambiguous_metadata)}")
    print(f"Still unresolved             : {len(unresolved_metadata)}")
    print(f"Raw-only after resolution    : {len(raw_only_after_resolution)}")
    print()
    print("Unresolved metadata by activity:")
    for k, v in unresolved_meta_activity.most_common():
        print(f"  {k}: {v}")
    print()
    print("Raw-only after resolution by activity:")
    for k, v in raw_only_activity.most_common():
        print(f"  {k}: {v}")
    print()
    print(f"Reports written to: {out}")


if __name__ == "__main__":
    main()
