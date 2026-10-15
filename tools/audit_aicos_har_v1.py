"""
AICOS-HAR forensic dataset audit.

Usage from repository root:
    python tools/audit_aicos_har_v1.py ^
        --data-dir datasets/raw/AICOS-HAR/AICOS-HAR ^
        --metadata datasets/raw/AICOS-HAR/AICOS-HAR/metadata.csv ^
        --readme datasets/raw/AICOS-HAR/AICOS-HAR/README.txt ^
        --out reports/aicos_har_audit_v1

The script is intentionally an audit only:
- no resampling
- no windowing
- no ML training
- no modification of raw files
- no random train/test split

It inspects metadata, raw hierarchy, sensor files, timestamps, sampling intervals,
acquisition durations, participant/fold relationships, device/position structure,
and possible leakage routes.
"""

from pathlib import Path
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None

SENSOR_ALIASES = {
    "accelerometer": "acc",
    "acceleration": "acc",
    "acc": "acc",
    "accel": "acc",
    "gyroscope": "gyro",
    "gyroscope.txt": "gyro",
    "gyro": "gyro",
    "magnetometer": "mag",
    "mag": "mag",
    "barometer": "baro",
    "baro": "baro",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--readme", default=None)
    p.add_argument("--out", default="reports/aicos_har_audit_v1")
    p.add_argument("--max-files", type=int, default=0,
                   help="0 = inspect all sensor files")
    return p.parse_args()

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip())

def infer_sensor(path):
    stem = path.stem.lower()
    return SENSOR_ALIASES.get(stem, stem)

def read_metadata(path):
    if pd is not None:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [norm(c) for c in df.columns]
        return df

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header = [norm(x) for x in rows[0]]
    return [dict(zip(header, r)) for r in rows[1:]], header

def metadata_records(meta):
    if pd is not None and hasattr(meta, "to_dict"):
        return meta.to_dict("records")
    return meta

def acquisition_id_from_path(path, data_dir):
    rel = path.relative_to(data_dir)
    # Expected: Participant/Activity_Trial/Device_Position/Sensor.txt
    parts = rel.parts
    if len(parts) >= 4:
        return "/".join(parts[:3])
    return None

def parse_sensor_file(path):
    timestamps = []
    rows = 0
    malformed = 0
    numeric_values = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Accept whitespace, tab, comma, semicolon and pipe separators.
            fields = [x for x in re.split(r"[\t,;| ]+", s) if x != ""]
            if not fields:
                continue
            # Skip a header row if present.
            try:
                ts = float(fields[0])
            except ValueError:
                continue
            rows += 1
            try:
                vals = [float(x) for x in fields[1:]]
            except ValueError:
                malformed += 1
                continue
            timestamps.append(ts)
            numeric_values.extend(vals)

    result = {
        "rows": rows,
        "malformed_numeric_rows": malformed,
        "timestamp_count": len(timestamps),
    }
    if len(timestamps) >= 2:
        diffs = [b-a for a,b in zip(timestamps, timestamps[1:]) if b > a]
        result["timestamp_monotonic"] = all(
            b > a for a,b in zip(timestamps, timestamps[1:])
        )
        result["positive_dt_count"] = len(diffs)
        if diffs:
            med = statistics.median(diffs)
            result["median_dt_raw"] = med
            result["min_dt_raw"] = min(diffs)
            result["max_dt_raw"] = max(diffs)
            # README says ns. Convert ns -> Hz using median interval.
            result["estimated_hz_from_ns"] = 1e9 / med if med > 0 else None
            if len(diffs) >= 10:
                result["dt_cv"] = (
                    statistics.pstdev(diffs) / med if med else None
                )
        result["duration_s_from_ns"] = (
            (timestamps[-1] - timestamps[0]) / 1e9
            if timestamps[-1] >= timestamps[0] else None
        )
    else:
        result["timestamp_monotonic"] = None
    return result

def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise SystemExit(f"ERROR: data directory does not exist: {data_dir}")
    if not metadata_path.exists():
        raise SystemExit(f"ERROR: metadata does not exist: {metadata_path}")

    meta = read_metadata(metadata_path)
    records = metadata_records(meta)

    report = {
        "dataset": "AICOS-HAR",
        "protocol": "forensic_audit_v1",
        "raw_data_dir": str(data_dir),
        "metadata_file": str(metadata_path),
        "metadata_columns": list(meta.columns) if pd is not None else meta[1],
        "metadata_rows": len(records),
        "metadata_sample": records[:5],
        "unique_values": {},
        "relationships": {},
        "raw": {},
        "warnings": [],
    }

    columns = report["metadata_columns"]
    for c in columns:
        vals = [norm(r.get(c, "")) for r in records]
        nonempty = [v for v in vals if v]
        report["unique_values"][c] = {
            "nonempty": len(nonempty),
            "unique": len(set(nonempty)),
            "top20": Counter(nonempty).most_common(20),
        }

    # Parse acquisition identifiers conservatively from metadata.
    acq_col = next((c for c in columns if c.lower() == "acquisitionid"), None)
    fold_col = next((c for c in columns if c.lower() == "fold"), None)
    pos_col = next((c for c in columns if "position" in c.lower()), None)

    parsed = []
    for r in records:
        acq = norm(r.get(acq_col, "")) if acq_col else ""
        parts = acq.split("/")
        participant = parts[0] if len(parts) >= 1 else ""
        activity_trial = parts[1] if len(parts) >= 2 else ""
        device_position = parts[2] if len(parts) >= 3 else ""
        activity = activity_trial.rsplit("_", 1)[0] if "_" in activity_trial else activity_trial
        parsed.append({
            "acquisition_id": acq,
            "participant": participant,
            "activity_trial": activity_trial,
            "activity": activity,
            "device_position_from_id": device_position,
            "fold": norm(r.get(fold_col, "")) if fold_col else "",
            "position_metadata": norm(r.get(pos_col, "")) if pos_col else "",
        })

    report["relationships"]["participants"] = sorted(
        set(x["participant"] for x in parsed if x["participant"])
    )
    report["relationships"]["activities"] = Counter(
        x["activity"] for x in parsed if x["activity"]
    )
    report["relationships"]["folds"] = Counter(
        x["fold"] for x in parsed if x["fold"]
    )

    participant_folds = defaultdict(set)
    participant_activities = defaultdict(set)
    participant_devices = defaultdict(set)
    for x in parsed:
        if x["participant"]:
            participant_folds[x["participant"]].add(x["fold"])
            participant_activities[x["participant"]].add(x["activity"])
            participant_devices[x["participant"]].add(x["device_position_from_id"])

    report["relationships"]["participant_fold_map"] = {
        k: sorted(v) for k,v in sorted(participant_folds.items())
    }
    report["relationships"]["participants_in_multiple_folds"] = {
        k: sorted(v) for k,v in participant_folds.items() if len(v) > 1
    }
    report["relationships"]["participants_activity_counts"] = {
        k: len(v) for k,v in participant_activities.items()
    }
    report["relationships"]["participants_device_position_counts"] = {
        k: len(v) for k,v in participant_devices.items()
    }

    if report["relationships"]["participants_in_multiple_folds"]:
        report["warnings"].append(
            "Some participants appear in multiple Fold values. Do not use Fold as a subject-independent split without investigating this."
        )

    # Raw sensor inventory.
    sensor_files = []
    for p in data_dir.rglob("*.txt"):
        if p.name.lower() in {"readme.txt", "metadata.txt"}:
            continue
        sensor_files.append(p)
    sensor_files.sort()

    if args.max_files:
        inspect_files = sensor_files[:args.max_files]
        report["raw"]["inspection_limited"] = True
    else:
        inspect_files = sensor_files
        report["raw"]["inspection_limited"] = False

    sensor_counter = Counter()
    raw_acq_counter = Counter()
    malformed_paths = []
    per_sensor_stats = defaultdict(list)

    for i, path in enumerate(inspect_files, 1):
        sensor = infer_sensor(path)
        sensor_counter[sensor] += 1
        aid = acquisition_id_from_path(path, data_dir)
        if aid:
            raw_acq_counter[aid] += 1
        try:
            stats = parse_sensor_file(path)
        except Exception as exc:
            malformed_paths.append({"path": str(path), "error": repr(exc)})
            continue
        stats["path"] = str(path)
        stats["sensor"] = sensor
        stats["acquisition_id"] = aid
        per_sensor_stats[sensor].append(stats)

    report["raw"]["sensor_file_count"] = len(sensor_files)
    report["raw"]["inspected_sensor_file_count"] = len(inspect_files)
    report["raw"]["sensor_file_counts"] = dict(sensor_counter)
    report["raw"]["acquisition_file_counts_top20"] = raw_acq_counter.most_common(20)
    report["raw"]["malformed_files"] = malformed_paths

    summary = {}
    for sensor, stats_list in per_sensor_stats.items():
        hz = [x["estimated_hz_from_ns"] for x in stats_list
              if x.get("estimated_hz_from_ns") is not None and math.isfinite(x["estimated_hz_from_ns"])]
        dur = [x["duration_s_from_ns"] for x in stats_list
               if x.get("duration_s_from_ns") is not None and math.isfinite(x["duration_s_from_ns"])]
        mono_bad = sum(x.get("timestamp_monotonic") is False for x in stats_list)
        summary[sensor] = {
            "files": len(stats_list),
            "estimated_hz_median_across_files": statistics.median(hz) if hz else None,
            "estimated_hz_min": min(hz) if hz else None,
            "estimated_hz_max": max(hz) if hz else None,
            "duration_s_median": statistics.median(dur) if dur else None,
            "duration_s_min": min(dur) if dur else None,
            "duration_s_max": max(dur) if dur else None,
            "non_monotonic_timestamp_files": mono_bad,
        }
    report["raw"]["sensor_summary"] = summary

    # Compare metadata acquisition count with raw acquisition directories.
    raw_acq_ids = set(raw_acq_counter)
    meta_acq_ids = set(x["acquisition_id"] for x in parsed if x["acquisition_id"])
    report["relationships"]["metadata_acquisition_count"] = len(meta_acq_ids)
    report["relationships"]["raw_acquisition_count"] = len(raw_acq_ids)
    report["relationships"]["metadata_acquisitions_missing_from_raw"] = sorted(
        meta_acq_ids - raw_acq_ids
    )[:200]
    report["relationships"]["raw_acquisitions_missing_from_metadata"] = sorted(
        raw_acq_ids - meta_acq_ids
    )[:200]

    if report["relationships"]["metadata_acquisitions_missing_from_raw"]:
        report["warnings"].append("Some metadata acquisitions were not found in the inspected raw hierarchy.")
    if report["relationships"]["raw_acquisitions_missing_from_metadata"]:
        report["warnings"].append("Some raw acquisitions were not found in metadata.")

    # Human-readable summary.
    lines = [
        "# AICOS-HAR Forensic Audit V1",
        "",
        f"- Metadata rows: **{len(records)}**",
        f"- Metadata columns: `{', '.join(columns)}`",
        f"- Raw sensor files found: **{len(sensor_files)}**",
        f"- Sensor files inspected: **{len(inspect_files)}**",
        "",
        "## Metadata unique-value summary",
    ]
    for c, info in report["unique_values"].items():
        lines.append(f"- **{c}**: {info['unique']} unique non-empty values")
    lines += [
        "",
        "## Activities",
        "",
        "```text",
        json.dumps(report["relationships"]["activities"], indent=2),
        "```",
        "",
        "## Fold distribution",
        "",
        "```text",
        json.dumps(report["relationships"]["folds"], indent=2),
        "```",
        "",
        "## Raw sensor summary",
        "",
        "```text",
        json.dumps(report["raw"]["sensor_summary"], indent=2),
        "```",
        "",
        "## Leakage warnings",
    ]
    if report["warnings"]:
        lines.extend([f"- {w}" for w in report["warnings"]])
    else:
        lines.append("- None triggered by this first-pass audit.")

    (out / "aicos_har_audit_v1.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out / "AICOS_HAR_AUDIT_V1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nReports written to: {out}")

if __name__ == "__main__":
    main()
