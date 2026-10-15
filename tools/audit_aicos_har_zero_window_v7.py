from pathlib import Path
import argparse
import csv
import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd


ALIASES = {
    "LiftDown": "ElevatorDown",
    "LiftUp": "ElevatorUp",
}


def norm_activity(name: str) -> str:
    return ALIASES.get(name, name)


def raw_candidates(data_dir: Path, acquisition_id: str):
    parts = Path(acquisition_id).parts
    if len(parts) < 3:
        return []
    participant, trial, device = parts[-3], parts[-2], parts[-1]
    activity = trial.rsplit("_", 1)[0] if "_" in trial else trial
    trial_num = trial.rsplit("_", 1)[1] if "_" in trial and trial.rsplit("_", 1)[1].isdigit() else None
    raw_activity = norm_activity(activity)
    raw_trial = f"{raw_activity}_{trial_num}" if trial_num else raw_activity
    base = data_dir / participant / raw_trial / device

    # AICOS-HAR long sensor filenames vary by release. Search only within the
    # exact acquisition directory and accept files whose name identifies ACC.
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_file() and re.search(r"(accelerometer|acc)", p.name, re.I))


def read_sensor(path: Path):
    # Robust delimiter/column handling for AICOS-HAR inertial text files.
    df = pd.read_csv(path, sep=None, engine="python")
    cols = list(df.columns)
    lower = {str(c).strip().lower(): c for c in cols}

    tcol = next((lower[k] for k in lower if "timestamp" in k), None)
    xcol = next((lower[k] for k in lower if k in ("sensor_x", "x", "acc_x", "accelerometer_x")), None)
    ycol = next((lower[k] for k in lower if k in ("sensor_y", "y", "acc_y", "accelerometer_y")), None)
    zcol = next((lower[k] for k in lower if k in ("sensor_z", "z", "acc_z", "accelerometer_z")), None)

    if not all((tcol, xcol, ycol, zcol)):
        # Positional fallback: timestamp + xyz
        if len(cols) < 4:
            raise ValueError(f"Cannot identify timestamp/XYZ columns: {cols}")
        tcol, xcol, ycol, zcol = cols[:4]

    out = df[[tcol, xcol, ycol, zcol]].copy()
    out.columns = ["timestamp", "x", "y", "z"]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    return out


def timestamp_seconds(ts):
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 2:
        return ts
    # AICOS-HAR timestamps are documented as ns; use magnitude-based fallback
    # only to make the diagnostic robust to exported variants.
    scale = 1e9 if np.nanmedian(np.abs(ts)) > 1e12 else 1.0
    return ts / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, default=2.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    metadata_path = Path(args.metadata)
    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_path)
    man = pd.read_csv(manifest_path)

    if "AcquisitionID" not in meta.columns:
        raise ValueError(f"Metadata missing AcquisitionID. Columns: {list(meta.columns)}")

    # Manifest V4/V5 naming compatibility.
    m_acq = next((c for c in ["acquisition_id", "AcquisitionID", "acquisition"] if c in man.columns), None)
    if m_acq is None:
        raise ValueError(f"Manifest missing acquisition identifier. Columns: {list(man.columns)}")

    meta_ids = set(meta["AcquisitionID"].astype(str))
    man_ids = set(man[m_acq].astype(str))
    missing = sorted(meta_ids - man_ids)

    details = []
    for aid in missing:
        row = meta.loc[meta["AcquisitionID"].astype(str) == aid].iloc[0]
        candidates = raw_candidates(data_dir, aid)

        if not candidates:
            reason = "raw_acc_not_found"
            details.append({
                "AcquisitionID": aid,
                "activity": row.get("Activity", row.get("activity", "")),
                "fold": row.get("Fold", ""),
                "position": row.get("Position_FullName", ""),
                "acc_file": "",
                "samples": 0,
                "duration_s": 0.0,
                "median_hz": 0.0,
                "usable_for_2s": False,
                "reason": reason,
            })
            continue

        best = candidates[0]
        try:
            d = read_sensor(best)
            t = timestamp_seconds(d["timestamp"].to_numpy())
            t = t[np.isfinite(t)]
            samples = len(t)
            duration = float(t[-1] - t[0]) if samples >= 2 else 0.0
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            hz = float(1.0 / np.median(dt)) if len(dt) else 0.0

            if samples < 2:
                reason = "too_few_samples"
            elif duration < args.window_s:
                reason = "duration_below_window"
            elif not dt.size:
                reason = "invalid_timestamps"
            else:
                # If duration is sufficient, the zero-window result likely comes
                # from a resampling/implementation edge case rather than raw length.
                reason = "builder_edge_case"
            usable = duration >= args.window_s
        except Exception as exc:
            samples, duration, hz, usable = 0, 0.0, 0.0, False
            reason = f"read_error:{type(exc).__name__}"

        details.append({
            "AcquisitionID": aid,
            "activity": row.get("Activity", row.get("activity", "")),
            "fold": row.get("Fold", ""),
            "position": row.get("Position_FullName", ""),
            "acc_file": str(best),
            "samples": int(samples),
            "duration_s": round(duration, 6),
            "median_hz": round(hz, 6),
            "usable_for_2s": bool(usable),
            "reason": reason,
        })

    counts = Counter(d["reason"] for d in details)
    result = {
        "window_seconds": args.window_s,
        "metadata_acquisitions": len(meta_ids),
        "manifest_acquisitions": len(man_ids),
        "zero_window_acquisitions": len(missing),
        "reasons": dict(counts),
        "details": details,
        "status": "PASS" if not any(k == "builder_edge_case" for k in counts) else "INVESTIGATE_BUILDER",
    }

    out_path.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(details).to_csv(out_path.with_suffix(".csv"), index=False)

    print("AICOS-HAR ZERO-WINDOW DIAGNOSTIC V7")
    print("=" * 40)
    print(f"Metadata acquisitions : {len(meta_ids)}")
    print(f"Manifest acquisitions : {len(man_ids)}")
    print(f"Zero-window           : {len(missing)}")
    print(f"Reasons               : {dict(counts)}")
    for d in details:
        print(f"  {d['AcquisitionID']} | {d['duration_s']:.3f}s | {d['samples']} samples | {d['reason']}")
    print(f"JSON report: {out_path.with_suffix('.json')}")
    print(f"CSV report : {out_path.with_suffix('.csv')}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
