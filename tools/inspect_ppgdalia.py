from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np


EXPECTED_SUBJECTS = [f"S{i}" for i in range(1, 16)]
EXPECTED_ACTIVITY_IDS = set(range(9))


def parse_args():
    p = argparse.ArgumentParser(
        description="Forensic audit of the extracted PPG-DaLiA PPG_FieldStudy dataset."
    )
    p.add_argument(
        "--root",
        default=r"datasets/raw/PPG-DaLiA/ppg_dalia/data/PPG_FieldStudy",
        help="Path to PPG_FieldStudy.",
    )
    p.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subjects to inspect, e.g. S1 S2. Default: all discovered subjects.",
    )
    p.add_argument(
        "--json",
        default=None,
        help="Optional path for a machine-readable audit report.",
    )
    return p.parse_args()


def load_pickle(path: Path):
    # PPG-DaLiA was released as a Python-2-era pickle. latin1 preserves
    # the byte-string representation required by Python 3.
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def describe_array(x):
    if not isinstance(x, np.ndarray):
        return {
            "type": type(x).__name__,
            "shape": None,
            "dtype": None,
        }

    out = {
        "type": "ndarray",
        "shape": list(x.shape),
        "dtype": str(x.dtype),
    }

    if x.size:
        # Avoid expensive/unsafe statistics on non-numeric arrays.
        if np.issubdtype(x.dtype, np.number):
            finite = np.isfinite(x)
            out["finite_fraction"] = float(finite.mean())
            if finite.any():
                xf = x[finite]
                out["min"] = float(np.min(xf))
                out["max"] = float(np.max(xf))
                out["mean"] = float(np.mean(xf))
                out["std"] = float(np.std(xf))
    return out


def describe_tree(obj, prefix=""):
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            key = str(k)
            result[key] = describe_tree(v, prefix + key + ".")
        return result

    if isinstance(obj, np.ndarray):
        return describe_array(obj)

    if isinstance(obj, (list, tuple)):
        result = {
            "type": type(obj).__name__,
            "length": len(obj),
        }
        # Describe only a small number of elements; never repr a huge array.
        if len(obj) <= 20:
            result["items"] = [describe_tree(v, prefix) for v in obj]
        return result

    if np.isscalar(obj):
        return {
            "type": type(obj).__name__,
            "value": obj.item() if isinstance(obj, np.generic) else obj,
        }

    return {"type": type(obj).__name__}


def get_nested(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def subject_audit(subject_dir: Path):
    sid = subject_dir.name
    pkl = subject_dir / f"{sid}.pkl"

    result = {
        "subject": sid,
        "files": {},
        "errors": [],
        "derived": {},
    }

    for name in [
        f"{sid}.pkl",
        f"{sid}_activity.csv",
        f"{sid}_quest.csv",
        f"{sid}_RespiBAN.h5",
        f"{sid}_E4.zip",
    ]:
        path = subject_dir / name
        result["files"][name] = {
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 3)
            if path.exists()
            else None,
        }

    if not pkl.exists():
        result["errors"].append(f"Missing {pkl}")
        return result

    print(f"\n{'=' * 72}\n{sid}\n{'=' * 72}")
    print(f"Loading {pkl} ({result['files'][pkl.name]['size_mb']} MB)...")

    obj = load_pickle(pkl)

    result["pickle_type"] = type(obj).__name__
    result["top_level_keys"] = list(obj.keys()) if isinstance(obj, dict) else None

    if not isinstance(obj, dict):
        result["errors"].append("Top-level pickle object is not a dictionary.")
        del obj
        gc.collect()
        return result

    # Official schema checks.
    required = {"activity", "label", "questionnaire", "rpeaks", "signal", "subject"}
    missing = sorted(required - set(obj.keys()))
    if missing:
        result["errors"].append(f"Missing official top-level keys: {missing}")

    wrist = get_nested(obj, "signal", "wrist")
    chest = get_nested(obj, "signal", "chest")

    if not isinstance(wrist, dict):
        result["errors"].append("signal.wrist is not a dictionary.")
        wrist = {}
    if not isinstance(chest, dict):
        result["errors"].append("signal.chest is not a dictionary.")
        chest = {}

    result["wrist_keys"] = list(wrist.keys())
    result["chest_keys"] = list(chest.keys())

    for k in ["BVP", "ACC", "EDA", "TEMP"]:
        if k in wrist:
            result[f"wrist_{k}"] = describe_array(wrist[k])
        else:
            result["errors"].append(f"Missing wrist/{k}")

    # The released pickle uses "Resp" and "Temp" capitalization, while the
    # README describes these modalities as RESP/TEMP. These are not required
    # for the first wrist BVP+ACC HR experiment.
    chest_aliases = {
        "ACC": ["ACC"],
        "ECG": ["ECG"],
        "EMG": ["EMG"],
        "EDA": ["EDA"],
        "RESP": ["RESP", "Resp"],
        "TEMP": ["TEMP", "Temp"],
    }
    for canonical, aliases in chest_aliases.items():
        found = next((a for a in aliases if a in chest), None)
        if found is not None:
            result[f"chest_{canonical}"] = describe_array(chest[found])
        elif canonical in ("ACC", "ECG"):
            result["errors"].append(f"Missing required chest/{canonical}")
        else:
            result[f"chest_{canonical}"] = {
                "status": "not_present",
                "note": "Not required for the first wrist BVP+ACC experiment."
            }

    for k in ["activity", "label", "rpeaks"]:
        if k in obj:
            result[k] = describe_array(obj[k])

    result["subject_value"] = (
        obj["subject"].item() if isinstance(obj["subject"], np.generic)
        else obj.get("subject")
    )

    # Strong consistency checks based on the official specification.
    bvp = wrist.get("BVP")
    wacc = wrist.get("ACC")
    label = obj.get("label")
    activity = obj.get("activity")
    rpeaks = obj.get("rpeaks")
    chest_ecg = chest.get("ECG")

    if isinstance(bvp, np.ndarray):
        result["derived"]["bvp_duration_sec"] = float(len(bvp) / 64.0)
    if isinstance(wacc, np.ndarray):
        n_acc = wacc.shape[0] if wacc.ndim >= 1 else 0
        result["derived"]["wrist_acc_duration_sec"] = float(n_acc / 32.0)

    if isinstance(chest_ecg, np.ndarray):
        n_ecg = chest_ecg.shape[0] if chest_ecg.ndim >= 1 else 0
        result["derived"]["ecg_duration_sec"] = float(n_ecg / 700.0)

    if isinstance(label, np.ndarray) and label.ndim >= 1:
        n = label.shape[0]
        result["derived"]["label_count"] = int(n)
        if isinstance(bvp, np.ndarray):
            expected = max(0, (len(bvp) - 8 * 64) // (2 * 64) + 1)
            result["derived"]["expected_label_count_from_bvp"] = int(expected)
            result["derived"]["label_count_matches_bvp_8s2s"] = bool(n == expected)

    if isinstance(activity, np.ndarray):
        result["derived"]["activity_count"] = int(activity.shape[0])
        if activity.size and np.issubdtype(activity.dtype, np.number):
            vals, counts = np.unique(activity, return_counts=True)
            result["activity_distribution"] = {
                str(v.item() if isinstance(v, np.generic) else v): int(c)
                for v, c in zip(vals, counts)
            }
            bad = [int(v) for v in vals if int(v) not in EXPECTED_ACTIVITY_IDS]
            if bad:
                result["errors"].append(
                    f"Unexpected activity IDs: {bad}"
                )

    if isinstance(rpeaks, np.ndarray) and rpeaks.size:
        if np.issubdtype(rpeaks.dtype, np.integer):
            result["derived"]["rpeak_count"] = int(rpeaks.size)
            result["derived"]["rpeak_min"] = int(np.min(rpeaks))
            result["derived"]["rpeak_max"] = int(np.max(rpeaks))
            if isinstance(chest_ecg, np.ndarray) and rpeaks.max() >= len(chest_ecg):
                result["errors"].append("Some rpeak indices exceed ECG length.")

    # Print concise human-readable summary.
    print("Top-level keys:", result["top_level_keys"])
    print("Wrist keys:", result["wrist_keys"])
    print("Chest keys:", result["chest_keys"])
    for key in ["wrist_BVP", "wrist_ACC", "chest_ECG", "label", "activity", "rpeaks"]:
        print(f"{key}: {result.get(key)}")
    print("Derived:", result.get("derived", {}))
    if result.get("activity_distribution"):
        print("Activity distribution:", result["activity_distribution"])
    if result["errors"]:
        print("ERRORS:")
        for e in result["errors"]:
            print("  -", e)
    else:
        print("Status: PASS")

    del obj
    gc.collect()
    return result


def main():
    args = parse_args()
    root = Path(args.root)

    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")

    discovered = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("S") and p.name[1:].isdigit()
    )

    if args.subjects:
        wanted = set(args.subjects)
        subjects = [p for p in discovered if p.name in wanted]
        missing = sorted(wanted - {p.name for p in subjects})
        if missing:
            raise SystemExit(f"Requested subjects not found: {missing}")
    else:
        subjects = discovered

    print("PPG-DaLiA forensic audit")
    print("Root:", root.resolve())
    print("Discovered subjects:", [p.name for p in subjects])
    print("Expected official subjects:", EXPECTED_SUBJECTS)

    report = {
        "root": str(root.resolve()),
        "subjects_discovered": [p.name for p in subjects],
        "subjects": [],
    }

    for subject in subjects:
        try:
            report["subjects"].append(subject_audit(subject))
        except Exception as exc:
            print(f"\nFAIL {subject}: {type(exc).__name__}: {exc}")
            report["subjects"].append({
                "subject": subject.name,
                "errors": [f"{type(exc).__name__}: {exc}"],
            })
            gc.collect()

    passed = sum(not s.get("errors") for s in report["subjects"])
    failed = len(report["subjects"]) - passed

    report["summary"] = {
        "subjects": len(report["subjects"]),
        "passed": passed,
        "failed": failed,
    }

    print("\n" + "=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)
    print(json.dumps(report["summary"], indent=2))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("JSON report:", out.resolve())

    if failed:
        print("\nAudit completed with failures. Do not build the training dataset yet.")
        return 2

    print("\nAudit completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
