"""
SafeBand AI - PAMAP2 Stationary V3
Forensic subject audit + leave-one-subject-out sensitivity preparation.

Run from repository root:
    python tools\audit_pamap2_stationary_v3.py

The script is intentionally diagnostic only. It does NOT modify the source NPZ.
"""
from pathlib import Path
import sys, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets" / "processed" / "pamap2" / "stationary_accgyro_windows.npz"
OUT = ROOT / "models" / "pamap2_stationary_v3"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["LYING", "SITTING", "STANDING"]

def finite_stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }

def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Run this command from the SafeBand-AI repository root and verify "
            "datasets\\processed\\pamap2\\stationary_accgyro_windows.npz exists."
        )

    z = np.load(DATA, allow_pickle=True)
    required = {"X", "y", "groups"}
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"NPZ missing keys {sorted(missing)}; found {z.files}")

    X = np.asarray(z["X"])
    y = np.asarray(z["y"]).astype(int)
    groups = np.asarray(z["groups"]).astype(str)

    if X.ndim != 3 or X.shape[1] != 6:
        raise ValueError(f"Expected X shape (N,6,T), got {X.shape}")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, groups lengths do not match.")
    if sorted(np.unique(y).tolist()) != [0, 1, 2]:
        raise ValueError(f"Expected labels 0,1,2; found {sorted(np.unique(y).tolist())}")

    finite_fraction = np.isfinite(X).mean(axis=(1,2))
    bad_windows = np.where(finite_fraction < 1.0)[0]

    acc = X[:, 0:3, :]
    gyro = X[:, 3:6, :]
    acc_mag = np.sqrt(np.sum(acc * acc, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro * gyro, axis=1))

    # Window-level features useful for diagnosing orientation/domain shift.
    acc_mean = acc.mean(axis=2)
    gyro_mean = gyro.mean(axis=2)
    acc_std = acc.std(axis=2)
    gyro_std = gyro.std(axis=2)
    acc_mag_mean = acc_mag.mean(axis=1)
    acc_mag_std = acc_mag.std(axis=1)
    gyro_mag_mean = gyro_mag.mean(axis=1)
    gyro_mag_std = gyro_mag.std(axis=1)

    subjects = sorted(np.unique(groups), key=lambda s: (len(s), s))
    subject_class_counts = {}
    feature_rows = []

    for s in subjects:
        mask_s = groups == s
        subject_class_counts[s] = {
            CLASSES[c]: int(np.sum(mask_s & (y == c))) for c in range(3)
        }
        for c in range(3):
            m = mask_s & (y == c)
            if not np.any(m):
                continue
            row = {
                "subject": s,
                "class": CLASSES[c],
                "windows": int(m.sum()),
            }
            for i, name in enumerate(["ax_mean","ay_mean","az_mean"]):
                row[name] = float(acc_mean[m, i].mean())
                row[name + "_std_between_windows"] = float(acc_mean[m, i].std())
            for i, name in enumerate(["gx_mean","gy_mean","gz_mean"]):
                row[name] = float(gyro_mean[m, i].mean())
                row[name + "_std_between_windows"] = float(gyro_mean[m, i].std())
            row["acc_mag_mean"] = float(acc_mag_mean[m].mean())
            row["acc_mag_std"] = float(acc_mag_std[m].mean())
            row["gyro_mag_mean"] = float(gyro_mag_mean[m].mean())
            row["gyro_mag_std"] = float(gyro_mag_std[m].mean())
            feature_rows.append(row)

    # Overall and subject-specific distributions.
    global_stats = {}
    for name, arr in {
        "acc_mag_mean": acc_mag_mean,
        "acc_mag_std": acc_mag_std,
        "gyro_mag_mean": gyro_mag_mean,
        "gyro_mag_std": gyro_mag_std,
    }.items():
        global_stats[name] = finite_stats(arr)

    subject_stats = {}
    for s in subjects:
        m = groups == s
        subject_stats[s] = {
            "windows": int(m.sum()),
            "acc_mag_mean": finite_stats(acc_mag_mean[m]),
            "gyro_mag_mean": finite_stats(gyro_mag_mean[m]),
        }

    # Leave-one-subject-out is diagnostic, not the deployment benchmark.
    loo_plan = []
    for s in subjects:
        train_subjects = [x for x in subjects if x != s]
        loo_plan.append({
            "test_subject": s,
            "train_subjects": train_subjects,
            "train_windows": int(np.sum(groups != s)),
            "test_windows": int(np.sum(groups == s)),
        })

    report = {
        "experiment": "pamap2_stationary_v3_forensic_audit",
        "dataset": str(DATA),
        "samples": int(len(X)),
        "shape": list(X.shape),
        "subjects": subjects,
        "classes": CLASSES,
        "bad_nonfinite_windows": int(len(bad_windows)),
        "subject_class_counts": subject_class_counts,
        "global_magnitude_stats": global_stats,
        "subject_magnitude_stats": subject_stats,
        "loo_subject_plan": loo_plan,
        "notes": [
            "Source NPZ is read-only; no samples are removed by this audit.",
            "Subject 5 is retained in the canonical dataset.",
            "A later no-subject-5 experiment must be reported as a sensitivity analysis, not as a replacement benchmark.",
            "LOO subject evaluation is diagnostic and intentionally separate from the 5-fold benchmark."
        ]
    }

    (OUT / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # CSV-like tables without pandas dependency.
    import csv
    with open(OUT / "subject_class_feature_summary.csv", "w", newline="", encoding="utf-8") as f:
        if feature_rows:
            writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            writer.writeheader()
            writer.writerows(feature_rows)

    print("PAMAP2 Stationary V3 forensic audit completed.")
    print(f"Repository root : {ROOT}")
    print(f"Dataset         : {DATA}")
    print(f"Samples         : {len(X)}")
    print(f"Shape           : {X.shape}")
    print(f"Subjects        : {subjects}")
    print(f"Non-finite wins : {len(bad_windows)}")
    print("\nSubject x class windows:")
    for s in subjects:
        print(f"  Subject {s}: " + ", ".join(
            f"{CLASSES[c]}={subject_class_counts[s][CLASSES[c]]}" for c in range(3)
        ))
    print(f"\nReport: {OUT / 'audit_report.json'}")
    print(f"CSV   : {OUT / 'subject_class_feature_summary.csv'}")

if __name__ == "__main__":
    main()
