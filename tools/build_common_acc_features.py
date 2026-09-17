"""Build the common ACC-only 20-Hz / 40-sample feature dataset from FORTH-TRACE.

The source is recorded at 51.2 Hz. A 40-sample target window at 20 Hz is
~2.0 s, so each source window uses ~102 raw samples (~2.0 s) and is then
resampled to exactly 40 samples. This avoids the earlier V1 bug where a
40-sample raw block (~0.78 s) was resampled and then discarded for being too
short.

Usage:
  python tools/build_common_acc_features.py
"""
from __future__ import annotations
from pathlib import Path
import argparse, re, sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ai.common_acc_features import extract_acc_features, FEATURES

CORE = {
    "sit": "SITTING", "sit and talk": "SITTING",
    "walk": "WALKING", "walk and talk": "WALKING",
    "2": "SITTING", "3": "SITTING",
    "4": "WALKING", "5": "WALKING",
}
FS_RAW = 51.2
FS_TARGET = 20.0
WINDOW = 40
STEP = 20
MIN_PURITY = 0.90


def resample_1d(x: np.ndarray, n: int) -> np.ndarray:
    old = np.linspace(0.0, 1.0, len(x), endpoint=False)
    new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(new, old, x)


def resample_block(a: np.ndarray, n: int) -> np.ndarray:
    return np.column_stack([resample_1d(a[:, j], n) for j in range(3)])


def find_csvs(root: Path):
    return sorted(root.glob("part*/part*dev1.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets/raw/FORTH_TRACE/FORTH_TRACE_DATASET-master")
    ap.add_argument("--out", default="datasets/processed/activity_common_acc/forth_trace_common_acc.csv")
    ap.add_argument("--window", type=int, default=WINDOW, help="target samples at 20 Hz")
    ap.add_argument("--step", type=int, default=STEP, help="target step at 20 Hz")
    ap.add_argument("--purity", type=float, default=MIN_PURITY)
    args = ap.parse_args()

    if args.window <= 0 or args.step <= 0 or not (0.5 <= args.purity <= 1.0):
        raise SystemExit("Invalid --window/--step/--purity")

    root = ROOT / args.data_dir
    out = ROOT / args.out
    files = find_csvs(root)
    if not files:
        raise SystemExit(f"No FORTH-TRACE dev1 files found under {root}")

    raw_window = max(2, int(round(args.window * FS_RAW / FS_TARGET)))
    raw_step = max(1, int(round(args.step * FS_RAW / FS_TARGET)))

    rows = []
    audit = []
    for f in files:
        pid = re.search(r"part(\d+)", f.parent.name, re.I)
        subject = str(int(pid.group(1))) if pid else f.parent.name
        df = pd.read_csv(f)
        if df.shape[1] < 12:
            raise ValueError(f"{f} has {df.shape[1]} columns; expected at least 12")

        acc = df.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # FORTH-TRACE commonly stores activity labels as numeric codes in column 12.
        # Accept both numeric codes and textual labels so the builder is robust to
        # the distributed CSV variants.
        label_col = None
        normalized_cols = {str(c).strip().lower(): c for c in df.columns}
        for key in ("activity label", "activity_label", "activity", "label", "class"):
            if key in normalized_cols:
                label_col = normalized_cols[key]
                break
        if label_col is None:
            label_col = df.columns[11]
        labels = df[label_col].astype(str).str.strip().str.lower()
        kept = 0
        rejected_nan = 0
        rejected_label = 0
        rejected_purity = 0

        for i in range(0, len(df) - raw_window + 1, raw_step):
            block = acc[i:i + raw_window]
            lab = labels.iloc[i:i + raw_window]

            if np.isnan(block).any():
                rejected_nan += 1
                continue

            def normalize_label(x):
                x = str(x).strip().lower()
                if x in CORE:
                    return CORE[x]
                try:
                    fx = float(x)
                    if fx.is_integer():
                        return CORE.get(str(int(fx)))
                except (TypeError, ValueError):
                    pass
                return None

            mapped = [normalize_label(x) for x in lab]
            valid = [x for x in mapped if x is not None]
            if len(valid) < int(np.ceil(args.purity * raw_window)):
                rejected_label += 1
                continue

            counts = pd.Series(valid).value_counts()
            chosen = str(counts.index[0])
            dominant = int(counts.iloc[0])
            if dominant < int(np.ceil(args.purity * raw_window)):
                rejected_purity += 1
                continue

            rb = resample_block(block, args.window)
            row = extract_acc_features(rb, fs=FS_TARGET)
            row.update({
                "activity_label": chosen,
                "subject_id": subject,
                "source": "FORTH_TRACE",
                "window_start_raw": i,
                "window_samples_raw": raw_window,
            })
            rows.append(row)
            kept += 1

        audit.append({
            "subject": subject,
            "file": str(f.relative_to(ROOT)),
            "windows": kept,
            "rejected_nan": rejected_nan,
            "rejected_label": rejected_label,
            "rejected_purity": rejected_purity,
        })

    dfout = pd.DataFrame(rows, columns=FEATURES + [
        "activity_label", "subject_id", "source", "window_start_raw", "window_samples_raw"
    ])

    if dfout.empty:
        raise SystemExit(
            "No valid windows were produced. Check the FORTH-TRACE path/columns and label mapping. "
            f"Expected raw_window={raw_window} samples (~{raw_window / FS_RAW:.2f}s) "
            f"for target window={args.window} at {FS_TARGET} Hz."
        )

    classes = sorted(dfout.activity_label.unique().tolist())
    subjects = sorted(dfout.subject_id.astype(str).unique().tolist())
    if classes != ["SITTING", "WALKING"]:
        raise SystemExit(f"Common-label build must contain both SITTING and WALKING; got {classes}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(out.stem + ".tmp.csv")
    tmp_audit = out.with_suffix(".audit.tmp.json")
    dfout.to_csv(tmp_out, index=False)
    audit_path = out.with_suffix(".audit.json")
    audit_payload = {
        "contract": "SafeBand common ACC V1.2",
        "source_rate_hz": FS_RAW,
        "target_rate_hz": FS_TARGET,
        "target_window_samples": args.window,
        "target_step_samples": args.step,
        "raw_window_samples": raw_window,
        "raw_step_samples": raw_step,
        "window_seconds": args.window / FS_TARGET,
        "step_seconds": args.step / FS_TARGET,
        "purity_threshold": args.purity,
        "classes": classes,
        "rows": len(dfout),
        "subjects": subjects,
        "files": audit,
        "label_column_used": str(label_col) if "label_col" in locals() else "auto",
        "numeric_label_mapping": {"2":"SITTING","3":"SITTING","4":"WALKING","5":"WALKING"},
    }
    tmp_audit.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    tmp_out.replace(out)
    tmp_audit.replace(audit_path)

    print(dfout.groupby(["activity_label", "subject_id"]).size().unstack(fill_value=0))
    print(f"\nSaved: {out}")
    print(f"Rows={len(dfout)} Subjects={dfout.subject_id.nunique()} Classes={classes}")
    print(f"Resampling: {FS_RAW:.1f} Hz -> {FS_TARGET:.1f} Hz | raw_window={raw_window} | raw_step={raw_step}")
    print(f"Label column: {label_col}")


if __name__ == "__main__":
    main()
