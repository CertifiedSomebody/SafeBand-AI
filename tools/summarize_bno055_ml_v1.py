
"""Summarize completed SafeBand BNO055 ML V1 JSON reports.

This utility deliberately does not rank models or declare a winner. It reports
the measured metrics and sensor-channel configurations side by side.
"""
from pathlib import Path
import argparse, json, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT/"reports/bno055_ml_v1")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON reports found in {args.input_dir}")

    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d["pooled_outer"]
        rows.append({
            "report": f.name,
            "model": d["model"],
            "channels": d["channels"],
            "samples": d["samples"],
            "subjects": d["subjects"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "event_overlap_n": int(sum(
                (row["event_metrics"]["n"] if row.get("event_metrics") else 0)
                for row in d.get("folds_detail", [])
            )),
            "event_macro_f1": (
                d["pooled_event_overlap"]["macro_f1"]
                if d.get("pooled_event_overlap") else None
            ),
        })

    df = pd.DataFrame(rows).sort_values(["model", "channels"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"[PASS] Summary CSV: {args.output}")

if __name__ == "__main__":
    main()
