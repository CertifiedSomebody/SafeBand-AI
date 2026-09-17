from pathlib import Path
import sys, json, re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CSV = REPO_ROOT / "datasets" / "raw" / "PAMAP2" / "dataset2.csv"

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(columns, *patterns):
    n = {norm(c): c for c in columns}
    for p in patterns:
        np = norm(p)
        if np in n:
            return n[np]
    for c in columns:
        nc = norm(c)
        if all(norm(p) in nc for p in patterns):
            return c
    return None

def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Auditing: {csv_path}")
    head = pd.read_csv(csv_path, nrows=5)
    cols = list(head.columns)

    activity_col = find_col(cols, "activityID") or find_col(cols, "activity")
    people_col = find_col(cols, "PeopleId") or find_col(cols, "person") or find_col(cols, "subject")

    hand_acc = [find_col(cols, "hand acceleration X"),
                find_col(cols, "hand acceleration Y"),
                find_col(cols, "hand acceleration Z")]
    hand_gyro = [find_col(cols, "hand gyroscope X"),
                 find_col(cols, "hand gyroscope Y"),
                 find_col(cols, "hand gyroscope Z")]
    hand_mag = [find_col(cols, "hand magnetometer X"),
                find_col(cols, "hand magnetometer Y"),
                find_col(cols, "hand magnetometer Z")]

    print("\nCOLUMNS:")
    for i, c in enumerate(cols):
        print(f"{i:3d}: {c}")

    print("\nDETECTED SCHEMA:")
    print("activity:", activity_col)
    print("people:", people_col)
    print("hand_acc:", hand_acc)
    print("hand_gyro:", hand_gyro)
    print("hand_mag:", hand_mag)

    missing = [x for x in [activity_col, people_col, *hand_acc, *hand_gyro] if x is None]
    if missing:
        raise RuntimeError("Required activity/PeopleId/hand ACC+GYRO columns could not be detected.")

    report = {
        "csv": str(csv_path),
        "columns": cols,
        "detected": {
            "activity": activity_col,
            "people": people_col,
            "hand_acc": hand_acc,
            "hand_gyro": hand_gyro,
            "hand_mag": hand_mag,
        },
        "chunk_rows": 100000,
        "people": {},
        "activity_counts": {},
        "missing_required": {},
        "rows": 0,
    }

    # Chunked so the ~800 MB CSV is never loaded fully into RAM.
    for chunk in pd.read_csv(csv_path, chunksize=100000, low_memory=False):
        report["rows"] += len(chunk)

        for p, n in chunk[people_col].value_counts(dropna=False).items():
            k = str(p)
            report["people"][k] = report["people"].get(k, 0) + int(n)

        for a, n in chunk[activity_col].value_counts(dropna=False).items():
            k = str(a)
            report["activity_counts"][k] = report["activity_counts"].get(k, 0) + int(n)

        for c in [activity_col, people_col, *hand_acc, *hand_gyro]:
            report["missing_required"][c] = report["missing_required"].get(c, 0) + int(chunk[c].isna().sum())

    print("\nROWS:", report["rows"])
    print("\nPEOPLE:")
    for k, v in sorted(report["people"].items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v:,}")

    print("\nACTIVITY COUNTS:")
    for k, v in sorted(report["activity_counts"].items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v:,}")

    print("\nMISSING REQUIRED VALUES:")
    for k, v in report["missing_required"].items():
        print(f"  {k}: {v:,}")

    out = REPO_ROOT / "models" / "pamap2_stationary_v1"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "pamap2_schema_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {report_path}")

if __name__ == "__main__":
    main()
