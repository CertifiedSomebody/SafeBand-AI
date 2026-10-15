"""Audit the handcrafted ACC feature matrix used by the AICOS-HAR benchmark.

This specifically checks the downstream feature-generation stage that previously
emitted np.corrcoef warnings, while leaving the raw window dataset untouched.
"""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_corr(a, b):
    sa = np.std(a)
    sb = np.std(b)
    if not np.isfinite(sa) or not np.isfinite(sb) or sa == 0 or sb == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def feature_vector(w, fs=50.0):
    ax, ay, az = w
    mag = np.sqrt(np.sum(w * w, axis=0))
    d = np.diff(w, axis=1)
    mag_d = np.diff(mag)
    f = []
    for a in (ax, ay, az):
        f += [a.mean(), a.std(), a.min(), a.max(), np.sqrt(np.mean(a * a))]
    f += [
        mag.mean(), mag.std(), mag.min(), mag.max(), np.sqrt(np.mean(mag * mag)),
        *np.percentile(mag, [10, 25, 50, 75, 90]),
        np.mean(np.abs(w)), mag.max() - mag.min(),
        np.mean(np.abs(mag_d)), np.std(mag_d), np.max(np.abs(mag_d)),
        np.sum(d * d),
        np.sum(np.diff(np.sign(mag - np.mean(mag))) != 0),
    ]
    for a, b in ((ax, ay), (ax, az), (ay, az)):
        f.append(safe_corr(a, b))
    spec = np.abs(np.fft.rfft(mag - np.mean(mag))) ** 2
    freq = np.fft.rfftfreq(len(mag), 1 / fs)
    for lo, hi in ((0, 2), (2, 5), (5, 10)):
        f.append(spec[(freq >= lo) & (freq < hi)].sum())
    return np.asarray(f, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="reports/aicos_har_feature_audit_v1.json")
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    X = np.load(args.npz, allow_pickle=False)["X"]
    with open(args.manifest, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(X) != len(rows):
        raise SystemExit(f"Window/manifest mismatch: {len(X)} vs {len(rows)}")

    n = min(len(X), max(1, args.sample))
    idx = np.linspace(0, len(X) - 1, n, dtype=np.int64)
    F = np.stack([feature_vector(X[i]) for i in idx])
    finite = np.isfinite(F)
    result = {
        "windows_total": int(len(X)),
        "windows_sampled": int(n),
        "feature_shape_sample": list(F.shape),
        "feature_count": int(F.shape[1]),
        "nan_values": int(np.isnan(F).sum()),
        "inf_values": int(np.isinf(F).sum()),
        "nonfinite_rows": int((~finite.all(axis=1)).sum()),
        "feature_min": np.nanmin(F, axis=0).tolist(),
        "feature_max": np.nanmax(F, axis=0).tolist(),
        "status": "PASS" if finite.all() else "FAIL",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Audit report: {out}")
    print("PASS: sampled handcrafted features are finite." if result["status"] == "PASS" else "FAIL: non-finite handcrafted features detected.")


if __name__ == "__main__":
    main()
