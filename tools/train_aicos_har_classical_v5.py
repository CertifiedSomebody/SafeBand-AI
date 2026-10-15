"""AICOS-HAR ACC-only classical ML benchmark V5.

Purpose: test whether fixed classical models outperform the existing TinyCNN/DAE
on exactly the same official test protocol.

Protocol:
- ACC-only, 50 Hz, 2 s windows, 1 s step
- official Fold=test is untouched until final evaluation
- all non-test folds are training data
- fixed model configurations; no test-driven tuning
- robust handcrafted features with safe correlation calculation
"""
from pathlib import Path
import argparse, csv, json, sys, time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_corr(a, b):
    sa, sb = np.std(a), np.std(b)
    if not np.isfinite(sa) or not np.isfinite(sb) or sa == 0 or sb == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def features(X, fs=50.0):
    out = np.empty((len(X), 38), dtype=np.float32)
    for i, w in enumerate(X):
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
        out[i] = f
    if not np.isfinite(out).all():
        raise RuntimeError("Feature matrix contains non-finite values after safe feature extraction.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="models/aicos_har/classical_v5")
    ap.add_argument("--skip-rbf", action="store_true", help="Skip RBF-SVM if runtime is a concern.")
    args = ap.parse_args()

    t0 = time.time()
    X = np.load(args.npz, allow_pickle=False)["X"]
    with open(args.manifest, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(X) != len(rows):
        raise SystemExit(f"Window/manifest mismatch: {len(X)} vs {len(rows)}")

    y = np.asarray([r["activity"] for r in rows])
    fold = np.char.lower(np.asarray([r["fold"] for r in rows], dtype=str))
    test = fold == "test"
    train = ~test
    if not test.any():
        raise SystemExit("No official Fold=test samples found.")

    print("Extracting 38 robust ACC features...")
    F = features(X)
    print(f"Feature matrix: {F.shape}; finite=True")
    classes = sorted(np.unique(y))

    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)),
        ]),
        "linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)),
        ]),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced", max_depth=24, min_samples_leaf=5, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, random_state=42),
    }
    if not args.skip_rbf:
        models["rbf_svm"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=4.0, gamma="scale", class_weight="balanced")),
        ])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol": "AICOS-HAR ACC-only classical benchmark V5",
        "test_partition": "official Fold=test",
        "train_windows": int(train.sum()),
        "test_windows": int(test.sum()),
        "feature_count": int(F.shape[1]),
        "models": {},
    }

    for name, model in models.items():
        start = time.time()
        print(f"\n=== {name} ===")
        model.fit(F[train], y[train])
        pred = model.predict(F[test])
        metrics = {
            "accuracy": float(accuracy_score(y[test], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "macro_f1": float(f1_score(y[test], pred, average="macro")),
            "classes": classes,
            "confusion_matrix": confusion_matrix(y[test], pred, labels=classes).tolist(),
            "elapsed_s": float(time.time() - start),
        }
        results["models"][name] = metrics
        joblib.dump(model, out / f"{name}.joblib")
        print(f"accuracy={metrics['accuracy']:.6f} balanced={metrics['balanced_accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}")
        print(f"elapsed={metrics['elapsed_s']:.1f}s")

    results["elapsed_total_s"] = float(time.time() - t0)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nOfficial test samples:", int(test.sum()))
    print("Training samples:", int(train.sum()))
    print("Results:", out / "results.json")


if __name__ == "__main__":
    main()
