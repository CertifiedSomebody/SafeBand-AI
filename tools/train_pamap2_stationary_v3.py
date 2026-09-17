from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


# This file is intentionally placed directly under tools/.
# Therefore parents[1] is the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "datasets" / "processed" / "pamap2" / "stationary_accgyro_windows.npz"
DEFAULT_OUT = ROOT / "models" / "pamap2_stationary_v2"

CLASS_NAMES = ["LYING", "SITTING", "STANDING"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_npz(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("X", "y", "groups")
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"NPZ is missing required keys: {missing}")

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    groups = np.asarray(data["groups"]).astype(str)

    if X.ndim != 3 or X.shape[1] != 6:
        raise RuntimeError(
            f"Expected X with shape (N, 6, T) for ACC+GYRO, got {X.shape}"
        )
    if X.shape[0] != len(y) or X.shape[0] != len(groups):
        raise RuntimeError(
            f"Length mismatch: X={X.shape[0]}, y={len(y)}, groups={len(groups)}"
        )
    if X.shape[2] < 20:
        raise RuntimeError(f"Window is unexpectedly short: {X.shape}")
    if not np.isfinite(X).all():
        raise RuntimeError("X contains non-finite values.")
    if not set(np.unique(y)).issubset({0, 1, 2}):
        raise RuntimeError(f"Expected labels 0,1,2; got {np.unique(y)}")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise RuntimeError(
            f"Need at least 5 subjects for 5-fold grouped CV; found {len(unique_groups)}"
        )

    # Group-feasibility preflight. Every class must occur in enough distinct
    # subjects for the requested outer/inner grouped CV.
    for label in np.unique(y):
        n_groups = len(np.unique(groups[y == label]))
        if n_groups < 5:
            raise RuntimeError(
                f"Class {label} occurs in only {n_groups} subjects. "
                "Cannot safely run 5-fold subject-independent CV."
            )
        if n_groups < 4:
            raise RuntimeError(
                f"Class {label} occurs in only {n_groups} subjects. "
                "Cannot safely run the 4-fold inner grouped CV."
            )

    return X, y, groups


def safe_std(x: np.ndarray) -> float:
    v = float(np.std(x))
    return v if v > 1e-8 else 1e-8


def window_features(X: np.ndarray) -> np.ndarray:
    """
    Convert (N, 6, T) raw hand IMU windows to a compact feature matrix.

    Channels:
      0:3 = ACC
      3:6 = GYRO

    Features are computed independently for each window.
    """
    if X.ndim != 3 or X.shape[1] != 6:
        raise ValueError(f"Expected (N,6,T), got {X.shape}")

    n, _, t = X.shape
    out = np.empty((n, 0), dtype=np.float32)
    rows = []

    # Fixed, auditable feature ordering.
    for w in X:
        acc = w[:3]
        gyro = w[3:]

        f = []

        # Per-axis time statistics.
        for sig in (acc, gyro):
            for axis in range(3):
                s = sig[axis]
                f.extend([
                    float(np.mean(s)),
                    float(np.std(s)),
                    float(np.min(s)),
                    float(np.max(s)),
                    float(np.sqrt(np.mean(s * s))),
                    float(np.percentile(s, 10)),
                    float(np.percentile(s, 50)),
                    float(np.percentile(s, 90)),
                ])

        # Vector magnitudes: orientation-independent motion intensity.
        acc_mag = np.linalg.norm(acc, axis=0)
        gyro_mag = np.linalg.norm(gyro, axis=0)

        for mag in (acc_mag, gyro_mag):
            f.extend([
                float(np.mean(mag)),
                float(np.std(mag)),
                float(np.min(mag)),
                float(np.max(mag)),
                float(np.sqrt(np.mean(mag * mag))),
                float(np.percentile(mag, 10)),
                float(np.percentile(mag, 50)),
                float(np.percentile(mag, 90)),
            ])

        # Direction of the mean acceleration vector.
        # This is useful for posture because the quasi-static gravity
        # component changes its direction relative to the wrist axes.
        mean_acc = np.mean(acc, axis=1)
        denom = safe_std(np.linalg.norm(mean_acc))
        # For a vector, use its own norm; safe_std is only a convenient
        # positive floor function.
        denom = max(float(np.linalg.norm(mean_acc)), 1e-8)
        unit_acc = mean_acc / denom
        f.extend([float(v) for v in unit_acc])

        # Pairwise ACC covariance: captures coordinated axis motion.
        cov = np.cov(acc)
        f.extend([
            float(cov[0, 1]),
            float(cov[0, 2]),
            float(cov[1, 2]),
        ])

        # Low-frequency spectral energy ratios.
        # For a 2 s stationary window this summarizes slow posture/motion
        # variation without relying on exact phase.
        for sig in (acc_mag, gyro_mag):
            centered = sig - np.mean(sig)
            spec = np.abs(np.fft.rfft(centered)) ** 2
            total = float(np.sum(spec[1:]))  # exclude DC
            if total <= 1e-12:
                low = mid = high = 0.0
            else:
                freqs = np.fft.rfftfreq(t, d=1.0 / 100.0)
                low = float(np.sum(spec[(freqs >= 0.5) & (freqs < 2.0)]) / total)
                mid = float(np.sum(spec[(freqs >= 2.0) & (freqs < 5.0)]) / total)
                high = float(np.sum(spec[freqs >= 5.0]) / total)
            f.extend([low, mid, high])

        rows.append(f)

    out = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("Feature extraction produced non-finite values.")
    return out


def evaluate(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y_true, pred, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, pred, labels=[0, 1, 2]
        ).tolist(),
    }

def explicit_class_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import precision_score, recall_score

    return {
        "per_class_precision": precision_score(
            y_true, pred, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, pred, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            y_true, pred, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
    }


def select_params(X_train, y_train, groups_train, seed: int) -> tuple[float, str, float]:
    """
    Inner grouped CV. Selection is made only from the outer training set.
    Returns C, gamma, and best macro-F1.
    """
    candidates = [
        (0.5, "scale"),
        (1.0, "scale"),
        (2.0, "scale"),
        (4.0, "scale"),
        (8.0, "scale"),
        (1.0, 0.01),
        (2.0, 0.01),
        (4.0, 0.01),
        (2.0, 0.1),
        (4.0, 0.1),
    ]

    inner = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=seed
    )

    best_score = -np.inf
    best = candidates[0]

    for C, gamma in candidates:
        scores = []
        for itr, iva in inner.split(X_train, y_train, groups_train):
            model = Pipeline([
                ("scale", StandardScaler()),
                ("svc", SVC(
                    C=C,
                    gamma=gamma,
                    kernel="rbf",
                    class_weight="balanced",
                )),
            ])
            model.fit(X_train[itr], y_train[itr])
            pred = model.predict(X_train[iva])
            scores.append(f1_score(
                y_train[iva],
                pred,
                average="macro",
                zero_division=0,
            ))

        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best = (C, gamma)

    return float(best[0]), best[1], best_score


def run_benchmark(F, y, groups, seed, label):
    """Unbiased 5-fold grouped benchmark for one subject set."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pooled_true, pooled_pred, folds = [], [], []

    for fold, (tr, te) in enumerate(cv.split(F, y, groups), start=1):
        print(f"\n===== {label}: OUTER FOLD {fold}/5 =====")
        C, gamma, inner_score = select_params(F[tr], y[tr], groups[tr], seed + fold)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced")),
        ])
        model.fit(F[tr], y[tr])
        pred = model.predict(F[te])
        m = evaluate(y[te], pred)
        m.update(explicit_class_metrics(y[te], pred))
        m.update({
            "fold": fold,
            "selected_C": C,
            "selected_gamma": gamma,
            "inner_best_macro_f1": inner_score,
            "test_samples": int(len(te)),
            "test_subjects": sorted(np.unique(groups[te]).tolist()),
        })
        folds.append(m)
        pooled_true.extend(y[te].tolist())
        pooled_pred.extend(pred.tolist())
        print(f"Fold {fold}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
        print("  test subjects:", m["test_subjects"])
        print("  F1:", [round(v,4) for v in m["per_class_f1"]])

    yt, yp = np.asarray(pooled_true), np.asarray(pooled_pred)
    pooled = evaluate(yt, yp)
    pooled.update(explicit_class_metrics(yt, yp))
    return {"label": label, "subjects": sorted(np.unique(groups).tolist()), "samples": int(len(y)), "folds": folds, "pooled": pooled}


def main() -> None:
    ap = argparse.ArgumentParser(description="PAMAP2 Stationary V3 subject-5 sensitivity benchmark.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "pamap2_stationary_v3")
    ap.add_argument("--test-subject", default="5", help="Subject to isolate as the diagnostic held-out subject.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    data_path = args.data.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found:\n  {data_path}\nPass --data with the NPZ path if necessary.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root : {ROOT}")
    print(f"Dataset         : {data_path}")
    print(f"Output          : {out_dir}")
    data = np.load(data_path, allow_pickle=True)
    X, y, groups = validate_npz(data)
    del data
    subjects = sorted(np.unique(groups).tolist(), key=lambda s: (len(s), s))
    if args.test_subject not in subjects:
        raise RuntimeError(f"Requested test subject {args.test_subject!r} not found. Available: {subjects}")

    print(f"Raw X shape     : {X.shape}")
    print(f"Samples         : {len(y):,}")
    print(f"Subjects        : {subjects}")
    print("\nExtracting the EXACT V2 76-feature representation...")
    F = window_features(X)
    if F.shape[1] != 76:
        raise RuntimeError(f"V2 feature contract violation: expected 76 features, got {F.shape[1]}")
    np.save(out_dir / "stationary_v3_features.npy", F)
    print(f"Feature matrix  : {F.shape}")

    # Primary control: exact V2 protocol on all 8 subjects.
    all_result = run_benchmark(F, y, groups, args.seed, "ALL SUBJECTS CONTROL")

    # Diagnostic: Subject 5 is a completely unseen test subject in a dedicated run.
    # This is not used for hyperparameter selection.
    test_mask = groups == args.test_subject
    train_mask = ~test_mask
    Xf_train, y_train, g_train = F[train_mask], y[train_mask], groups[train_mask]
    Xf_test, y_test = F[test_mask], y[test_mask]

    # Fit hyperparameters ONLY on the 7 training subjects, then evaluate on S5.
    C, gamma, inner_score = select_params(Xf_train, y_train, g_train, args.seed + 500)
    test_model = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced")),
    ])
    test_model.fit(Xf_train, y_train)
    pred_test = test_model.predict(Xf_test)
    isolated = evaluate(y_test, pred_test)
    isolated.update(explicit_class_metrics(y_test, pred_test))
    isolated.update({
        "test_subject": args.test_subject,
        "train_subjects": sorted(np.unique(g_train).tolist()),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "selected_C": C,
        "selected_gamma": gamma,
        "inner_best_macro_f1": inner_score,
    })

    # Sensitivity: remove the diagnostic subject, then perform the same 5-fold grouped CV.
    clean_result = run_benchmark(F[train_mask], y[train_mask], groups[train_mask], args.seed, f"EXCLUDING SUBJECT {args.test_subject}")

    report = {
        "experiment": "pamap2_stationary_v3_subject5_diagnostic",
        "model": "same_76_feature_rbf_svm_as_v2",
        "seed": args.seed,
        "data": str(data_path),
        "feature_count": 76,
        "validation_control": "Exact V2 5-fold StratifiedGroupKFold benchmark on all subjects",
        "canonical_all_subjects": all_result,
        "isolated_test_subject": isolated,
        "sensitivity_without_test_subject": clean_result,
        "delta_without_subject_vs_all": {
            "accuracy_pp": (clean_result["pooled"]["accuracy"] - all_result["pooled"]["accuracy"]) * 100,
            "balanced_accuracy_pp": (clean_result["pooled"]["balanced_accuracy"] - all_result["pooled"]["balanced_accuracy"]) * 100,
            "macro_f1_pp": (clean_result["pooled"]["macro_f1"] - all_result["pooled"]["macro_f1"]) * 100,
        },
        "interpretation_guardrail": "Subject 5 remains part of the canonical dataset. Excluding it is a sensitivity analysis. It may only become a formal exclusion if independent data-quality evidence justifies that decision.",
    }
    report_path = out_dir / "pamap2_stationary_v3_subject5_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== SUBJECT 5 AS A COMPLETELY UNSEEN TEST SUBJECT =====")
    print(f"Accuracy          : {isolated['accuracy']:.4%}")
    print(f"Balanced accuracy : {isolated['balanced_accuracy']:.4%}")
    print(f"Macro F1          : {isolated['macro_f1']:.4%}")
    print("Class F1          :", [f"{CLASS_NAMES[i]}={isolated['per_class_f1'][i]:.4f}" for i in range(3)])
    print("Confusion matrix:")
    print(np.asarray(isolated["confusion_matrix"]))
    print("\n===== SENSITIVITY: SUBJECT 5 REMOVED FROM DATASET =====")
    print(f"Accuracy          : {clean_result['pooled']['accuracy']:.4%}")
    print(f"Balanced accuracy : {clean_result['pooled']['balanced_accuracy']:.4%}")
    print(f"Macro F1          : {clean_result['pooled']['macro_f1']:.4%}")
    d = report["delta_without_subject_vs_all"]
    print("\nDelta vs all-subject control:")
    print(f"Accuracy          : {d['accuracy_pp']:+.2f} pp")
    print(f"Balanced accuracy : {d['balanced_accuracy_pp']:+.2f} pp")
    print(f"Macro F1          : {d['macro_f1_pp']:+.2f} pp")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
