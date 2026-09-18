from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 42
CLASSES = ["LYING", "SITTING", "STANDING"]
N_CLASSES = 3
N_CHANNELS = 6

def load_data(path=None):
    path = Path(path or ROOT / "datasets/processed/pamap2/stationary_accgyro_windows.npz")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=True) as z:
        missing = {"X", "y", "groups"} - set(z.files)
        if missing:
            raise ValueError(f"NPZ missing required keys: {sorted(missing)}")
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        groups = np.asarray(z["groups"]).astype(str)

    if X.ndim != 3 or X.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected X shape (N,6,T); got {X.shape}")
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups lengths do not match.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf.")
    if not np.isin(y, np.arange(N_CLASSES)).all():
        raise ValueError(f"Labels must be 0,1,2; found {np.unique(y)}")
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least 5 subjects for the outer benchmark.")
    return X, y, groups

def grouped_splits(X, y, groups, n_splits):
    from sklearn.model_selection import StratifiedGroupKFold
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"Requested {n_splits} folds but only {n_groups} groups exist.")
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=SEED
    )
    # MATERIALIZE the generator immediately.
    return list(cv.split(X, y, groups))

def standardize_train(X_train, *others):
    """Fit normalization on X_train only; return transformed train + others."""
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def transform(a):
        if a is None:
            return None
        return ((a - mean) / std).astype(np.float32)

    return transform(X_train), *(transform(a) for a in others)

def features(X):
    """
    Explicit 109-feature contract.

    8 streams (ax,ay,az,gx,gy,gz,acc_mag,gyro_mag) x 11 stats = 88
    6 aggregate features = 6
    15 pairwise raw-channel correlations = 15
    Total = 109.
    """
    rows = []
    for w in X:
        acc_mag = np.linalg.norm(w[:3], axis=0)
        gyro_mag = np.linalg.norm(w[3:], axis=0)
        streams = [*w, acc_mag, gyro_mag]
        f = []

        for a in streams:
            d = np.diff(a)
            f.extend([
                float(np.mean(a)), float(np.std(a)),
                float(np.min(a)), float(np.max(a)),
                float(np.sqrt(np.mean(a * a))),
                float(np.percentile(a, 10)),
                float(np.percentile(a, 25)),
                float(np.percentile(a, 50)),
                float(np.percentile(a, 75)),
                float(np.percentile(a, 90)),
                float(np.mean(np.abs(d))) if len(d) else 0.0,
            ])

        f.extend([
            float(np.mean(acc_mag)),
            float(np.std(acc_mag)),
            float(np.ptp(acc_mag)),
            float(np.mean(np.abs(np.diff(acc_mag)))),
            float(np.mean(gyro_mag)),
            float(np.std(gyro_mag)),
        ])

        for i in range(6):
            for j in range(i + 1, 6):
                a, b = w[i], w[j]
                sa, sb = np.std(a), np.std(b)
                if sa < 1e-8 or sb < 1e-8:
                    c = 0.0
                else:
                    c = float(np.corrcoef(a, b)[0, 1])
                    if not np.isfinite(c):
                        c = 0.0
                f.append(c)

        if len(f) != 109:
            raise AssertionError(f"Feature contract broken: {len(f)}")
        rows.append(f)

    F = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(F).all():
        raise ValueError("Handcrafted features contain NaN/Inf.")
    return F

def metrics(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_recall_fscore_support, confusion_matrix
    )
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": [
            {"class": CLASSES[i], "precision": float(p[i]),
             "recall": float(r[i]), "f1": float(f[i])}
            for i in range(N_CLASSES)
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(N_CLASSES)
        ).tolist(),
    }
