
"""Common utilities for the SafeBand BNO055 ML benchmark."""

from pathlib import Path
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHANNELS = [
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
]

CHANNEL_SETS = {
    "acc": [0, 1, 2],
    "accgyro": [0, 1, 2, 3, 4, 5],
    "accgyromag": list(range(9)),
}

CLASS_NAMES = [
    "FALL", "LYING", "RUNNING", "SITTING", "SIT_TO_STAND",
    "STAIRS", "STANDING", "STAND_TO_SIT", "WALKING",
]


def canonicalize_x(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X with 3 dimensions, got {X.shape}.")
    if X.shape[1] == 9:
        return X
    if X.shape[2] == 9:
        return np.transpose(X, (0, 2, 1)).copy()
    raise ValueError(
        f"Expected channel dimension of 9 in X, got shape {X.shape}."
    )


def load_windows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Window dataset not found: {path}")
    d = np.load(path, allow_pickle=False)
    required = {"X", "y", "groups"}
    missing = required.difference(d.files)
    if missing:
        raise ValueError(f"NPZ missing required arrays: {sorted(missing)}")
    X = canonicalize_x(d["X"])
    y = np.asarray(d["y"]).astype(np.int64)
    groups = np.asarray(d["groups"]).astype(np.int64)
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must have equal length.")
    if len(X) == 0:
        raise ValueError("Window dataset is empty.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")
    if not np.isfinite(y).all() or not np.isfinite(groups).all():
        raise ValueError("y/groups contain non-finite values.")
    sessions = np.asarray(d["sessions"]) if "sessions" in d.files else np.full(len(y), -1)
    event = np.asarray(d["is_event_window"]).astype(bool) if "is_event_window" in d.files else np.zeros(len(y), bool)
    if len(sessions) != len(y):
        raise ValueError("sessions must have the same length as y.")
    if len(event) != len(y):
        raise ValueError("is_event_window must have the same length as y.")
    labels = d["classes"].tolist() if "classes" in d.files else CLASS_NAMES
    labels = [str(x) for x in labels]
    return X, y, groups, sessions, event, labels


def standardize_train_only(X_train, X_test):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((X_train - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32), mean, std


def extract_features(X):
    """Deterministic handcrafted features computed independently per window."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("Expected X=(N,C,T).")
    feats = []
    eps = 1e-8
    for w in X:
        f = []
        # Per-channel statistical/energy features.
        for c in range(w.shape[0]):
            s = w[c]
            f.extend([
                float(np.mean(s)),
                float(np.std(s)),
                float(np.min(s)),
                float(np.max(s)),
                float(np.sqrt(np.mean(s*s))),
                float(np.percentile(s, 10)),
                float(np.percentile(s, 50)),
                float(np.percentile(s, 90)),
                float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0,
                float(np.sum(np.diff(s)**2)) if len(s) > 1 else 0.0,
            ])
        # Vector magnitudes for ACC/GYRO/MAG when available.
        if w.shape[0] >= 3:
            groups = [(0,1,2)]
            if w.shape[0] >= 6:
                groups.append((3,4,5))
            if w.shape[0] >= 9:
                groups.append((6,7,8))
            for a,b,c in groups:
                m = np.sqrt(w[a]**2 + w[b]**2 + w[c]**2)
                f.extend([
                    float(m.mean()), float(m.std()), float(m.min()),
                    float(m.max()), float(np.sqrt(np.mean(m*m))),
                    float(np.percentile(m, 10)), float(np.percentile(m, 50)),
                    float(np.percentile(m, 90)),
                ])
        # Pairwise correlations, with safe handling of constant channels.
        for i in range(w.shape[0]):
            for j in range(i+1, w.shape[0]):
                a, b = w[i], w[j]
                sa, sb = np.std(a), np.std(b)
                corr = 0.0 if sa < eps or sb < eps else float(np.corrcoef(a, b)[0,1])
                f.append(corr if np.isfinite(corr) else 0.0)
        feats.append(f)
    out = np.asarray(feats, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Feature extraction produced non-finite values.")
    return out


def metrics(y_true, y_pred, class_names):
    """Compute fixed-vocabulary multiclass metrics without sklearn warning paths."""
    from sklearn.metrics import accuracy_score, confusion_matrix

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    # Balanced accuracy convention: mean recall over classes actually present
    # in y_true. This avoids undefined-class warnings for event-only subsets.
    present = support > 0
    balanced = float(recall[present].mean()) if present.any() else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class_f1": f1.tolist(),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
        "class_support": support.astype(int).tolist(),
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
