from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["LYING", "SITTING", "STANDING"]
N_CLASSES = len(CLASSES)
SEED = 42

def load_data(path=None):
    path = Path(path or ROOT / "datasets/processed/pamap2/stationary_accgyro_windows.npz")
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)
    y = np.asarray(z["y"], dtype=np.int64)
    groups = np.asarray(z["groups"]).astype(str)
    if X.ndim != 3 or X.shape[1] != 6:
        raise ValueError(f"Expected X=(N,6,T), got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X/y/groups lengths differ.")
    return X, y, groups

def outer_splits(X, y, groups):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(cv.split(X, y, groups))

def grouped_inner_splits(X, y, groups, n_splits=4):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(X, y, groups))

def standardize_train(Xtr, Xva=None, Xte=None):
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    def f(x): return None if x is None else ((x - mean) / std).astype(np.float32)
    return f(Xtr), f(Xva), f(Xte), mean.astype(np.float32), std.astype(np.float32)

def window_features(X):
    # 100 deterministic features per 6-channel window.
    out = []
    eps = 1e-8
    for w in X:
        feats = []
        mag_acc = np.linalg.norm(w[:3], axis=0)
        mag_gyr = np.linalg.norm(w[3:], axis=0)
        channels = [*w, mag_acc, mag_gyr]
        for a in channels:
            d = np.diff(a)
            feats += [
                float(np.mean(a)), float(np.std(a)), float(np.min(a)),
                float(np.max(a)), float(np.sqrt(np.mean(a*a))),
                float(np.percentile(a,10)), float(np.percentile(a,25)),
                float(np.percentile(a,50)), float(np.percentile(a,75)),
                float(np.percentile(a,90)), float(np.mean(np.abs(d))),
            ]
        # 8 channels * 11 = 88
        feats += [
            float(np.mean(np.abs(w[0])) + np.mean(np.abs(w[1])) + np.mean(np.abs(w[2]))),
            float(np.std(mag_acc)),
            float(np.std(mag_gyr)),
            float(np.mean(np.abs(np.diff(mag_acc)))),
            float(np.mean(np.abs(np.diff(mag_gyr)))),
            float(np.std(np.diff(mag_acc))),
        ]
        # six pairwise correlations among raw channels
        for i in range(6):
            for j in range(i+1,6):
                a,b=w[i],w[j]
                sa,sb=np.std(a),np.std(b)
                feats.append(float(np.corrcoef(a,b)[0,1]) if sa>eps and sb>eps else 0.0)
        out.append(feats)
    F=np.asarray(out,dtype=np.float32)
    if F.shape[1] != 100:
        raise AssertionError(f"Feature contract expected 100, got {F.shape}")
    return F
