
"""SafeBand AI — BNO055 ML Benchmark V1.

Subject-independent benchmark with channel ablations:
ACC, ACC+GYRO, ACC+GYRO+MAG.

Models:
- Classical ML: Logistic Regression, k-NN, Linear SVM, Decision Tree,
  Random Forest, Extra Trees, RBF SVM, HistGradientBoosting
- CNN: small 1-D CNN
- TCN: residual dilated temporal CNN

Protocol:
- StratifiedGroupKFold by subject
- preprocessing fitted on outer training data only
- test fold never used for model selection
- pooled outer predictions reported
- event-overlap subset reported separately
"""

from pathlib import Path
import argparse
import json
import time
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_bno055_ml import (
    CHANNEL_SETS, CLASS_NAMES, load_windows, standardize_train_only,
    extract_features, metrics, save_json
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=ROOT/"datasets/synthetic/bno055/bno055_windows_v2_1.npz")
    p.add_argument("--model", choices=["logreg", "knn", "linear_svm", "decision_tree", "rf", "extra_trees", "rbf_svm", "hist_gradient_boosting", "cnn", "tcn"], default="rf")
    p.add_argument("--channels", choices=["acc", "accgyro", "accgyromag"], default="accgyromag")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT/"reports/bno055_ml_v1")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--rf-trees", type=int, default=400)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_splits(y, groups, folds, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError(f"Need at least {folds} unique subjects; found {len(unique_groups)}.")
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    if len(splits) != folds:
        raise RuntimeError("Unexpected number of CV splits.")
    return splits


def fit_predict_classical(model_name, Xtr, ytr, Xte, seed, trees):
    """Fit one deterministic-feature classical ML baseline.

    All preprocessing is fit inside the training fold only.  The same
    extracted feature matrix is used for every classical model so the
    comparison isolates the classifier family rather than changing the
    representation.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
    )

    scaled_models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                n_jobs=1,
            ),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(
                C=4.0,
                gamma="scale",
                class_weight="balanced",
                kernel="rbf",
                random_state=seed,
            ),
        ),
    }

    tree_models = {
        "decision_tree": DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "rf": RandomForestClassifier(
            n_estimators=trees,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=2,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=trees,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        ),
    }

    if model_name in scaled_models:
        clf = scaled_models[model_name]
    elif model_name in tree_models:
        clf = tree_models[model_name]
    else:
        raise ValueError(f"Unsupported classical model: {model_name}")

    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def build_torch_model(model_name, channels, classes):
    import torch
    import torch.nn as nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, 32, 5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 96, 5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(96, classes)
        def forward(self, x):
            return self.head(self.net(x).squeeze(-1))

    class TCNBlock(nn.Module):
        def __init__(self, cin, cout, dilation):
            super().__init__()
            pad = 2 * dilation
            self.conv1 = nn.Conv1d(cin, cout, 5, padding=pad, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(cout)
            self.conv2 = nn.Conv1d(cout, cout, 5, padding=pad, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(cout)
            self.act = nn.ReLU()
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        def forward(self, x):
            # Center-crop because dilation+padding increases temporal length.
            y = self.act(self.bn1(self.conv1(x)))
            y = self.act(self.bn2(self.conv2(y)))
            target = x.shape[-1]
            if y.shape[-1] != target:
                start = (y.shape[-1] - target) // 2
                y = y[..., start:start+target]
            return self.act(y + self.skip(x))

    class TinyTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = TCNBlock(channels, 32, 1)
            self.b2 = TCNBlock(32, 64, 2)
            self.b3 = TCNBlock(64, 96, 4)
            self.b4 = TCNBlock(96, 96, 8)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(96, classes)
        def forward(self, x):
            x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
            return self.head(self.pool(x).squeeze(-1))

    return TinyCNN() if model_name == "cnn" else TinyTCN()


def fit_predict_dl(Xtr, ytr, groups_tr, Xte, model_name, seed, epochs, batch_size, patience, classes):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(model_name, Xtr.shape[1], classes).to(device)

    # Internal validation is also grouped by subject. This keeps subject
    # independence throughout model development, not only at the outer test.
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, val_idx = next(gss.split(Xtr, ytr, groups_tr))
    if len(np.unique(groups_tr[train_idx])) < 2:
        raise ValueError("Internal training split has too few subjects.")
    if len(np.unique(ytr[val_idx])) < classes:
        raise ValueError(
            "Internal validation split does not contain all classes. "
            "Use a dataset where each subject contributes all benchmark classes."
        )

    tr_ds = TensorDataset(torch.from_numpy(Xtr[train_idx]), torch.from_numpy(ytr[train_idx]))
    va_x = torch.from_numpy(Xtr[val_idx]).to(device)
    va_y = torch.from_numpy(ytr[val_idx]).to(device)
    loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best_state = None
    best_score = -np.inf
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = model(va_x).argmax(1).cpu().numpy()
        score = metrics(ytr[val_idx], vp, CLASS_NAMES)["macro_f1"]
        if score > best_score + 1e-5:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError("No valid DL checkpoint was produced.")
    model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(Xte), batch_size):
            xb = torch.from_numpy(Xte[start:start+batch_size]).to(device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >= 2.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    set_seed(args.seed)
    X, y, groups, sessions, event, labels = load_windows(args.data)

    if len(labels) != len(CLASS_NAMES):
        raise ValueError(f"Expected 9 classes in the V2.1 artifact, got {labels}.")
    if set(labels) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected class vocabulary: {labels}")

    ch_idx = CHANNEL_SETS[args.channels]
    X = X[:, ch_idx, :]
    splits = make_splits(y, groups, args.folds, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    fold_results = []
    pooled_true, pooled_pred, pooled_event = [], [], []

    print(f"[INFO] Data: {args.data}")
    print(f"[INFO] X: {X.shape}; model={args.model}; channels={args.channels}")
    print(f"[INFO] Subjects={len(np.unique(groups))}; folds={args.folds}")

    for fold, (tr, te) in enumerate(splits, 1):
        t0 = time.time()
        Xtr_raw, Xte_raw = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        classical_models = {
            "logreg", "knn", "linear_svm", "decision_tree",
            "rf", "extra_trees", "rbf_svm", "hist_gradient_boosting",
        }

        if args.model in classical_models:
            Ftr = extract_features(Xtr_raw)
            Fte = extract_features(Xte_raw)
            pred = fit_predict_classical(
                args.model, Ftr, ytr, Fte, args.seed + fold, args.rf_trees
            )
        else:
            Xtr, Xte, _, _ = standardize_train_only(Xtr_raw, Xte_raw)
            pred = fit_predict_dl(
                Xtr, ytr, groups[tr], Xte, args.model, args.seed + fold,
                args.epochs, args.batch_size, args.patience, len(CLASS_NAMES)
            )

        m = metrics(yte, pred, CLASS_NAMES)
        event_mask = event[te]
        em = None
        if event_mask.any():
            em = metrics(yte[event_mask], pred[event_mask], CLASS_NAMES)
            em["n"] = int(event_mask.sum())

        row = {
            "fold": fold,
            "train_samples": int(len(tr)),
            "test_samples": int(len(te)),
            "test_subjects": sorted(np.unique(groups[te]).astype(int).tolist()),
            "metrics": m,
            "event_metrics": em,
            "elapsed_s": round(time.time() - t0, 3),
        }
        fold_results.append(row)
        pooled_true.append(yte)
        pooled_pred.append(pred)
        pooled_event.append(event_mask)

        print(
            f"[FOLD {fold}] acc={m['accuracy']:.4f} "
            f"bal={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
            f"n={len(te)} time={row['elapsed_s']:.1f}s"
        )

    yt = np.concatenate(pooled_true)
    yp = np.concatenate(pooled_pred)
    ev = np.concatenate(pooled_event)
    pooled = metrics(yt, yp, CLASS_NAMES)
    event_pooled = metrics(yt[ev], yp[ev], CLASS_NAMES) if ev.any() else None

    summary = {
        "experiment": "SafeBand_BNO055_ML_V1",
        "data": str(args.data),
        "model": args.model,
        "channels": args.channels,
        "channel_indices": ch_idx,
        "window_shape": [int(X.shape[1]), int(X.shape[2])],
        "subjects": int(len(np.unique(groups))),
        "samples": int(len(y)),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "protocol": "StratifiedGroupKFold grouped by subject; preprocessing fitted on outer training fold only.",
        "synthetic_data_note": "Synthetic BNO055-shaped data; not real-hardware validation.",
        "folds_detail": fold_results,
        "pooled_outer": pooled,
        "pooled_event_overlap": event_pooled,
    }
    out_path = args.out / f"{args.model}_{args.channels}.json"
    save_json(summary, out_path)

    print()
    print(f"[POOLED] accuracy={pooled['accuracy']:.4f}")
    print(f"[POOLED] balanced_accuracy={pooled['balanced_accuracy']:.4f}")
    print(f"[POOLED] macro_f1={pooled['macro_f1']:.4f}")
    if event_pooled:
        print(f"[EVENT] n={event_pooled and int(ev.sum())} macro_f1={event_pooled['macro_f1']:.4f}")
    print(f"[PASS] Report: {out_path}")


if __name__ == "__main__":
    main()
