from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import time
import random
import warnings
import numpy as np

from sklearn.exceptions import ConvergenceWarning
from pamap2_stationary_v5_common import (
    SEED, CLASSES, load_data, grouped_splits, standardize_train,
    features, metrics
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def run_svm(F, y, groups, outer):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    pred = np.full_like(y, -1)
    folds = []

    for k, (tr, te) in enumerate(outer, 1):
        inner = grouped_splits(F[tr], y[tr], groups[tr], 4)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced")),
        ])
        params = {
            "svc__C": [0.5, 1.0, 2.0, 4.0],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        }

        # inner is a concrete list, not a generator.
        search = GridSearchCV(
            model, params, scoring="f1_macro",
            cv=inner, n_jobs=1, refit=True, error_score="raise"
        )
        search.fit(F[tr], y[tr])
        pred[te] = search.predict(F[te])

        m = metrics(y[te], pred[te])
        folds.append({
            "fold": k,
            "test_subjects": sorted(np.unique(groups[te]).tolist()),
            "best_params": search.best_params_,
            "inner_macro_f1": float(search.best_score_),
            "test_metrics": m,
        })
        print(
            f"Fold {k}/5 | acc={m['accuracy']*100:.2f}% | "
            f"bal={m['balanced_accuracy']*100:.2f}% | F1={m['macro_f1']*100:.2f}%"
        )
    return folds, pred

def make_mlp(hidden, max_iter):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=max_iter,
            early_stopping=False,
            random_state=SEED,
        )),
    ])

def run_mlp(F, y, groups, outer):
    configs = [(64,), (128, 64), (128, 64, 32)]
    max_iters = [75, 125, 175]
    pred = np.full_like(y, -1)
    folds = []

    for k, (tr, te) in enumerate(outer, 1):
        inner = grouped_splits(F[tr], y[tr], groups[tr], 4)

        config_scores = []
        for cfg in configs:
            vals = []
            for a, b in inner:
                model = make_mlp(cfg, 125)
                model.fit(F[tr][a], y[tr][a])
                vals.append(metrics(
                    y[tr][b], model.predict(F[tr][b])
                )["macro_f1"])
            config_scores.append(float(np.mean(vals)))
        cfg = configs[int(np.argmax(config_scores))]

        iter_scores = []
        for it in max_iters:
            vals = []
            for a, b in inner:
                model = make_mlp(cfg, it)
                model.fit(F[tr][a], y[tr][a])
                vals.append(metrics(
                    y[tr][b], model.predict(F[tr][b])
                )["macro_f1"])
            iter_scores.append(float(np.mean(vals)))
        best_iter = max_iters[int(np.argmax(iter_scores))]

        final = make_mlp(cfg, best_iter)
        final.fit(F[tr], y[tr])
        pred[te] = final.predict(F[te])

        m = metrics(y[te], pred[te])
        folds.append({
            "fold": k,
            "test_subjects": sorted(np.unique(groups[te]).tolist()),
            "selected_hidden_layers": list(cfg),
            "inner_config_macro_f1": config_scores,
            "max_iter_candidates": max_iters,
            "inner_max_iter_macro_f1": iter_scores,
            "selected_max_iter": best_iter,
            "test_metrics": m,
        })
        print(
            f"Fold {k}/5 | acc={m['accuracy']*100:.2f}% | "
            f"bal={m['balanced_accuracy']*100:.2f}% | F1={m['macro_f1']*100:.2f}%"
        )
    return folds, pred

def torch_models():
    import torch.nn as nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(6, 32, 7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(64, 3)

        def forward(self, x):
            return self.fc(self.net(x).squeeze(-1))

    class BiRNN(nn.Module):
        def __init__(self, kind):
            super().__init__()
            RNN = nn.LSTM if kind == "LSTM" else nn.GRU
            self.rnn = RNN(
                input_size=6, hidden_size=48, num_layers=1,
                batch_first=True, bidirectional=True
            )
            self.fc = nn.Linear(96, 3)

        def forward(self, x):
            out = self.rnn(x)
            h = out[1][0] if isinstance(self.rnn, nn.LSTM) else out[1]
            h = h[-2:].transpose(0, 1).contiguous()
            return self.fc(h.view(x.size(0), 96))

    return TinyCNN, BiRNN

def train_nn(Xtr, ytr, Xva, yva, kind, epochs, device, seed):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    CNN, BiRNN = torch_models()
    model = CNN() if kind == "CNN" else BiRNN(kind)
    model.to(device)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xtr),
            torch.from_numpy(ytr.astype(np.int64))
        ),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_state = None
    best_score = -np.inf
    patience = 5
    stale = 0
    used = 0

    for epoch in range(1, epochs + 1):
        used = epoch
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if kind != "CNN":
                xb = xb.permute(0, 2, 1).contiguous()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xva).to(device)
            if kind != "CNN":
                xv = xv.permute(0, 2, 1).contiguous()
            pv = model(xv).argmax(1).cpu().numpy()
        score = metrics(yva, pv)["macro_f1"]

        if score > best_score + 1e-7:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, used, float(best_score)

def predict_nn(model, X, kind, device):
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).to(device)
        if kind != "CNN":
            x = x.permute(0, 2, 1).contiguous()
        return model(x).argmax(1).cpu().numpy()

def run_nn(X, y, groups, outer, kind, device):
    # Small, bounded candidate set: enough for family comparison,
    # not an open-ended architecture search.
    budgets = [10, 20, 30]
    pred = np.full_like(y, -1)
    folds = []

    for k, (tr, te) in enumerate(outer, 1):
        inner = grouped_splits(X[tr], y[tr], groups[tr], 4)
        scores = []

        for budget in budgets:
            vals = []
            for j, (a, b) in enumerate(inner, 1):
                Xa, Xb, _, = standardize_train(
                    X[tr][a], X[tr][b], None
                )
                _, _, score = train_nn(
                    Xa, y[tr][a], Xb, y[tr][b],
                    kind, budget, device, SEED + k * 100 + j
                )
                vals.append(score)
            scores.append(float(np.mean(vals)))

        best_budget = budgets[int(np.argmax(scores))]

        # Hold one GROUPED inner split out for early stopping.
        # It belongs only to the outer-training subjects.
        stop_tr, stop_va = inner[0]
        Xfit, Xstop, Xtest = standardize_train(
            X[tr][stop_tr], X[tr][stop_va], X[te]
        )
        model, used, stop_score = train_nn(
            Xfit, y[tr][stop_tr],
            Xstop, y[tr][stop_va],
            kind, best_budget, device, SEED + k
        )
        pred[te] = predict_nn(model, Xtest, kind, device)

        m = metrics(y[te], pred[te])
        folds.append({
            "fold": k,
            "test_subjects": sorted(np.unique(groups[te]).tolist()),
            "epoch_budgets": budgets,
            "inner_budget_macro_f1": scores,
            "selected_budget": best_budget,
            "early_stopping_epochs": used,
            "stopping_validation_macro_f1": stop_score,
            "test_metrics": m,
        })
        print(
            f"Fold {k}/5 | acc={m['accuracy']*100:.2f}% | "
            f"bal={m['balanced_accuracy']*100:.2f}% | F1={m['macro_f1']*100:.2f}%"
        )
    return folds, pred

def main():
    ap = argparse.ArgumentParser(
        description="SafeBand AI PAMAP2 Stationary Model Suite V5 patched."
    )
    ap.add_argument(
        "--data",
        default=str(ROOT / "datasets/processed/pamap2/stationary_accgyro_windows.npz")
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "models/pamap2_stationary_v5")
    )
    args = ap.parse_args()

    seed_all()
    X, y, groups = load_data(args.data)
    outer = grouped_splits(X, y, groups, 5)
    F = features(X)

    print(
        f"Dataset {X.shape}, subjects={sorted(np.unique(groups).tolist())}, "
        f"features={F.shape}"
    )

    results, timing = {}, {}

    for name, fn in [
        ("RBF_SVM", lambda: run_svm(F, y, groups, outer)),
        ("ANN_MLP", lambda: run_mlp(F, y, groups, outer)),
    ]:
        print(f"\n===== {name} =====")
        start = time.perf_counter()
        fr, p = fn()
        timing[name] = time.perf_counter() - start
        results[name] = {"folds": fr, "pooled": metrics(y, p)}

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Tiny_CNN, BiLSTM and BiGRU. "
            "Install PyTorch in the active venv and rerun."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nPyTorch device: {device}")

    for name, kind in [
        ("Tiny_CNN", "CNN"),
        ("BiLSTM", "LSTM"),
        ("BiGRU", "GRU"),
    ]:
        print(f"\n===== {name} =====")
        start = time.perf_counter()
        fr, p = run_nn(X, y, groups, outer, kind, device)
        timing[name] = time.perf_counter() - start
        results[name] = {
            "folds": fr,
            "pooled": metrics(y, p),
            "device": device,
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "pamap2_stationary_model_suite_v5_patched3",
        "seed": SEED,
        "dataset": str(Path(args.data)),
        "samples": int(len(y)),
        "subjects": sorted(np.unique(groups).tolist()),
        "classes": CLASSES,
        "input_shape": list(X.shape),
        "feature_count": int(F.shape[1]),
        "validation": (
            "Outer 5-fold StratifiedGroupKFold by PeopleId. "
            "All model selection uses grouped inner CV. "
            "Outer test subjects are untouched. "
            "Normalization is fitted only on each training partition."
        ),
        "models": results,
        "timing_seconds": timing,
    }
    report_path = out / "model_suite_v5_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== POOLED COMPARISON =====")
    for name, r in results.items():
        m = r["pooled"]
        print(
            f"{name:12s} "
            f"accuracy={m['accuracy']*100:.2f}% "
            f"balanced={m['balanced_accuracy']*100:.2f}% "
            f"macroF1={m['macro_f1']*100:.2f}%"
        )
    print(f"\nReport: {report_path}")

if __name__ == "__main__":
    main()
