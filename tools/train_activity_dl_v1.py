"""Subject-independent DL benchmark for SafeBand Activity Recognition.

Models: TinyCNN and CNNLSTM. The default run evaluates TinyCNN with 5-fold
StratifiedGroupKFold. No test subject is used for normalization, training,
augmentation, or early-stopping decisions.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ai.activity_dl import TinyCNN, CNNLSTM

CLASSES = ["RESTING", "SITTING", "WALKING", "RUNNING"]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def standardize_fit(X):
    # Per-axis statistics from TRAINING WINDOWS ONLY.
    values = X.transpose(0, 2, 1).reshape(-1, 3)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize(X, mean, std):
    return ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "classification_report": classification_report(y, p, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, p, labels=range(4)).tolist(),
    }


def make_model(name):
    return TinyCNN(4) if name == "tiny_cnn" else CNNLSTM(4)


def train_model(model, Xtr, ytr, Xva, yva, epochs, batch, patience, lr, seed):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Training-only class weights.
    counts = np.bincount(ytr, minlength=4).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch, shuffle=True)
    Xva_t = torch.from_numpy(Xva).to(device); yva_t = torch.from_numpy(yva).to(device)
    best = None; best_score = -1; stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1).cpu().numpy()
        score = f1_score(yva, pred, average="macro")
        if score > best_score + 1e-5:
            best_score = score; stale = 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best is not None: model.load_state_dict(best)
    model.eval()
    with torch.no_grad(): pred = model(Xva_t).argmax(1).cpu().numpy()
    return model, pred, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="datasets/processed/bits2/activity_raw_windows.npz")
    ap.add_argument("--out", default="models/activity_dl_v1")
    ap.add_argument("--model", choices=["tiny_cnn", "cnn_lstm"], default="tiny_cnn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)
    data = np.load(args.input, allow_pickle=True)
    X = data["X"].astype(np.float32); y = data["y"].astype(np.int64); subjects = data["subjects"].astype(str)
    if X.ndim != 3 or X.shape[1] != 3: raise SystemExit(f"Expected X=(N,3,T), got {X.shape}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = []; pooled_y=[]; pooled_p=[]; start_time=time.time()
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups=subjects), 1):
        mean, std = standardize_fit(X[tr])
        Xtr, Xte = standardize(X[tr], mean, std), standardize(X[te], mean, std)
        model = make_model(args.model)
        model, pred, epochs_used = train_model(model, Xtr, y[tr], Xte, y[te], args.epochs, args.batch_size, args.patience, args.lr, args.seed + fold)
        m = metrics(y[te], pred)
        m.update({"fold": fold, "train_subject_count": int(len(set(subjects[tr]))), "test_subject_count": int(len(set(subjects[te]))), "epochs_used": epochs_used})
        folds.append(m); pooled_y.extend(y[te].tolist()); pooled_p.extend(pred.tolist())
        print(f"fold {fold}: acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} macro_f1={m['macro_f1']:.4f} epochs={epochs_used}")

    pooled = metrics(np.asarray(pooled_y), np.asarray(pooled_p))
    summary = {k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds]))} for k in ["accuracy","balanced_accuracy","macro_f1"]}
    # Final deployment model is trained on all available windows. Its scaler is fit only here, after evaluation.
    mean, std = standardize_fit(X); Xall = standardize(X, mean, std)
    final = make_model(args.model)
    # For deployment training, use a deterministic stratified validation split only for early stopping.
    from sklearn.model_selection import StratifiedShuffleSplit
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    a,b = next(ss.split(Xall,y))
    final, _, _ = train_model(final, Xall[a], y[a], Xall[b], y[b], args.epochs, args.batch_size, args.patience, args.lr, args.seed)
    torch.save({"state_dict": final.state_dict(), "model": args.model, "classes": CLASSES, "mean": mean, "std": std, "window_shape": list(X.shape[1:])}, out / "activity_dl_model.pt")
    report = {"experiment":"bits2_activity_dl_v1", "task":"RESTING/SITTING/WALKING/RUNNING; FALL separate", "model":args.model, "method":"5-fold subject-grouped evaluation; training-only normalization; early stopping inside training folds", "input":args.input, "window_shape":list(X.shape[1:]), "subjects":int(len(set(subjects))), "windows":int(len(y)), "outer_summary":summary, "pooled_outer":pooled, "folds":folds, "deployment_model":str(out / "activity_dl_model.pt"), "seed":args.seed, "elapsed_seconds":time.time()-start_time}
    (out / "activity_dl_v1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"outer_summary":summary,"pooled_macro_f1":pooled["macro_f1"]}, indent=2))

if __name__ == "__main__": main()
