
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset


SEED = 42
CLASSES = [
    "Downstairs", "LiftDown", "LiftUp", "LyingDown", "RampDown",
    "RampUp", "Running", "Sitting", "Standing", "Upstairs", "Walking"
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_data(npz_path, manifest_path):
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.keys())
    X = None
    for k in keys:
        a = z[k]
        if isinstance(a, np.ndarray) and a.ndim == 3 and np.issubdtype(a.dtype, np.number):
            X = np.asarray(a, dtype=np.float32)
            break
    if X is None:
        raise RuntimeError(f"No numeric 3-D array in NPZ. Keys: {keys}")
    if X.shape[1] not in (3, 4) and X.shape[2] in (3, 4):
        X = np.transpose(X, (0, 2, 1))
    if X.shape[1] != 3:
        # If magnitude is already present, keep it; otherwise create it below.
        if X.shape[1] != 4:
            raise RuntimeError(f"Expected 3 or 4 channels, got {X.shape}")

    df = pd.read_csv(manifest_path)
    if len(df) != len(X):
        raise RuntimeError(f"Manifest rows ({len(df)}) != windows ({len(X)})")

    label_col = pick_column(df, ["activity", "activity_label", "label", "class", "Activity"])
    acq_col = pick_column(df, ["acquisition_id", "AcquisitionID", "acquisition", "trial_id"])
    part_col = pick_column(df, ["participant_id", "ParticipantID", "participant", "subject_id", "subject"])
    fold_col = pick_column(df, ["fold", "Fold", "partition", "split"])
    if not all([label_col, acq_col, part_col, fold_col]):
        raise RuntimeError(
            f"Could not resolve required manifest columns. "
            f"label={label_col}, acquisition={acq_col}, participant={part_col}, fold={fold_col}. "
            f"Columns={list(df.columns)}"
        )

    # The official test partition remains completely untouched.
    is_test = df[fold_col].astype(str).str.lower().eq("test").to_numpy()
    y_raw = df[label_col].astype(str).to_numpy()
    labels = sorted(pd.unique(y_raw).tolist())
    # Prefer the documented class order when all classes are present.
    if set(CLASSES).issubset(set(labels)):
        labels = CLASSES
    label_to_id = {v: i for i, v in enumerate(labels)}
    y = np.array([label_to_id[v] for v in y_raw], dtype=np.int64)

    # ACC magnitude is a deterministic fourth channel. It adds orientation-robust
    # information without importing another sensor modality.
    if X.shape[1] == 3:
        mag = np.sqrt(np.sum(X[:, :3, :] ** 2, axis=1, keepdims=True))
        X = np.concatenate([X, mag], axis=1)

    # Remove only non-finite windows, if any; never touch labels/test otherwise.
    finite = np.all(np.isfinite(X), axis=(1, 2))
    if not finite.all():
        keep = finite
        X, y, df, is_test = X[keep], y[keep], df.loc[keep].reset_index(drop=True), is_test[keep]

    return X, y, df, is_test, labels, acq_col, part_col


class WindowDataset(Dataset):
    def __init__(self, X, y, mean, std, train=False):
        self.X = X
        self.y = y
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std, 1e-6).astype(np.float32)
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.train:
            # Mild, physically plausible augmentation.
            if np.random.rand() < 0.7:
                x[:3] += np.random.normal(0, 0.015, size=x[:3].shape).astype(np.float32)
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.96, 1.04, size=(3, 1)).astype(np.float32)
                x[:3] *= scale
            if np.random.rand() < 0.25:
                # Short temporal masking forces robust representations.
                w = np.random.randint(3, 10)
                s = np.random.randint(0, max(1, x.shape[1] - w + 1))
                x[:3, s:s+w] = 0.0
            x[3] = np.sqrt(np.sum(x[:3] ** 2, axis=0))

        x = (x - self.mean[:, None]) / self.std[:, None]
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


class ResidualBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        p = dilation
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=p, dilation=dilation),
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.Conv1d(c, c, 3, padding=p, dilation=dilation),
            nn.BatchNorm1d(c),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SupervisedDenoisingAE(nn.Module):
    def __init__(self, n_classes, latent_dim=128):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv1d(4, 48, 5, padding=2),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.enc = nn.Sequential(
            ResidualBlock(48, 1),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 96, 5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            ResidualBlock(96, 2),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            ResidualBlock(128, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, n_classes),
        )
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 25),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 96, 4, stride=2, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.ConvTranspose1d(96, 48, 4, stride=2, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 4, 5, padding=2),
        )

    def encode(self, x):
        x = self.enc0(x)
        x = self.enc(x)
        return self.to_latent(x)

    def forward(self, x):
        z = self.encode(x)
        logits = self.classifier(z)
        d = self.dec_fc(z).view(x.size(0), 128, 25)
        recon = self.decoder(d)
        return logits, recon, z


def compute_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    # Inverse sqrt is deliberately gentler than inverse frequency.
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w /= w.mean()
    return torch.tensor(w, dtype=torch.float32)


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits, _, _ = model(x)
        ps.append(torch.argmax(logits, dim=1).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--recon-weight", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, df, is_test, labels, acq_col, part_col = load_data(args.npz, args.manifest)
    n_classes = len(labels)

    train_idx = np.flatnonzero(~is_test)
    test_idx = np.flatnonzero(is_test)
    if len(test_idx) == 0:
        raise RuntimeError("No official test partition found.")

    # One grouped, stratified validation split from the non-test data.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    split = None
    for tr, va in sgkf.split(np.zeros(len(train_idx)), y[train_idx], groups=df.iloc[train_idx][part_col].astype(str)):
        split = (train_idx[tr], train_idx[va])
        break
    tr_idx, va_idx = split

    mean = X[tr_idx].mean(axis=(0, 2))
    std = X[tr_idx].std(axis=(0, 2))
    train_ds = WindowDataset(X[tr_idx], y[tr_idx], mean, std, train=True)
    val_ds = WindowDataset(X[va_idx], y[va_idx], mean, std, train=False)
    test_ds = WindowDataset(X[test_idx], y[test_idx], mean, std, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedDenoisingAE(n_classes, args.latent_dim).to(device)

    weights = compute_weights(y[tr_idx], n_classes).to(device)
    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.04)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=2e-5)

    best_f1 = -1
    best_state = None
    bad = 0
    history = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            logits, recon, _ = model(x)
            loss_cls = ce(logits, target)
            # Reconstruction only stabilizes the encoder; classification remains primary.
            loss_rec = torch.mean((recon - x) ** 2)
            loss = loss_cls + args.recon_weight * loss_rec
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            running += float(loss.item()) * len(target)

        yv, pv = evaluate(model, val_loader, device)
        vm = metrics(yv, pv)
        sched.step(vm["macro_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": running / len(train_ds),
            **{f"val_{k}": v for k, v in vm.items()},
            "lr": float(opt.param_groups[0]["lr"]),
        })
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"loss={history[-1]['train_loss']:.5f} "
            f"val_acc={vm['accuracy']:.4f} "
            f"val_bal={vm['balanced_accuracy']:.4f} "
            f"val_macro_f1={vm['macro_f1']:.4f}"
        )

        if vm["macro_f1"] > best_f1 + 1e-4:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    yt, pt = evaluate(model, test_loader, device)
    tm = metrics(yt, pt)
    cm = confusion_matrix(yt, pt, labels=np.arange(n_classes)).tolist()

    artifact = {
        "state_dict": model.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "labels": labels,
        "input_channels": 4,
        "latent_dim": args.latent_dim,
    }
    torch.save(artifact, out / "aicos_har_supervised_dae_v1.pt")

    report = {
        "dataset": "AICOS-HAR",
        "model": "supervised_denoising_autoencoder",
        "input": "ACC xyz + acceleration magnitude",
        "window_shape": list(X.shape[1:]),
        "labels": labels,
        "train_windows": int(len(tr_idx)),
        "validation_windows": int(len(va_idx)),
        "official_test_windows": int(len(test_idx)),
        "train_participants": int(df.iloc[tr_idx][part_col].nunique()),
        "validation_participants": int(df.iloc[va_idx][part_col].nunique()),
        "test_participants": int(df.iloc[test_idx][part_col].nunique()),
        "best_validation_macro_f1": float(best_f1),
        "official_test": {**tm, "confusion_matrix": cm},
        "history": history,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "reconstruction_weight": args.recon_weight,
        "seed": args.seed,
        "elapsed_seconds": time.time() - start,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["official_test"], indent=2))
    print(f"Model: {out / 'aicos_har_supervised_dae_v1.pt'}")
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
