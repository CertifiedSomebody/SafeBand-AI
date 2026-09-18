from pathlib import Path
import sys
import argparse
import json
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight


LABELS = ["breath", "cough", "crying", "laugh", "screaming", "sneeze", "yawn"]
N_CLASSES = len(LABELS)


class SmallLogMelCNN(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_proportion_error(indices, y, reference_indices):
    candidate_counts = np.bincount(
        y[indices], minlength=N_CLASSES
    ).astype(np.float64)
    reference_counts = np.bincount(
        y[reference_indices], minlength=N_CLASSES
    ).astype(np.float64)

    candidate_props = candidate_counts / max(candidate_counts.sum(), 1.0)
    reference_props = reference_counts / max(reference_counts.sum(), 1.0)

    return float(np.mean(np.abs(candidate_props - reference_props)))


def make_inner_validation(outer_train, y, groups, seed):
    """
    Grouped validation split with class-balance selection.

    GroupShuffleSplit itself is not stratified. Instead of trusting one
    arbitrary grouped split, generate deterministic candidates and choose
    the candidate whose validation class proportions are closest to the
    outer-training distribution.

    A hard requirement is that every class occurs in both train and val.
    """
    candidates = []

    for attempt in range(50):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=seed * 1000 + attempt,
        )

        relative_train, relative_val = next(
            splitter.split(
                np.zeros(len(outer_train)),
                y[outer_train],
                groups[outer_train],
            )
        )

        train_idx = outer_train[relative_train]
        val_idx = outer_train[relative_val]

        train_counts = np.bincount(y[train_idx], minlength=N_CLASSES)
        val_counts = np.bincount(y[val_idx], minlength=N_CLASSES)

        if np.all(train_counts > 0) and np.all(val_counts > 0):
            score = class_proportion_error(
                val_idx, y, outer_train
            )
            candidates.append((score, train_idx, val_idx))

    if not candidates:
        raise RuntimeError(
            "Could not construct a grouped inner split containing all 7 "
            "classes in both train and validation after 50 attempts."
        )

    candidates.sort(key=lambda item: item[0])
    score, train_idx, val_idx = candidates[0]

    return train_idx, val_idx, score


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(N_CLASSES)),
        ).tolist(),
    }


def predict(model, X, device, batch_size):
    model.eval()
    predictions = []

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            predictions.append(
                torch.argmax(logits, dim=1).cpu().numpy()
            )

    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets/processed/nonspeech7k/logmel_v3_1.npz",
    )
    parser.add_argument(
        "--out",
        default="reports/nonspeech7k_audio_v3_1",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.is_file():
        raise FileNotFoundError(f"Cache not found: {input_path}")

    data = np.load(input_path, allow_pickle=False)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["groups"].astype(str)

    if X.shape != (6283, 1, 64, 128):
        raise RuntimeError(f"Unexpected X shape: {X.shape}")

    if len(y) != 6283 or len(groups) != 6283:
        raise RuntimeError("Cache length mismatch.")

    if len(set(groups.tolist())) != 1899:
        raise RuntimeError(
            f"Expected 1899 groups, got {len(set(groups.tolist()))}."
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Input cache contains non-finite values.")

    if sorted(np.unique(y).tolist()) != list(range(N_CLASSES)):
        raise RuntimeError("Cache labels are not exactly 0..6.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] device: {device}")
    print(f"[INFO] samples: {len(y)}")
    print(f"[INFO] groups : {len(set(groups.tolist()))}")

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed,
    )

    all_predictions = np.full(
        len(y), fill_value=-1, dtype=np.int64
    )

    report = {
        "protocol": "5-fold StratifiedGroupKFold grouped by file_id",
        "inner_validation": (
            "50 deterministic GroupShuffleSplit candidates; "
            "retain candidates with all classes in train/val and choose "
            "closest validation class proportions"
        ),
        "samples": int(len(y)),
        "groups": int(len(set(groups.tolist()))),
        "input_shape": [1, 64, 128],
        "model": "SmallLogMelCNN",
        "seed": args.seed,
        "device": str(device),
        "epochs_max": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "folds": [],
    }

    for fold, (outer_train, outer_test) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        inner_train, inner_val, split_error = make_inner_validation(
            outer_train,
            y,
            groups,
            seed=args.seed + fold,
        )

        train_counts = np.bincount(
            y[inner_train], minlength=N_CLASSES
        )
        val_counts = np.bincount(
            y[inner_val], minlength=N_CLASSES
        )

        print(
            f"\n[FOLD {fold}] "
            f"train/val/test = "
            f"{len(inner_train)}/{len(inner_val)}/{len(outer_test)}"
        )
        print(
            f"[FOLD {fold}] validation class-proportion error: "
            f"{split_error:.6f}"
        )
        print(
            f"[FOLD {fold}] train counts: "
            f"{train_counts.tolist()}"
        )
        print(
            f"[FOLD {fold}] val counts  : "
            f"{val_counts.tolist()}"
        )

        # Fit normalization using inner training data ONLY.
        mean = float(X[inner_train].mean(dtype=np.float64))
        std = float(X[inner_train].std(dtype=np.float64))

        if not np.isfinite(mean) or not np.isfinite(std):
            raise RuntimeError(f"Invalid normalization in fold {fold}.")

        std = max(std, 1e-6)

        X_train = ((X[inner_train] - mean) / std).astype(np.float32)
        X_val = ((X[inner_val] - mean) / std).astype(np.float32)
        X_test = ((X[outer_test] - mean) / std).astype(np.float32)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(N_CLASSES),
            y=y[inner_train],
        ).astype(np.float32)

        model = SmallLogMelCNN().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device,
            )
        )

        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y[inner_train]),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        best_val_f1 = -np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        best_state = None
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []

            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = model(xb)
                loss = criterion(logits, yb)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at fold {fold}, epoch {epoch}."
                    )

                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))

            val_pred = predict(
                model,
                X_val,
                device,
                args.batch_size,
            )

            val_f1 = float(
                f1_score(
                    y[inner_val],
                    val_pred,
                    labels=list(range(N_CLASSES)),
                    average="macro",
                    zero_division=0,
                )
            )

            mean_loss = float(np.mean(losses))

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "val_macro_f1": val_f1,
                }
            )

            print(
                f"[FOLD {fold}] "
                f"epoch={epoch:02d} "
                f"loss={mean_loss:.4f} "
                f"val_macroF1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1 + 1e-5:
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_without_improvement = 0

                # CPU clone avoids retaining the live model tensors.
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            elif epoch >= args.min_epochs:
                epochs_without_improvement += 1

                if epochs_without_improvement >= args.patience:
                    print(
                        f"[FOLD {fold}] early stopping at epoch {epoch}; "
                        f"best epoch={best_epoch}"
                    )
                    break

        if best_state is None:
            raise RuntimeError(
                f"No best checkpoint was created for fold {fold}."
            )

        model.load_state_dict(best_state)
        model.eval()

        test_pred = predict(
            model,
            X_test,
            device,
            args.batch_size,
        )

        if np.any(all_predictions[outer_test] != -1):
            raise RuntimeError(
                f"Outer test indices overlap in fold {fold}."
            )

        all_predictions[outer_test] = test_pred

        fold_metrics = classification_metrics(
            y[outer_test],
            test_pred,
        )

        fold_metrics.update(
            {
                "fold": fold,
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(best_val_f1),
                "test_samples": int(len(outer_test)),
                "inner_train_samples": int(len(inner_train)),
                "inner_val_samples": int(len(inner_val)),
                "normalization_mean": mean,
                "normalization_std": std,
                "train_class_counts": train_counts.tolist(),
                "val_class_counts": val_counts.tolist(),
                "history": history,
            }
        )

        report["folds"].append(fold_metrics)

        print(
            f"[FOLD {fold} RESULT] "
            f"accuracy={fold_metrics['accuracy']:.4f} "
            f"balanced={fold_metrics['balanced_accuracy']:.4f} "
            f"macroF1={fold_metrics['macro_f1']:.4f}"
        )

    if np.any(all_predictions < 0):
        raise RuntimeError("Some outer samples were never evaluated.")

    pooled = classification_metrics(y, all_predictions)

    fold_accuracies = [
        fold["accuracy"] for fold in report["folds"]
    ]
    fold_balanced = [
        fold["balanced_accuracy"] for fold in report["folds"]
    ]
    fold_f1 = [
        fold["macro_f1"] for fold in report["folds"]
    ]

    report["pooled"] = pooled
    report["mean_accuracy"] = float(np.mean(fold_accuracies))
    report["std_accuracy"] = float(np.std(fold_accuracies))
    report["mean_balanced_accuracy"] = float(np.mean(fold_balanced))
    report["std_balanced_accuracy"] = float(np.std(fold_balanced))
    report["mean_macro_f1"] = float(np.mean(fold_f1))
    report["std_macro_f1"] = float(np.std(fold_f1))

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "cnn_v3_1_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n[POOLED]")
    print(json.dumps(pooled, indent=2))
    print(
        f"[SUMMARY] mean accuracy={report['mean_accuracy']:.4f} "
        f"+/- {report['std_accuracy']:.4f}"
    )
    print(
        f"[SUMMARY] mean balanced accuracy="
        f"{report['mean_balanced_accuracy']:.4f} "
        f"+/- {report['std_balanced_accuracy']:.4f}"
    )
    print(
        f"[SUMMARY] mean macroF1={report['mean_macro_f1']:.4f} "
        f"+/- {report['std_macro_f1']:.4f}"
    )
    print(f"[PASS] report -> {report_path}")


if __name__ == "__main__":
    main()
