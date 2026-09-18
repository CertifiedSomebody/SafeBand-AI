from pathlib import Path
import sys, argparse, json, time, warnings
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from tools.common_nonspeech7k_bestfit import LABELS

def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }

def build_models(seed):
    # These are intentionally small, strong classical candidates for the
    # already-established V2 representation. HGB is included because it was
    # the strongest completed V2 model; the others test complementary inductive biases.
    return {
        "hgb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "rbf_svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(C=4.0, gamma="scale", kernel="rbf", class_weight="balanced")),
        ]),
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=3000,
                                       class_weight="balanced", random_state=seed)),
        ]),
    }

def main():
    p = argparse.ArgumentParser(
        description="Best-fit selection for SafeBand Nonspeech7k audio V2 features."
    )
    p.add_argument("--features", default="datasets/processed/nonspeech7k/train_tf_features_v2.csv")
    p.add_argument("--out", default="models/nonspeech7k_audio_bestfit_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", type=int, default=5)
    args = p.parse_args()

    feature_path = Path(args.features)
    if not feature_path.is_absolute():
        feature_path = ROOT / feature_path
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    df = pd.read_csv(feature_path, dtype={"file_id": "string"})
    required = {"label", "file_id"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Missing required columns: {required - set(df.columns)}")

    df["file_id"] = df["file_id"].astype(str).str.strip()
    y_raw = df["label"].astype(str).str.strip().str.lower()
    label_to_id = {v:i for i,v in enumerate(LABELS)}
    unknown = sorted(set(y_raw) - set(label_to_id))
    if unknown:
        raise RuntimeError(f"Unknown labels: {unknown}")

    y = y_raw.map(label_to_id).to_numpy(dtype=np.int64)
    groups = df["file_id"].to_numpy(dtype=str)

    meta = {"label", "file_id", "audio_path", "source", "split"}
    feature_cols = [c for c in df.columns if c not in meta]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if len(df) != 6283:
        raise RuntimeError(f"Expected 6283 clean training rows, found {len(df)}")
    if len(np.unique(groups)) != 1899:
        raise RuntimeError(f"Expected 1899 File-ID groups, found {len(np.unique(groups))}")
    if not np.isfinite(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)).all():
        raise RuntimeError("Invalid feature matrix")

    cv = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    fold_records = []
    pooled = {name: {"y": [], "pred": []} for name in build_models(args.seed)}

    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        print(f"\n===== Fold {fold}/{args.splits} =====")
        models = build_models(args.seed + fold)
        for name, model in models.items():
            t0 = time.time()
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            m = metrics(y[te], pred)
            rec = {"fold": fold, "model": name, **m, "seconds": time.time()-t0,
                   "test_samples": int(len(te))}
            fold_records.append(rec)
            pooled[name]["y"].extend(y[te].tolist())
            pooled[name]["pred"].extend(pred.tolist())
            print(f"{name:10s} acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    summary = []
    for name, vals in pooled.items():
        m = metrics(np.array(vals["y"]), np.array(vals["pred"]))
        rows = [r for r in fold_records if r["model"] == name]
        summary.append({
            "model": name,
            "mean_accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "std_accuracy": float(np.std([r["accuracy"] for r in rows])),
            "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in rows])),
            "std_balanced_accuracy": float(np.std([r["balanced_accuracy"] for r in rows])),
            "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "std_macro_f1": float(np.std([r["macro_f1"] for r in rows])),
            "pooled_accuracy": m["accuracy"],
            "pooled_balanced_accuracy": m["balanced_accuracy"],
            "pooled_macro_f1": m["macro_f1"],
        })

    # Selection criterion is explicit and non-arbitrary:
    # Macro-F1 first, then balanced accuracy, then accuracy.
    # This is appropriate for the imbalanced 7-class reference task.
    summary_sorted = sorted(
        summary,
        key=lambda r: (r["mean_macro_f1"], r["mean_balanced_accuracy"], r["mean_accuracy"]),
        reverse=True
    )
    selected = summary_sorted[0]["model"]
    print("\n===== BEST-FIT SELECTION =====")
    for r in summary_sorted:
        print(f"{r['model']:10s} mean_macroF1={r['mean_macro_f1']:.4f} "
              f"mean_bal={r['mean_balanced_accuracy']:.4f} mean_acc={r['mean_accuracy']:.4f}")
    print(f"Selected by frozen criterion: {selected}")

    # Fit selected model on all approved training rows.
    final_model = build_models(args.seed)[selected]
    final_model.fit(X, y)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    import joblib
    model_path = out / "nonspeech7k_audio_bestfit_model.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "model_file": str(model_path.relative_to(ROOT)),
        "representation": "Nonspeech7k V2 time-frequency handcrafted features",
        "training_samples": int(len(df)),
        "file_id_groups": int(len(np.unique(groups))),
        "labels": LABELS,
        "cv": "5-fold StratifiedGroupKFold grouped by file_id",
        "selection_rule": "highest mean Macro-F1; tie-break balanced accuracy; tie-break accuracy",
        "selected_model": selected,
        "candidate_summary": summary_sorted,
        "official_test_used": False,
        "note": "Best-fit selection is an internal cross-validation model-selection result. The official 725-recording test set remains untouched.",
    }
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[PASS] saved -> {model_path}")
    print(f"[PASS] metadata -> {out/'metadata.json'}")

if __name__ == "__main__":
    main()
