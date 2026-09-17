from pathlib import Path
import sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_NAME = "extra_trees"
SEED = 42


def make_model():
    return ExtraTreesClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )


def metrics(y, p, labels):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(y, p, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
    }


def choose_confusing_pair(X, y, groups, classes, seed=SEED):
    """Choose a pair using INNER grouped CV only. Outer validation remains untouched."""
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    cm_total = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_i = {c: i for i, c in enumerate(classes)}
    for tr, va in splitter.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[va])
        cm_total += confusion_matrix(y.iloc[va], p, labels=classes)

    # Symmetric normalized confusion. This prevents the largest class from
    # automatically winning simply because it has more samples.
    row_sums = cm_total.sum(axis=1, keepdims=True)
    norm = cm_total / np.maximum(row_sums, 1)
    score = norm + norm.T
    np.fill_diagonal(score, -1)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    pair = [classes[i], classes[j]]
    return pair, cm_total.tolist(), norm.tolist()


def run_outer_cv(df, features, label_col, group_col, n_splits=5):
    X = df[features].astype(float)
    y = df[label_col].astype(str)
    g = df[group_col].astype(str)
    classes = sorted(y.unique().tolist())
    if len(classes) < 2:
        raise ValueError("Need at least two activity classes")

    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    baseline_rows, hierarchical_rows, fold_details = [], [], []

    for fold, (tr, va) in enumerate(outer.split(X, y, g), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        gtr = g.iloc[tr]

        baseline = make_model()
        baseline.fit(Xtr, ytr)
        p_base = baseline.predict(Xva)

        pair, inner_cm, inner_norm = choose_confusing_pair(Xtr, ytr, gtr, classes, seed=SEED + fold)

        specialist_mask = ytr.isin(pair).to_numpy()
        specialist = clone(make_model())
        specialist.fit(Xtr.loc[specialist_mask], ytr.loc[specialist_mask])

        p_hier = p_base.copy()
        route = np.isin(p_base, pair)
        if route.any():
            p_hier[route] = specialist.predict(Xva.loc[route])

        bm = metrics(yva, p_base, classes)
        hm = metrics(yva, p_hier, classes)
        baseline_rows.append({k: bm[k] for k in ["accuracy", "balanced_accuracy", "macro_f1"]})
        hierarchical_rows.append({k: hm[k] for k in ["accuracy", "balanced_accuracy", "macro_f1"]})
        fold_details.append({
            "fold": fold,
            "specialist_pair": pair,
            "routed_samples": int(route.sum()),
            "inner_confusion_matrix": inner_cm,
            "inner_normalized_confusion": inner_norm,
            "baseline": bm,
            "hierarchical": hm,
        })
        print(f"fold {fold}: specialist={pair} routed={int(route.sum())} "
              f"baseline_macroF1={bm['macro_f1']:.4f} hierarchical_macroF1={hm['macro_f1']:.4f}")

    def summary(rows):
        return {m: {"mean": float(np.mean([r[m] for r in rows])),
                     "std": float(np.std([r[m] for r in rows], ddof=1))}
                for m in ["accuracy", "balanced_accuracy", "macro_f1"]}

    return {
        "dataset": "BITS2",
        "experiment": "confusion_guided_hierarchical_activity_v3",
        "model": MODEL_NAME,
        "selection": "inner 3-fold StratifiedGroupKFold on training portion of each outer fold",
        "routing": "baseline prediction in selected pair -> pair specialist; otherwise baseline prediction",
        "classes": classes,
        "baseline_summary": summary(baseline_rows),
        "hierarchical_summary": summary(hierarchical_rows),
        "delta_macro_f1": float(np.mean([r["macro_f1"] for r in hierarchical_rows]) - np.mean([r["macro_f1"] for r in baseline_rows])),
        "delta_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in hierarchical_rows]) - np.mean([r["balanced_accuracy"] for r in baseline_rows])),
        "delta_accuracy": float(np.mean([r["accuracy"] for r in hierarchical_rows]) - np.mean([r["accuracy"] for r in baseline_rows])),
        "folds": fold_details,
    }


def main():
    ap = argparse.ArgumentParser(description="BITS2 confusion-guided hierarchical activity experiment V3")
    ap.add_argument("--input", default="datasets/processed/bits2/activity_windows.csv")
    ap.add_argument("--out-dir", default="models/activity_hierarchical_v3")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    path = ROOT / args.input
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    label_col = "activity_label"
    group_col = "subject_id"
    required = {label_col, group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    meta = {label_col, group_col, "session_id", "source_file", "recording_type",
            "window_start_sample", "window_end_sample", "window_samples"}
    features = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        raise ValueError("No numeric feature columns found")

    # Drop rows with missing feature values; record this explicitly.
    before = len(df)
    df = df.dropna(subset=features + [label_col, group_col]).reset_index(drop=True)
    dropped = before - len(df)
    subjects = df[group_col].nunique()
    if subjects < args.splits:
        raise ValueError(f"Need >= {args.splits} subjects, found {subjects}")

    print(f"Dataset rows={len(df)} subjects={subjects} classes={sorted(df[label_col].unique())}")
    print(f"Features={len(features)} dropped_rows={dropped}")
    report = run_outer_cv(df, features, label_col, group_col, args.splits)
    report["input"] = str(path)
    report["feature_count"] = len(features)
    report["features"] = features
    report["dropped_rows"] = dropped

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "hierarchical_v3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nBASELINE:", report["baseline_summary"])
    print("HIERARCHICAL:", report["hierarchical_summary"])
    print("DELTA:", {k: report[k] for k in ["delta_accuracy", "delta_balanced_accuracy", "delta_macro_f1"]})
    print(f"Saved: {out / 'hierarchical_v3_report.json'}")


if __name__ == "__main__":
    main()
