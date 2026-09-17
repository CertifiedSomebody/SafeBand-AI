from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

LABEL = "IAQ_class"
RAW_SENSOR = ["temperature_c", "pressure_hpa", "humidity_rh", "gas_resistance_ohms"]
# IAQ_proxy is deliberately excluded from the primary model because it is an IAQ-derived
# variable and may encode the same rule used to construct IAQ_class.
BASE = RAW_SENSOR + ["heat_stable"]
DROP_ALWAYS = {
    "Date", "location", LABEL,
    "temperature_c_zscore", "pressure_hpa_zscore", "humidity_rh_zscore",
    "gas_resistance_ohms_zscore", "IAQ_proxy_zscore", "Z-score", "samples_per_hour",
    "IAQ_proxy",
}


def discover(path: str | None) -> Path:
    if path:
        p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"AIRWISE CSV not found: {p}")
        return p
    d = ROOT / "datasets" / "processed" / "airwise_bme680"
    candidates = sorted(d.glob("*.csv"))
    preferred = [p for p in candidates if p.name == "airwise_bme680_features.csv"]
    candidates = preferred + [p for p in candidates if p not in preferred]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No processed AIRWISE CSV found under {d}. Run prepare_airwise_bme680.py first.")
    raise RuntimeError("Multiple processed AIRWISE CSVs found; pass --input explicitly: " + ", ".join(map(str, candidates)))


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date", "location", LABEL, *RAW_SENSOR}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Invalid Date values found.")
    if df[LABEL].isna().any() or df["location"].isna().any():
        raise ValueError("Missing labels or locations found.")
    for c in RAW_SENSOR:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[RAW_SENSOR].isna().any().any():
        raise ValueError("Missing/non-numeric raw sensor values found.")
    return df


def _rolling(grouped: pd.core.groupby.SeriesGroupBy, width: int, stat: str) -> pd.Series:
    r = grouped.rolling(width, min_periods=max(2, width // 2)).agg(stat)
    # Second MultiIndex level is the original row index after the global sort/reset.
    return r.reset_index(level=0, drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values(["location", "Date"], kind="mergesort").reset_index(drop=True)

    # Pointwise transformations of physical BME680 measurements.
    gas = x["gas_resistance_ohms"].clip(lower=1.0)
    temp_abs = x["temperature_c"].abs().clip(lower=1.0)
    x["gas_log10"] = np.log10(gas)
    x["gas_inv_log"] = 1.0 / x["gas_log10"].clip(lower=0.1)
    x["temp_humidity_product"] = x["temperature_c"] * x["humidity_rh"]
    x["temp_humidity_gap"] = x["temperature_c"] - x["humidity_rh"]
    x["pressure_temp_ratio"] = x["pressure_hpa"] / temp_abs
    x["gas_temp_ratio"] = x["gas_resistance_ohms"] / temp_abs

    minute = x["Date"].dt.hour * 60 + x["Date"].dt.minute
    x["tod_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    x["tod_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    x["dow_sin"] = np.sin(2 * np.pi * x["Date"].dt.dayofweek / 7.0)
    x["dow_cos"] = np.cos(2 * np.pi * x["Date"].dt.dayofweek / 7.0)

    # All temporal features are causal: current sample and preceding samples only.
    # They are computed after sorting within each physical environment.
    for c in RAW_SENSOR:
        g = x.groupby("location", sort=False)[c]
        x[f"{c}_diff1"] = g.diff(1)
        x[f"{c}_diff5"] = g.diff(5)
        prev5 = g.shift(5)
        x[f"{c}_pct5"] = ((x[c] - prev5) / prev5.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        for w in (5, 15, 30):
            for stat, suffix in (("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")):
                s = _rolling(g, w, stat).reindex(x.index)
                x[f"{c}_roll{w}_{suffix}"] = s.to_numpy()

    return x


def split_chronological(df: pd.DataFrame):
    parts = {"train": [], "validation": [], "test": []}
    for location, g in df.groupby("location", sort=True):
        g = g.sort_values("Date", kind="mergesort").reset_index(drop=True)
        n = len(g)
        i = int(0.60 * n)
        j = int(0.80 * n)
        if min(i, j - i, n - j) == 0:
            raise ValueError(f"Location {location!r} is too small for a 60/20/20 split.")
        parts["train"].append(g.iloc[:i])
        parts["validation"].append(g.iloc[i:j])
        parts["test"].append(g.iloc[j:])
    return tuple(pd.concat(parts[k], ignore_index=True) for k in ("train", "validation", "test"))


def xy(df: pd.DataFrame, columns: list[str] | None = None):
    if columns is None:
        columns = [
            c for c in df.columns
            if c not in DROP_ALWAYS and c != LABEL and c != "Date"
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    X = df[columns].replace([np.inf, -np.inf], np.nan)
    y = df[LABEL].astype(str)
    return X, y, columns


def metrics(y, pred, labels):
    mp, mr, mf, _ = precision_recall_fscore_support(
        y, pred, labels=labels, average="macro", zero_division=0
    )
    pp, rr, ff, _ = precision_recall_fscore_support(
        y, pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "per_class": {
            str(c): {"precision": float(a), "recall": float(b), "f1": float(d)}
            for c, a, b, d in zip(labels, pp, rr, ff)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Train research-safe AIRWISE BME680 V2 classifier.")
    ap.add_argument("--input")
    ap.add_argument("--out-dir", default="models/bme680_airwise_v2")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    src = discover(args.input)
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = load_source(src)
    df = build_features(raw)
    train, val, test = split_chronological(df)

    Xtr, ytr, cols = xy(train)
    Xv, yv, _ = xy(val, cols)
    Xt, yt, _ = xy(test, cols)

    le = LabelEncoder()
    ytri = le.fit_transform(ytr)
    yvi = le.transform(yv)
    yti = le.transform(yt)
    labels = list(range(len(le.classes_)))

    candidates = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=450, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=args.random_state
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=2, n_jobs=-1, random_state=args.random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=args.random_state
        ),
    }

    validation = {}
    for name, model in candidates.items():
        model.fit(Xtr, ytri)
        validation[name] = metrics(yvi, model.predict(Xv), labels)

    best_name = max(
        validation,
        key=lambda n: (
            validation[n]["macro_f1"],
            validation[n]["balanced_accuracy"],
            validation[n]["accuracy"],
        ),
    )

    # Refit only after model selection. The test partition is never used for selection.
    train_val = pd.concat([train, val], ignore_index=True)
    Xtv, ytv, _ = xy(train_val, cols)
    ytv_i = le.transform(ytv)
    best = candidates[best_name]
    best.fit(Xtv, ytv_i)
    test_pred = best.predict(Xt)
    test_metrics = metrics(yti, test_pred, labels)

    predictions = pd.DataFrame({
        "Date": test["Date"].to_numpy(),
        "location": test["location"].to_numpy(),
        "actual": test[LABEL].to_numpy(),
        "prediction": le.inverse_transform(test_pred),
    })
    predictions.to_csv(out / "test_predictions.csv", index=False)

    artifact = {
        "version": "SafeBand BME680 AIRWISE V2.1",
        "model": best,
        "label_encoder": le,
        "feature_columns": cols,
        "source_file": str(src),
        "split": "chronological 60/20/20 per location; causal features; train+validation refit; untouched test",
        "selection_metric": "validation macro_f1 -> balanced_accuracy -> accuracy",
        "label_leakage_guard": "IAQ_proxy excluded from primary model because it is IAQ-derived and may encode IAQ_class construction",
        "notes": "IAQ_class and all IAQ_proxy-derived columns are excluded from primary model inputs.",
    }
    joblib.dump(artifact, out / "bme680_airwise_v2.joblib")

    report = {
        "version": "SafeBand BME680 AIRWISE V2.1",
        "source": str(src),
        "rows": len(df),
        "features": len(cols),
        "feature_columns": cols,
        "excluded_from_primary": sorted(DROP_ALWAYS),
        "class_names": le.classes_.tolist(),
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "split_by_location": {
            loc: {k: int(sum(len(g) for g in [train[train.location == loc], val[val.location == loc], test[test.location == loc]][i:i+1]))
                  for i, k in enumerate(("train", "validation", "test"))}
            for loc in sorted(df.location.unique())
        },
        "validation_candidates": validation,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "research_guardrails": {
            "test_used_for_selection": False,
            "iaQ_proxy_used_as_input": False,
            "future_temporal_values_used": False,
            "label_column_used_as_input": False,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "source": str(src),
        "selected_model": best_name,
        "validation_macro_f1": validation[best_name]["macro_f1"],
        "test": test_metrics,
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
