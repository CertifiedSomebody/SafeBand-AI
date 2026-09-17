from __future__ import annotations
import numpy as np
import pandas as pd

RAW = ["temperature_c", "pressure_hpa", "humidity_rh", "gas_resistance_ohms"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in RAW:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Physically meaningful instantaneous features.
    x["gas_log10"] = np.log10(np.clip(x["gas_resistance_ohms"], 1.0, None))
    x["temp_humidity_product"] = x["temperature_c"] * x["humidity_rh"]
    x["temp_humidity_gap"] = x["temperature_c"] - x["humidity_rh"]

    # Within-location temporal dynamics. Grouping prevents cross-room rolling.
    if "location" in x.columns and "Date" in x.columns:
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
        x = x.sort_values(["location", "Date"]).reset_index(drop=True)
        for c in RAW:
            g = x.groupby("location", sort=False)[c]
            x[f"{c}_diff_1"] = g.diff()
            x[f"{c}_pct_1"] = g.pct_change().replace([np.inf, -np.inf], np.nan)
            x[f"{c}_roll_mean_5"] = g.transform(lambda s: s.rolling(5, min_periods=2).mean())
            x[f"{c}_roll_std_5"] = g.transform(lambda s: s.rolling(5, min_periods=2).std())
    return x

def feature_columns(df: pd.DataFrame):
    excluded = {
        "Date", "location", "IAQ_class", "IAQ_proxy",
        "temperature_c_zscore", "pressure_hpa_zscore",
        "humidity_rh_zscore", "gas_resistance_ohms_zscore",
        "IAQ_proxy_zscore", "Z-score", "samples_per_hour"
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
