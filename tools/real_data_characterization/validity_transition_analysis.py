import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("datasets/real sensor data")

OUTPUT = Path(
    "reports/real_data_characterization/"
    "stage_a_validity_transition_analysis.json"
)

FILES = [
    INPUT / "stationary.csv",
    INPUT / "stationary_2.csv",
    INPUT / "walking.csv",
]

STATE_COLUMNS = [
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
]

PAIR_COLUMNS = [
    ("finger_detected", "raw_hr_valid"),
    ("finger_detected", "raw_spo2_valid"),
    ("finger_detected", "final_hr_valid"),
    ("finger_detected", "final_spo2_valid"),
    ("raw_hr_valid", "raw_spo2_valid"),
    ("raw_hr_valid", "final_hr_valid"),
    ("raw_spo2_valid", "final_spo2_valid"),
    ("final_hr_valid", "final_spo2_valid"),
]


def as_bool(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )


def transition_counts(series):
    values = as_bool(series).to_numpy()

    result = {
        "valid_to_valid": 0,
        "valid_to_invalid": 0,
        "invalid_to_valid": 0,
        "invalid_to_invalid": 0,
    }

    for previous, current in zip(values[:-1], values[1:]):
        if previous and current:
            result["valid_to_valid"] += 1
        elif previous and not current:
            result["valid_to_invalid"] += 1
        elif not previous and current:
            result["invalid_to_valid"] += 1
        else:
            result["invalid_to_invalid"] += 1

    total = sum(result.values())

    probabilities = {}

    if total:
        probabilities = {
            key: value / total
            for key, value in result.items()
        }

    result["total_transitions"] = total
    result["probabilities"] = probabilities

    return result


def run_lengths(series):
    values = as_bool(series).to_numpy()

    if len(values) == 0:
        return {
            "valid": {},
            "invalid": {},
        }

    runs = {
        "valid": [],
        "invalid": [],
    }

    current = values[0]
    length = 1

    for value in values[1:]:
        if value == current:
            length += 1
        else:
            runs["valid" if current else "invalid"].append(length)
            current = value
            length = 1

    runs["valid" if current else "invalid"].append(length)

    result = {}

    for state in ["valid", "invalid"]:
        values_for_state = runs[state]

        if not values_for_state:
            result[state] = {
                "count": 0,
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "max": 0,
            }
            continue

        array = np.asarray(values_for_state, dtype=float)

        result[state] = {
            "count": int(len(array)),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
            "p95": float(np.quantile(array, 0.95)),
            "max": int(np.max(array)),
        }

    return result


def conditional_probability(df, condition, target):
    condition_values = as_bool(df[condition])
    target_values = as_bool(df[target])

    mask = condition_values

    count = int(mask.sum())

    if count == 0:
        return {
            "condition": condition,
            "target": target,
            "condition_count": 0,
            "target_valid_count": 0,
            "probability": None,
        }

    target_count = int(target_values[mask].sum())

    return {
        "condition": condition,
        "target": target,
        "condition_count": count,
        "target_valid_count": target_count,
        "probability": target_count / count,
    }


def pairwise_state_table(df, first, second):
    first_values = as_bool(df[first]).to_numpy()
    second_values = as_bool(df[second]).to_numpy()

    table = {
        "first_valid_second_valid": 0,
        "first_valid_second_invalid": 0,
        "first_invalid_second_valid": 0,
        "first_invalid_second_invalid": 0,
    }

    for first_value, second_value in zip(
        first_values,
        second_values,
    ):
        if first_value and second_value:
            table["first_valid_second_valid"] += 1
        elif first_value and not second_value:
            table["first_valid_second_invalid"] += 1
        elif not first_value and second_value:
            table["first_invalid_second_valid"] += 1
        else:
            table["first_invalid_second_invalid"] += 1

    total = len(first_values)

    if total:
        table["fractions"] = {
            key: value / total
            for key, value in table.items()
        }
    else:
        table["fractions"] = {}

    return table


def timing_gap_analysis(df):
    elapsed = pd.to_numeric(
        df["elapsed_ms"],
        errors="coerce",
    )

    delta = elapsed.diff()

    gaps = delta[delta > 1000]

    if gaps.empty:
        return {
            "gap_count": 0,
            "gap_values_ms": [],
            "gap_mean_ms": None,
            "gap_median_ms": None,
            "gap_std_ms": None,
            "gap_min_ms": None,
            "gap_max_ms": None,
            "gap_positions": [],
            "gap_spacing_ms": [],
            "gap_spacing_mean_ms": None,
            "gap_spacing_median_ms": None,
        }

    positions = gaps.index.to_list()

    gap_values = gaps.to_numpy(dtype=float)

    position_spacing = np.diff(
        elapsed.iloc[positions].to_numpy(dtype=float)
    )

    return {
        "gap_count": int(len(gap_values)),
        "gap_values_ms": [
            float(x)
            for x in gap_values
        ],
        "gap_mean_ms": float(np.mean(gap_values)),
        "gap_median_ms": float(np.median(gap_values)),
        "gap_std_ms": float(np.std(gap_values)),
        "gap_min_ms": float(np.min(gap_values)),
        "gap_max_ms": float(np.max(gap_values)),
        "gap_positions": [
            int(x)
            for x in positions
        ],
        "gap_spacing_ms": [
            float(x)
            for x in position_spacing
        ],
        "gap_spacing_mean_ms": (
            float(np.mean(position_spacing))
            if len(position_spacing)
            else None
        ),
        "gap_spacing_median_ms": (
            float(np.median(position_spacing))
            if len(position_spacing)
            else None
        ),
    }


def analyze_file(path):
    df = pd.read_csv(path)

    result = {
        "file": str(path),
        "rows": int(len(df)),
        "state_transitions": {},
        "run_lengths": {},
        "pairwise_states": {},
        "conditional_probabilities": [],
        "timing_gaps": timing_gap_analysis(df),
    }

    for column in STATE_COLUMNS:
        if column not in df.columns:
            continue

        result["state_transitions"][column] = (
            transition_counts(df[column])
        )

        result["run_lengths"][column] = (
            run_lengths(df[column])
        )

    for first, second in PAIR_COLUMNS:
        if first not in df.columns or second not in df.columns:
            continue

        key = f"{first}__{second}"

        result["pairwise_states"][key] = (
            pairwise_state_table(
                df,
                first,
                second,
            )
        )

        result["conditional_probabilities"].append(
            conditional_probability(
                df,
                first,
                second,
            )
        )

    # Additional useful conditional relationships.
    for condition in ["finger_detected"]:
        for target in [
            "raw_hr_valid",
            "raw_spo2_valid",
            "final_hr_valid",
            "final_spo2_valid",
        ]:
            if condition in df.columns and target in df.columns:
                result["conditional_probabilities"].append(
                    conditional_probability(
                        df,
                        condition,
                        target,
                    )
                )

    return result


def main():
    output = {
        "analysis": (
            "SafeBand validity transition and "
            "acquisition gap analysis"
        ),
        "version": "1.0",
        "files": [],
    }

    for path in FILES:
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"Analyzing: {path.name}")

        output["files"].append(
            analyze_file(path)
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
