from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PPG_COLUMNS = [
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
    "raw_hr",
    "raw_spo2",
    "final_hr",
    "final_spo2",
]

VALIDITY_COLUMNS = [
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
]


def run_lengths(
    values: np.ndarray,
) -> list[int]:

    if len(values) == 0:
        return []

    lengths = []

    start = 0

    current = values[0]

    for index in range(
        1,
        len(values),
    ):

        if values[index] != current:

            lengths.append(
                index - start
            )

            start = index
            current = values[index]

    lengths.append(
        len(values) - start
    )

    return lengths


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "SafeBand 2.10.3 PPG "
            "augmentation validator"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--augmented",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    source = pd.read_csv(
        args.source
    )

    augmented = pd.read_csv(
        args.augmented
    )

    report = {
        "stage": "2.10.3",
        "source": str(args.source),
        "augmented": str(args.augmented),
        "checks": {},
    }

    schema_pass = (
        list(source.columns)
        == list(augmented.columns)
    )

    rows_pass = (
        len(source)
        == len(augmented)
    )

    activity_pass = (
        "activity" in source.columns
        and
        "activity" in augmented.columns
        and
        source["activity"].tolist()
        ==
        augmented["activity"].tolist()
    )

    duplicate_count = int(
        augmented.duplicated().sum()
    )

    numeric = augmented.select_dtypes(
        include=[np.number]
    )

    nonfinite_count = int(
        (~np.isfinite(
            numeric.to_numpy()
        )).sum()
    )

    for column in PPG_COLUMNS:

        if column not in augmented.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    validity_pass = True

    for column in VALIDITY_COLUMNS:

        values = pd.to_numeric(
            augmented[column],
            errors="coerce",
        )

        valid_values = set(
            values.dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not valid_values.issubset(
            {0, 1}
        ):
            validity_pass = False

    sentinel_pass = True

    for validity, value in [
        (
            "raw_hr_valid",
            "raw_hr",
        ),
        (
            "raw_spo2_valid",
            "raw_spo2",
        ),
        (
            "final_hr_valid",
            "final_hr",
        ),
        (
            "final_spo2_valid",
            "final_spo2",
        ),
    ]:

        invalid = (
            pd.to_numeric(
                augmented[validity],
                errors="coerce",
            )
            < 0.5
        )

        values = pd.to_numeric(
            augmented[value],
            errors="coerce",
        )

        if not (
            values[invalid]
            == -999
        ).all():

            sentinel_pass = False

    coupling_pass = True

    finger = (
        augmented[
            "finger_detected"
        ].to_numpy()
        >= 0.5
    )

    raw_hr = (
        augmented[
            "raw_hr_valid"
        ].to_numpy()
        >= 0.5
    )

    raw_spo2 = (
        augmented[
            "raw_spo2_valid"
        ].to_numpy()
        >= 0.5
    )

    final_hr = (
        augmented[
            "final_hr_valid"
        ].to_numpy()
        >= 0.5
    )

    final_spo2 = (
        augmented[
            "final_spo2_valid"
        ].to_numpy()
        >= 0.5
    )

    if np.any(
        raw_hr & ~finger
    ):
        coupling_pass = False

    if np.any(
        raw_spo2 & ~finger
    ):
        coupling_pass = False

    if np.any(
        final_hr & ~finger
    ):
        coupling_pass = False

    if np.any(
        final_spo2 & ~finger
    ):
        coupling_pass = False

    finger_rate = float(
        np.mean(finger)
    )

    hr_given_finger = (
        float(
            np.mean(raw_hr[finger])
        )
        if np.any(finger)
        else 0.0
    )

    spo2_given_finger = (
        float(
            np.mean(raw_spo2[finger])
        )
        if np.any(finger)
        else 0.0
    )

    finger_runs = run_lengths(
        finger
    )

    report[
        "checks"
    ] = {

        "schema":
            schema_pass,

        "rows":
            rows_pass,

        "activity":
            activity_pass,

        "duplicates":
            duplicate_count,

        "nonfinite":
            nonfinite_count,

        "validity_flags":
            validity_pass,

        "sentinels":
            sentinel_pass,

        "ppg_coupling":
            coupling_pass,

        "finger_rate":
            finger_rate,

        "raw_hr_given_finger":
            hr_given_finger,

        "raw_spo2_given_finger":
            spo2_given_finger,

        "finger_run_count":
            len(finger_runs),

        "finger_longest_run":
            max(
                finger_runs
            )
            if finger_runs
            else 0,
    }

    overall = (
        schema_pass
        and rows_pass
        and activity_pass
        and duplicate_count == 0
        and nonfinite_count == 0
        and validity_pass
        and sentinel_pass
        and coupling_pass
    )

    report[
        "overall_status"
    ] = (
        "PASS"
        if overall
        else
        "FAIL"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SAFEband MILESTONE 2.10.3 — "
        "PPG PROFILE VALIDATION"
    )

    print(
        f"Schema          : "
        f"{'PASS' if schema_pass else 'FAIL'}"
    )

    print(
        f"Rows            : "
        f"{'PASS' if rows_pass else 'FAIL'}"
    )

    print(
        f"Activity        : "
        f"{'PASS' if activity_pass else 'FAIL'}"
    )

    print(
        f"Duplicates      : "
        f"{duplicate_count}"
    )

    print(
        f"Non-finite      : "
        f"{nonfinite_count}"
    )

    print(
        f"Validity flags  : "
        f"{'PASS' if validity_pass else 'FAIL'}"
    )

    print(
        f"Sentinels       : "
        f"{'PASS' if sentinel_pass else 'FAIL'}"
    )

    print(
        f"PPG coupling    : "
        f"{'PASS' if coupling_pass else 'FAIL'}"
    )

    print(
        f"Finger rate     : "
        f"{finger_rate:.4f}"
    )

    print(
        f"HR | finger     : "
        f"{hr_given_finger:.4f}"
    )

    print(
        f"SpO2 | finger   : "
        f"{spo2_given_finger:.4f}"
    )

    print(
        f"Overall status  : "
        f"{report['overall_status']}"
    )

    print(
        f"Report          : "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
