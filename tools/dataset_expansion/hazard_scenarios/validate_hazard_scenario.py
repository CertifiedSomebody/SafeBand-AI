from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL = -999.0

META_COLUMNS = {
    "hazard_label",
    "hazard_type",
    "hazard_event_id",
    "is_hazard",
}


def check_schema(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:
    source_columns = list(source.columns)

    required_added = {
        "hazard_label",
        "hazard_type",
        "hazard_event_id",
        "is_hazard",
    }

    return (
        all(
            column in augmented.columns
            for column in required_added
        )
        and all(
            column in augmented.columns
            for column in source_columns
        )
    )


def check_rows(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:
    return len(source) == len(augmented)


def check_sample_index(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    if "sample_index" not in source.columns:
        return True

    return source["sample_index"].equals(
        augmented["sample_index"]
    )


def check_activity(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    if "activity" not in source.columns:
        return True

    return source["activity"].equals(
        augmented["activity"]
    )


def check_validity_flags(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    for column in source.columns:

        lowered = column.lower()

        if (
            "valid" not in lowered
            and "finger_detected" not in lowered
        ):
            continue

        source_values = pd.to_numeric(
            source[column],
            errors="coerce",
        )

        augmented_values = pd.to_numeric(
            augmented[column],
            errors="coerce",
        )

        if not source_values.equals(
            augmented_values
        ):
            return False

    return True


def check_sentinels(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    for column in source.columns:

        source_values = pd.to_numeric(
            source[column],
            errors="coerce",
        )

        augmented_values = pd.to_numeric(
            augmented[column],
            errors="coerce",
        )

        source_sentinel = (
            source_values == SENTINEL
        )

        if not source_sentinel.any():
            continue

        if not (
            augmented_values[
                source_sentinel
            ] == SENTINEL
        ).all():
            return False

    return True


def check_elapsed(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    if "elapsed_ms" not in source.columns:
        return True

    source_values = pd.to_numeric(
        source["elapsed_ms"],
        errors="coerce",
    )

    augmented_values = pd.to_numeric(
        augmented["elapsed_ms"],
        errors="coerce",
    )

    return (
        source_values.equals(
            augmented_values
        )
    )


def check_nonfinite(
    augmented: pd.DataFrame,
) -> bool:

    numeric = augmented.select_dtypes(
        include=[np.number]
    )

    return bool(
        np.isfinite(
            numeric.to_numpy(
                dtype=float
            )
        ).all()
    )


def check_hazard_labels(
    augmented: pd.DataFrame,
) -> bool:

    required = {
        "hazard_label",
        "hazard_type",
        "hazard_event_id",
        "is_hazard",
    }

    if not required.issubset(
        augmented.columns
    ):
        return False

    hazard = pd.to_numeric(
        augmented["hazard_label"],
        errors="coerce",
    )

    is_hazard = pd.to_numeric(
        augmented["is_hazard"],
        errors="coerce",
    )

    if not hazard.isin(
        [0, 1]
    ).all():
        return False

    if not hazard.equals(
        is_hazard
    ):
        return False

    return bool(
        (hazard == 1).any()
    )


def check_hazard_type(
    augmented: pd.DataFrame,
) -> bool:

    hazard = augmented[
        "hazard_label"
    ]

    hazard_rows = (
        hazard == 1
    )

    types = augmented.loc[
        hazard_rows,
        "hazard_type",
    ]

    if types.empty:
        return False

    if (
        types.isna()
        | (types == "")
        | (types == "NONE")
    ).any():
        return False

    return True


def count_changed_values(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> int:

    comparable = [
        column
        for column in source.columns
        if column in augmented.columns
    ]

    changed = (
        source[comparable]
        != augmented[comparable]
    )

    changed = changed.fillna(False)

    return int(
        changed.to_numpy().sum()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--augmented",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    source = pd.read_csv(
        args.source
    )

    augmented = pd.read_csv(
        args.augmented
    )

    checks = {
        "Schema": check_schema(
            source,
            augmented,
        ),

        "Rows": check_rows(
            source,
            augmented,
        ),

        "Sample index": check_sample_index(
            source,
            augmented,
        ),

        "Activity preserved": check_activity(
            source,
            augmented,
        ),

        "Validity flags": check_validity_flags(
            source,
            augmented,
        ),

        "Sentinels": check_sentinels(
            source,
            augmented,
        ),

        "Elapsed preserved": check_elapsed(
            source,
            augmented,
        ),

        "Non-finite": check_nonfinite(
            augmented,
        ),

        "Hazard labels": check_hazard_labels(
            augmented,
        ),

        "Hazard types": check_hazard_type(
            augmented,
        ),
    }

    changed_values = count_changed_values(
        source,
        augmented,
    )

    checks["Hazard sensor effect"] = (
        changed_values > 0
    )

    overall = all(
        checks.values()
    )

    report = {
        "stage": "2.10.7",
        "status": (
            "PASS"
            if overall
            else "FAIL"
        ),
        "checks": checks,
        "changed_values": changed_values,
        "source_rows": len(source),
        "augmented_rows": len(augmented),
        "hazard_rows": int(
            (
                augmented["hazard_label"]
                == 1
            ).sum()
        ),
        "hazard_types": sorted(
            augmented.loc[
                augmented["hazard_label"] == 1,
                "hazard_type",
            ].dropna().unique().tolist()
        ),
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print()
    print(
        "SAFEband MILESTONE 2.10.7 — "
        "HAZARD SCENARIO VALIDATION"
    )
    print()

    for name, passed in checks.items():
        print(
            f"{name:<24}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        f"{'Changed values':<24}: "
        f"{changed_values}"
    )

    print(
        f"{'Hazard rows':<24}: "
        f"{report['hazard_rows']}"
    )

    print(
        f"{'Overall status':<24}: "
        f"{'PASS' if overall else 'FAIL'}"
    )

    print(
        f"Report              : {output}"
    )

    raise SystemExit(
        0 if overall else 1
    )


if __name__ == "__main__":
    main()
