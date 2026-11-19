import csv
import json
import math
import statistics
from pathlib import Path


BASE = Path("reports/real_data_characterization")
REAL_DIR = Path("datasets/real sensor data")
GENERATED_DIR = Path("datasets/generated")

MODEL_FILE = BASE / "stage_a5_synthetic_running_model.json"
OUTPUT_FILE = BASE / "stage_a5_2_synthetic_running_validation.json"

FILES = {
    "stationary": REAL_DIR / "stationary.csv",
    "stationary_2": REAL_DIR / "stationary_2.csv",
    "walking": REAL_DIR / "walking.csv",
    "running_synthetic": GENERATED_DIR / "running_synthetic_dataset.csv",
}


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    return rows, fields


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def number(value):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return None


def column(fields, candidates):
    lowered = {
        field.lower(): field
        for field in fields
    }

    for candidate in candidates:

        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for field in fields:

        lower = field.lower()

        for candidate in candidates:

            if candidate.lower() in lower:
                return field

    return None


def percentile(values, q):

    if not values:
        return None

    values = sorted(values)

    position = (
        (len(values) - 1)
        * q
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def mean(values):
    return (
        statistics.mean(values)
        if values
        else None
    )


def std(values):
    return (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )


def summarize(values):

    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": mean(values),
        "std": std(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def extract_intervals(rows, elapsed):

    values = []

    for row in rows:

        value = number(
            row.get(elapsed)
        )

        if value is not None:
            values.append(value)

    return [
        values[index] - values[index - 1]
        for index in range(
            1,
            len(values),
        )
    ]


def extract_vector_magnitude(
    rows,
    x_column,
    y_column,
    z_column,
):

    result = []

    for row in rows:

        x = number(
            row.get(x_column)
        )

        y = number(
            row.get(y_column)
        )

        z = number(
            row.get(z_column)
        )

        if (
            x is None
            or y is None
            or z is None
        ):
            continue

        result.append(
            math.sqrt(
                x * x
                + y * y
                + z * z
            )
        )

    return result


def lag1_autocorrelation(values):

    if len(values) < 3:
        return None

    x = values[:-1]
    y = values[1:]

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum(
        (
            a - mean_x
        )
        * (
            b - mean_y
        )
        for a, b in zip(x, y)
    )

    denominator_x = math.sqrt(
        sum(
            (
                value - mean_x
            ) ** 2
            for value in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (
                value - mean_y
            ) ** 2
            for value in y
        )
    )

    denominator = (
        denominator_x
        * denominator_y
    )

    if denominator == 0:
        return None

    return numerator / denominator


def validity_rate(rows, field):

    values = []

    for row in rows:

        value = number(
            row.get(field)
        )

        if value is not None:
            values.append(
                value >= 0.5
            )

    if not values:
        return None

    return (
        sum(values)
        / len(values)
    )


def invalid_sentinel_count(rows, field):

    count = 0

    for row in rows:

        value = row.get(field)

        if value is None:
            continue

        try:
            if float(value) == -999:
                count += 1
        except ValueError:
            pass

    return count


def build_metrics(rows, fields):

    elapsed = column(
        fields,
        ["elapsed_ms"],
    )

    ax = column(
        fields,
        ["accel_x_m_s2", "accel_x"],
    )

    ay = column(
        fields,
        ["accel_y_m_s2", "accel_y"],
    )

    az = column(
        fields,
        ["accel_z_m_s2", "accel_z"],
    )

    gx = column(
        fields,
        ["gyro_x_rad_s", "gyro_x"],
    )

    gy = column(
        fields,
        ["gyro_y_rad_s", "gyro_y"],
    )

    gz = column(
        fields,
        ["gyro_z_rad_s", "gyro_z"],
    )

    intervals = (
        extract_intervals(
            rows,
            elapsed,
        )
        if elapsed
        else []
    )

    normal_intervals = [
        value
        for value in intervals
        if value < 1000
    ]

    stalls = [
        value
        for value in intervals
        if value >= 1000
    ]

    acceleration = (
        extract_vector_magnitude(
            rows,
            ax,
            ay,
            az,
        )
        if ax and ay and az
        else []
    )

    gyro = (
        extract_vector_magnitude(
            rows,
            gx,
            gy,
            gz,
        )
        if gx and gy and gz
        else []
    )

    ppg_fields = {
        "finger_detected": column(
            fields,
            ["finger_detected"],
        ),

        "raw_hr_valid": column(
            fields,
            ["raw_hr_valid"],
        ),

        "raw_spo2_valid": column(
            fields,
            ["raw_spo2_valid"],
        ),

        "final_hr_valid": column(
            fields,
            ["final_hr_valid"],
        ),

        "final_spo2_valid": column(
            fields,
            ["final_spo2_valid"],
        ),
    }

    ppg = {}

    for name, field in ppg_fields.items():

        if field:
            ppg[name] = {
                "rate": validity_rate(
                    rows,
                    field,
                ),
                "minus_999": invalid_sentinel_count(
                    rows,
                    field,
                ),
            }

    return {
        "rows": len(rows),

        "columns": len(fields),

        "intervals_ms": summarize(
            intervals
        ),

        "normal_intervals_ms": summarize(
            normal_intervals
        ),

        "stall_intervals_ms": summarize(
            stalls
        ),

        "stall_count": len(stalls),

        "acceleration_magnitude": {
            **summarize(acceleration),
            "lag1_autocorrelation":
                lag1_autocorrelation(
                    acceleration
                ),
        },

        "gyro_magnitude": {
            **summarize(gyro),
            "lag1_autocorrelation":
                lag1_autocorrelation(
                    gyro
                ),
        },

        "ppg": ppg,
    }


def ratio(a, b):

    if (
        a is None
        or b is None
        or b == 0
    ):
        return None

    return a / b


def compare_metric(
    synthetic,
    reference,
    field,
):

    synthetic_value = synthetic.get(
        field
    )

    reference_value = reference.get(
        field
    )

    return {
        "synthetic": synthetic_value,
        "reference": reference_value,
        "ratio": ratio(
            synthetic_value,
            reference_value,
        ),
    }


def main():

    model = load_json(
        MODEL_FILE
    )

    datasets = {}

    for name, path in FILES.items():

        rows, fields = load_csv(
            path
        )

        datasets[name] = {
            "rows": rows,
            "fields": fields,
            "metrics": build_metrics(
                rows,
                fields,
            ),
        }

    synthetic = datasets[
        "running_synthetic"
    ]

    walking = datasets[
        "walking"
    ]

    stationary = datasets[
        "stationary"
    ]

    schema_checks = {

        "synthetic_columns":
            len(synthetic["fields"]),

        "walking_columns":
            len(walking["fields"]),

        "stationary_columns":
            len(stationary["fields"]),

        "column_count_matches_walking":
            synthetic["fields"]
            == walking["fields"],

        "column_count_matches_stationary":
            synthetic["fields"]
            == stationary["fields"],

        "activity_values": sorted(
            set(
                row.get(
                    column(
                        synthetic["fields"],
                        ["activity"],
                    )
                )
                for row in synthetic["rows"]
            )
        ),
    }

    motion_comparison = {

        "acceleration_mean": compare_metric(
            synthetic["metrics"][
                "acceleration_magnitude"
            ],
            walking["metrics"][
                "acceleration_magnitude"
            ],
            "mean",
        ),

        "acceleration_std": compare_metric(
            synthetic["metrics"][
                "acceleration_magnitude"
            ],
            walking["metrics"][
                "acceleration_magnitude"
            ],
            "std",
        ),

        "acceleration_lag1": compare_metric(
            synthetic["metrics"][
                "acceleration_magnitude"
            ],
            walking["metrics"][
                "acceleration_magnitude"
            ],
            "lag1_autocorrelation",
        ),

        "gyro_mean": compare_metric(
            synthetic["metrics"][
                "gyro_magnitude"
            ],
            walking["metrics"][
                "gyro_magnitude"
            ],
            "mean",
        ),

        "gyro_std": compare_metric(
            synthetic["metrics"][
                "gyro_magnitude"
            ],
            walking["metrics"][
                "gyro_magnitude"
            ],
            "std",
        ),

        "gyro_lag1": compare_metric(
            synthetic["metrics"][
                "gyro_magnitude"
            ],
            walking["metrics"][
                "gyro_magnitude"
            ],
            "lag1_autocorrelation",
        ),
    }

    ppg_comparison = {}

    for field in [
        "finger_detected",
        "raw_hr_valid",
        "raw_spo2_valid",
        "final_hr_valid",
        "final_spo2_valid",
    ]:

        synthetic_ppg = (
            synthetic["metrics"]["ppg"]
            .get(field)
        )

        walking_ppg = (
            walking["metrics"]["ppg"]
            .get(field)
        )

        ppg_comparison[field] = {
            "synthetic":
                synthetic_ppg,

            "walking":
                walking_ppg,
        }

    timing = {

        "synthetic": synthetic[
            "metrics"
        ][
            "intervals_ms"
        ],

        "walking": walking[
            "metrics"
        ][
            "intervals_ms"
        ],

        "synthetic_normal":
            synthetic[
                "metrics"
            ][
                "normal_intervals_ms"
            ],

        "walking_normal":
            walking[
                "metrics"
            ][
                "normal_intervals_ms"
            ],

        "synthetic_stalls":
            synthetic[
                "metrics"
            ][
                "stall_intervals_ms"
            ],

        "walking_stalls":
            walking[
                "metrics"
            ][
                "stall_intervals_ms"
            ],

        "synthetic_stall_count":
            synthetic[
                "metrics"
            ][
                "stall_count"
            ],

        "walking_stall_count":
            walking[
                "metrics"
            ][
                "stall_count"
            ],
    }

    acceleration_ratio = (
        motion_comparison[
            "acceleration_mean"
        ][
            "ratio"
        ]
    )

    gyro_ratio = (
        motion_comparison[
            "gyro_mean"
        ][
            "ratio"
        ]
    )

    expected_accel = model[
        "generation_parameters"
    ][
        "running_intensity"
    ][
        "acceleration"
    ]

    expected_gyro = model[
        "generation_parameters"
    ][
        "running_intensity"
    ][
        "gyro"
    ]

    # Tolerance is deliberately broad.
    # This stage diagnoses the generator; it does not
    # claim that synthetic RUNNING is physically correct.
    accel_pass = (
        acceleration_ratio is not None
        and abs(
            acceleration_ratio
            - expected_accel
        )
        <= 0.12
    )

    gyro_pass = (
        gyro_ratio is not None
        and abs(
            gyro_ratio
            - expected_gyro
        )
        <= 0.12
    )

    schema_pass = (
        schema_checks[
            "column_count_matches_walking"
        ]
        and schema_checks[
            "column_count_matches_stationary"
        ]
        and schema_checks[
            "activity_values"
        ] == ["RUNNING"]
    )

    validation = {

        "document":
            "SafeBand Stage A.5.2 Synthetic RUNNING Validation",

        "version":
            "1.0",

        "status":
            "VALIDATION_COMPLETE",

        "source_model":
            str(MODEL_FILE),

        "synthetic_dataset":
            str(FILES["running_synthetic"]),

        "datasets": {
            name: {
                "rows":
                    value["metrics"]["rows"],

                "columns":
                    value["metrics"]["columns"],
            }
            for name, value
            in datasets.items()
        },

        "schema": schema_checks,

        "timing": timing,

        "motion": motion_comparison,

        "ppg": ppg_comparison,

        "expected_model_parameters": {
            "acceleration_scale":
                expected_accel,

            "gyro_scale":
                expected_gyro,

            "target_stall_duration_ms":
                model[
                    "generation_parameters"
                ][
                    "timing"
                ][
                    "stall_duration_ms"
                ],

            "target_stall_spacing_ms":
                model[
                    "generation_parameters"
                ][
                    "timing"
                ][
                    "stall_spacing_ms"
                ],
        },

        "automated_checks": {

            "schema":
                schema_pass,

            "acceleration_scale":
                accel_pass,

            "gyro_scale":
                gyro_pass,

            "synthetic_row_count":
                synthetic[
                    "metrics"
                ][
                    "rows"
                ] == model[
                    "generation_parameters"
                ][
                    "default_rows"
                ],

            "synthetic_activity":
                schema_checks[
                    "activity_values"
                ] == ["RUNNING"],

            "has_acquisition_stalls":
                synthetic[
                    "metrics"
                ][
                    "stall_count"
                ] > 0,

            "has_temporal_acceleration_correlation":
                (
                    synthetic[
                        "metrics"
                    ][
                        "acceleration_magnitude"
                    ][
                        "lag1_autocorrelation"
                    ] is not None
                ),

            "has_temporal_gyro_correlation":
                (
                    synthetic[
                        "metrics"
                    ][
                        "gyro_magnitude"
                    ][
                        "lag1_autocorrelation"
                    ] is not None
                ),
        },

        "interpretation": {
            "synthetic_running_is_not_real_evidence":
                True,

            "purpose":
                "Generator validation and downstream software development.",

            "real_running_required_for_final_calibration":
                True,
        },

        "next_stage": {
            "if_checks_pass":
                "A.5.3 Generator refinement and acceptance",

            "if_checks_fail":
                "Diagnose failing empirical constraint before acceptance.",
        },
    }

    checks = validation[
        "automated_checks"
    ]

    validation[
        "overall_pass"
    ] = all(
        checks.values()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            validation,
            handle,
            indent=2,
        )

    print()
    print("=" * 100)
    print("SAFEband STAGE A.5.2 — SYNTHETIC RUNNING VALIDATION")
    print("=" * 100)
    print()

    print(
        f"Synthetic rows : "
        f"{synthetic['metrics']['rows']}"
    )

    print(
        f"Synthetic cols : "
        f"{synthetic['metrics']['columns']}"
    )

    print()

    print(
        "Acceleration mean ratio "
        "(RUNNING/WALKING): "
        f"{acceleration_ratio:.4f}"
        if acceleration_ratio is not None
        else
        "Acceleration mean ratio: unavailable"
    )

    print(
        "Expected acceleration scale : "
        f"{expected_accel:.2f}x"
    )

    print()

    print(
        "Gyro mean ratio "
        "(RUNNING/WALKING): "
        f"{gyro_ratio:.4f}"
        if gyro_ratio is not None
        else
        "Gyro mean ratio: unavailable"
    )

    print(
        "Expected gyro scale         : "
        f"{expected_gyro:.2f}x"
    )

    print()

    print(
        "Synthetic stall count : "
        f"{synthetic['metrics']['stall_count']}"
    )

    print(
        "Walking stall count   : "
        f"{walking['metrics']['stall_count']}"
    )

    print()

    print(
        "Schema check       : "
        f"{'PASS' if schema_pass else 'FAIL'}"
    )

    print(
        "Acceleration check : "
        f"{'PASS' if accel_pass else 'FAIL'}"
    )

    print(
        "Gyro check         : "
        f"{'PASS' if gyro_pass else 'FAIL'}"
    )

    print(
        "Overall validation : "
        f"{'PASS' if validation['overall_pass'] else 'FAIL'}"
    )

    print()

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
