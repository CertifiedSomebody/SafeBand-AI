import csv
import json
import math
import random
from pathlib import Path


BASE = Path("reports/real_data_characterization")
REAL_DIR = Path("datasets/real sensor data")
GENERATED_DIR = Path("datasets/generated")

MODEL_FILE = BASE / "stage_a5_synthetic_running_model.json"
SOURCE_FILE = REAL_DIR / "walking.csv"

OUTPUT_FILE = GENERATED_DIR / "running_synthetic_dataset.csv"
PROVENANCE_FILE = GENERATED_DIR / "running_synthetic_dataset.provenance.json"

PPG_MEASUREMENT_FLAGS = {
    "raw_hr": "raw_hr_valid",
    "raw_spo2": "raw_spo2_valid",
    "final_hr": "final_hr_valid",
    "final_spo2": "final_spo2_valid",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        fields = reader.fieldnames

    if not fields:
        raise ValueError("CSV contains no header.")

    return rows, fields


def number(value):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return None


def infer_column(fields, candidates):

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


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value),
    )


def smoothstep(value):

    value = clamp(
        value,
        0.0,
        1.0,
    )

    return (
        value
        * value
        * (3.0 - 2.0 * value)
    )


def magnitude(x, y, z):
    return math.sqrt(
        x * x
        + y * y
        + z * z
    )


def extract_intervals(
    rows,
    elapsed_column,
):

    elapsed = []

    for row in rows:

        value = number(
            row.get(elapsed_column)
        )

        if value is not None:
            elapsed.append(value)

    return [
        elapsed[i] - elapsed[i - 1]
        for i in range(
            1,
            len(elapsed),
        )
        if elapsed[i] >= elapsed[i - 1]
    ]


def empirical_timing(
    rows,
    elapsed_column,
):

    intervals = extract_intervals(
        rows,
        elapsed_column,
    )

    normal = [
        value
        for value in intervals
        if value < 1000
    ]

    stalls = [
        value
        for value in intervals
        if value >= 1000
    ]

    if not normal:
        normal = [394.0]

    if not stalls:
        stalls = [5400.0]

    return {
        "intervals": intervals,
        "normal": normal,
        "stalls": stalls,
    }


def choose_transition_rows(
    model,
    rng,
):

    config = (
        model[
            "session_model"
        ][
            "transition_behavior"
        ][
            "walking_to_running"
        ][
            "duration_rows"
        ]
    )

    return rng.randint(
        config["minimum"],
        config["maximum"],
    )


def interpolate_scale(
    index,
    total_rows,
    transition_rows,
    target,
):

    transition_rows = max(
        1,
        transition_rows,
    )

    if index < transition_rows:

        phase = smoothstep(
            index / transition_rows
        )

        return (
            1.0
            + (target - 1.0)
            * phase
        )

    recovery_start = (
        total_rows
        - transition_rows
    )

    if index >= recovery_start:

        phase = smoothstep(
            (
                index
                - recovery_start
            )
            / transition_rows
        )

        return (
            target
            + (1.0 - target)
            * phase
        )

    return target


def identify_columns(fields):

    return {
        "elapsed": infer_column(
            fields,
            ["elapsed_ms"],
        ),

        "sample_index": infer_column(
            fields,
            ["sample_index"],
        ),

        "activity": infer_column(
            fields,
            ["activity"],
        ),

        "accel_x": infer_column(
            fields,
            [
                "accel_x_m_s2",
                "accel_x",
            ],
        ),

        "accel_y": infer_column(
            fields,
            [
                "accel_y_m_s2",
                "accel_y",
            ],
        ),

        "accel_z": infer_column(
            fields,
            [
                "accel_z_m_s2",
                "accel_z",
            ],
        ),

        "gyro_x": infer_column(
            fields,
            [
                "gyro_x_rad_s",
                "gyro_x",
            ],
        ),

        "gyro_y": infer_column(
            fields,
            [
                "gyro_y_rad_s",
                "gyro_y",
            ],
        ),

        "gyro_z": infer_column(
            fields,
            [
                "gyro_z_rad_s",
                "gyro_z",
            ],
        ),

        "accel_magnitude": infer_column(
            fields,
            [
                "accel_magnitude",
                "acceleration_magnitude",
            ],
        ),

        "gyro_magnitude": infer_column(
            fields,
            [
                "gyro_magnitude",
            ],
        ),

        "raw_hr": infer_column(
            fields,
            ["raw_hr"],
        ),

        "raw_spo2": infer_column(
            fields,
            ["raw_spo2"],
        ),

        "final_hr": infer_column(
            fields,
            ["final_hr"],
        ),

        "final_spo2": infer_column(
            fields,
            ["final_spo2"],
        ),

        "finger": infer_column(
            fields,
            ["finger_detected"],
        ),

        "raw_hr_valid": infer_column(
            fields,
            ["raw_hr_valid"],
        ),

        "raw_spo2_valid": infer_column(
            fields,
            ["raw_spo2_valid"],
        ),

        "final_hr_valid": infer_column(
            fields,
            ["final_hr_valid"],
        ),

        "final_spo2_valid": infer_column(
            fields,
            ["final_spo2_valid"],
        ),
    }


def generate_ppg_state(
    source,
    rng,
    columns,
):

    finger_value = number(
        source.get(
            columns["finger"]
        )
    )

    source_finger = (
        finger_value is not None
        and finger_value >= 0.5
    )

    finger_probability = (
        0.97
        if source_finger
        else 0.75
    )

    finger = (
        rng.random()
        < finger_probability
    )

    if not finger:

        return {
            "finger": False,
            "raw_hr_valid": False,
            "raw_spo2_valid": False,
            "final_hr_valid": False,
            "final_spo2_valid": False,
        }

    raw_hr = (
        rng.random()
        < 0.88
    )

    raw_spo2 = (
        rng.random()
        < 0.42
    )

    final_hr = (
        raw_hr
        and rng.random() < 0.96
    )

    final_spo2 = (
        raw_spo2
        and rng.random() < 0.85
    )

    return {
        "finger": True,
        "raw_hr_valid": raw_hr,
        "raw_spo2_valid": raw_spo2,
        "final_hr_valid": final_hr,
        "final_spo2_valid": final_spo2,
    }


def write_ppg_state(
    row,
    state,
    columns,
    source,
    valid_measurements,
    rng,
):

    mapping = {
        "finger": "finger",
        "raw_hr_valid": "raw_hr_valid",
        "raw_spo2_valid": "raw_spo2_valid",
        "final_hr_valid": "final_hr_valid",
        "final_spo2_valid": "final_spo2_valid",
    }

    for state_key, column_key in mapping.items():

        column = columns.get(
            column_key
        )

        if column is None:
            continue

        row[column] = (
            "1"
            if state[state_key]
            else "0"
        )

    for measurement_key, validity_key in (
        PPG_MEASUREMENT_FLAGS.items()
    ):
        measurement_column = columns.get(
            measurement_key
        )
        validity_column = columns.get(
            validity_key
        )

        if (
            measurement_column is None
            or validity_column is None
        ):
            continue

        if not state[validity_key]:
            row[measurement_column] = "-999"
            continue

        source_value = number(
            source.get(measurement_column)
        )
        source_validity = number(
            source.get(validity_column)
        )

        if (
            source_value is None
            or source_value == -999
            or source_validity is None
            or source_validity < 0.5
        ):
            candidates = valid_measurements[
                measurement_key
            ]
            if not candidates:
                raise ValueError(
                    "No valid source measurements available for "
                    f"{measurement_key}."
                )
            source_value = rng.choice(candidates)

        row[measurement_column] = f"{source_value:.6f}"


def main():

    model = load_json(
        MODEL_FILE
    )

    source_rows, fields = load_csv(
        SOURCE_FILE
    )

    if len(source_rows) < 100:
        raise ValueError(
            "walking.csv does not contain enough rows."
        )

    columns = identify_columns(
        fields
    )

    required = [
        "elapsed",
        "sample_index",
        "activity",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]

    missing = [
        key
        for key in required
        if columns.get(key) is None
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    valid_measurements = {
        measurement_key: []
        for measurement_key in PPG_MEASUREMENT_FLAGS
    }
    for source_row in source_rows:
        for measurement_key, validity_key in (
            PPG_MEASUREMENT_FLAGS.items()
        ):
            measurement_column = columns.get(
                measurement_key
            )
            validity_column = columns.get(
                validity_key
            )
            value = number(
                source_row.get(measurement_column)
            )
            valid = number(
                source_row.get(validity_column)
            )
            if (
                value is not None
                and value != -999
                and valid is not None
                and valid >= 0.5
            ):
                valid_measurements[measurement_key].append(
                    value
                )

    rng = random.Random(
        model[
            "generation_parameters"
        ][
            "random_seed"
        ]
    )

    target_rows = model[
        "generation_parameters"
    ][
        "default_rows"
    ]

    acceleration_scale = model[
        "generation_parameters"
    ][
        "running_intensity"
    ][
        "acceleration"
    ]

    gyro_scale = model[
        "generation_parameters"
    ][
        "running_intensity"
    ][
        "gyro"
    ]

    timing = empirical_timing(
        source_rows,
        columns["elapsed"],
    )

    normal_intervals = timing[
        "normal"
    ]

    stall_intervals = timing[
        "stalls"
    ]

    real_interval_count = len(
        timing["intervals"]
    )

    real_stall_count = len(
        stall_intervals
    )

    real_stall_fraction = (
        real_stall_count
        / real_interval_count
        if real_interval_count
        else 0.0
    )

    # Use the measured walking stall fraction.
    #
    # A stall is scheduled by interval count rather than
    # elapsed-time threshold. This prevents a stall itself
    # from shifting the next scheduling point and producing
    # excessive stall frequency.
    target_stall_count = round(
        (
            target_rows - 1
        )
        * real_stall_fraction
    )

    target_stall_count = max(
        1,
        target_stall_count,
    )

    stall_positions = set()

    if target_stall_count > 0:

        spacing = (
            (target_rows - 1)
            / target_stall_count
        )

        for index in range(
            target_stall_count
        ):

            position = round(
                (
                    index + 1
                )
                * spacing
            )

            position = clamp(
                position,
                1,
                target_rows - 1,
            )

            stall_positions.add(
                int(position)
            )

    transition_rows = (
        choose_transition_rows(
            model,
            rng,
        )
    )

    generated = []

    elapsed = 0.0
    source_index = 0

    for index in range(
        target_rows
    ):

        source = source_rows[
            source_index
            % len(source_rows)
        ]

        row = dict(source)

        acceleration_factor = (
            interpolate_scale(
                index,
                target_rows,
                transition_rows,
                acceleration_scale,
            )
        )

        gyro_factor = (
            interpolate_scale(
                index,
                target_rows,
                transition_rows,
                gyro_scale,
            )
        )

        ax = number(
            source.get(
                columns["accel_x"]
            )
        ) or 0.0

        ay = number(
            source.get(
                columns["accel_y"]
            )
        ) or 0.0

        az = number(
            source.get(
                columns["accel_z"]
            )
        ) or 0.0

        gx = number(
            source.get(
                columns["gyro_x"]
            )
        ) or 0.0

        gy = number(
            source.get(
                columns["gyro_y"]
            )
        ) or 0.0

        gz = number(
            source.get(
                columns["gyro_z"]
            )
        ) or 0.0

        # Preserve waveform shape while applying
        # a stateful intensity transform.
        ax *= (
            acceleration_factor
            * 0.985
        )

        ay *= (
            acceleration_factor
            * 0.985
        )

        az *= (
            acceleration_factor
            * 0.985
        )

        gx *= (
            gyro_factor
            * 0.985
        )

        gy *= (
            gyro_factor
            * 0.985
        )

        gz *= (
            gyro_factor
            * 0.985
        )

        row[
            columns["accel_x"]
        ] = f"{ax:.6f}"

        row[
            columns["accel_y"]
        ] = f"{ay:.6f}"

        row[
            columns["accel_z"]
        ] = f"{az:.6f}"

        row[
            columns["gyro_x"]
        ] = f"{gx:.6f}"

        row[
            columns["gyro_y"]
        ] = f"{gy:.6f}"

        row[
            columns["gyro_z"]
        ] = f"{gz:.6f}"

        if columns["accel_magnitude"]:

            row[
                columns["accel_magnitude"]
            ] = (
                f"{magnitude(ax, ay, az):.6f}"
            )

        if columns["gyro_magnitude"]:

            row[
                columns["gyro_magnitude"]
            ] = (
                f"{magnitude(gx, gy, gz):.6f}"
            )

        row[
            columns["activity"]
        ] = "RUNNING"

        # Timing.
        #
        # Every interval starts from the measured normal
        # walking distribution. Scheduled stalls use the
        # actual measured stall distribution.
        if index == 0:

            interval = 0.0

        else:

            interval = rng.choice(
                normal_intervals
            )

            if index in stall_positions:

                interval = (
                    rng.choice(
                        stall_intervals
                    )
                )

        elapsed += interval

        row[
            columns["elapsed"]
        ] = f"{elapsed:.0f}"

        row[
            columns["sample_index"]
        ] = str(index)

        # Activity-conditioned PPG state.
        ppg_state = generate_ppg_state(
            source,
            rng,
            columns,
        )

        write_ppg_state(
            row,
            ppg_state,
            columns,
            source,
            valid_measurements,
            rng,
        )

        generated.append(
            row
        )

        source_index += 1

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            generated
        )

    provenance = {

        "dataset":
            "running_synthetic_dataset.csv",

        "status":
            "SYNTHETIC_DEVELOPMENT_FIXTURE",

        "activity":
            "RUNNING",

        "validated_against_real_running":
            False,

        "source_dataset":
            "walking.csv",

        "source_real_data":
            [
                "stationary.csv",
                "stationary_2.csv",
                "walking.csv",
            ],

        "generation_model":
            str(MODEL_FILE),

        "random_seed":
            model[
                "generation_parameters"
            ][
                "random_seed"
            ],

        "rows":
            target_rows,

        "columns":
            len(fields),

        "motion_scaling": {
            "acceleration":
                acceleration_scale,

            "gyro":
                gyro_scale,
        },

        "timing_model": {

            "source":
                "walking.csv",

            "real_intervals":
                real_interval_count,

            "real_stall_count":
                real_stall_count,

            "real_stall_fraction":
                real_stall_fraction,

            "synthetic_stall_count":
                len(stall_positions),

            "normal_interval_source":
                "empirical_walking_distribution",

            "stall_interval_source":
                "empirical_walking_distribution",
        },

        "limitations": [
            "RUNNING values are synthetic.",
            "Real RUNNING hardware data is unavailable.",
            "Synthetic data must not be presented as real experimental evidence.",
            "Real RUNNING data is required for final calibration.",
        ],
    }

    with PROVENANCE_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            provenance,
            handle,
            indent=2,
        )

    print()
    print("=" * 100)
    print("SAFEband STAGE A.5.3 — CALIBRATED SYNTHETIC RUNNING GENERATOR")
    print("=" * 100)
    print()

    print(
        f"Source rows             : {len(source_rows)}"
    )

    print(
        f"Generated rows          : {target_rows}"
    )

    print(
        f"Columns                 : {len(fields)}"
    )

    print(
        f"Acceleration scale      : {acceleration_scale:.2f}x"
    )

    print(
        f"Gyro scale              : {gyro_scale:.2f}x"
    )

    print()

    print(
        f"Real walking intervals  : {real_interval_count}"
    )

    print(
        f"Real walking stalls     : {real_stall_count}"
    )

    print(
        f"Real stall fraction     : "
        f"{real_stall_fraction:.4f}"
    )

    print(
        f"Synthetic stall count   : "
        f"{len(stall_positions)}"
    )

    print()

    print(
        "Timing scheduler        : EMPIRICAL INTERVAL FRACTION"
    )

    print(
        "Activity                : RUNNING"
    )

    print(
        "Provenance              : SYNTHETIC / UNVALIDATED"
    )

    print()

    print(
        f"Dataset saved           : {OUTPUT_FILE}"
    )

    print(
        f"Provenance saved        : {PROVENANCE_FILE}"
    )

    print()

    print(
        "Next: rerun A.5.2 validation."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
