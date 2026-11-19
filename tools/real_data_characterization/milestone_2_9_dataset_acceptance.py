import csv
import json
import math
import statistics
from pathlib import Path


REAL_DIR = Path("datasets/real sensor data")
GENERATED_DIR = Path("datasets/generated")
REPORT_DIR = Path("reports/real_data_characterization")

FILES = [
    REAL_DIR / "stationary.csv",
    REAL_DIR / "stationary_2.csv",
    REAL_DIR / "walking.csv",
    GENERATED_DIR / "running_synthetic_dataset.csv",
]

EXPECTED_COLUMNS = 36

ACTIVITY_EXPECTATIONS = {
    "stationary.csv": "STATIONARY",
    "stationary_2.csv": "STATIONARY",
    "walking.csv": "WALKING",
    "running_synthetic_dataset.csv": "RUNNING",
}


def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        rows = list(reader)
        fields = reader.fieldnames

    if not fields:
        raise ValueError(
            f"No header found in {path}"
        )

    return rows, fields


def number(value):

    try:

        result = float(value)

        if math.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return None


def find_column(fields, candidates):

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


def percentile(values, p):

    if not values:
        return None

    values = sorted(values)

    position = (
        (len(values) - 1)
        * p
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


def timing_report(rows, elapsed_column):

    elapsed = []

    for row in rows:

        value = number(
            row.get(elapsed_column)
        )

        if value is not None:
            elapsed.append(value)

    if len(elapsed) < 2:
        return {
            "interval_count": 0,
            "normal_count": 0,
            "stall_count": 0,
            "stall_fraction": 0.0,
            "negative_intervals": 0,
            "normal_mean_ms": None,
            "normal_median_ms": None,
            "normal_p95_ms": None,
            "stall_mean_ms": None,
            "stall_median_ms": None,
            "stall_p95_ms": None,
            "total_duration_ms": None,
        }

    intervals = [
        elapsed[i] - elapsed[i - 1]
        for i in range(1, len(elapsed))
    ]

    normal = [
        value
        for value in intervals
        if 0 <= value < 1000
    ]

    stalls = [
        value
        for value in intervals
        if value >= 1000
    ]

    negative = [
        value
        for value in intervals
        if value < 0
    ]

    return {
        "interval_count": len(intervals),
        "normal_count": len(normal),
        "stall_count": len(stalls),
        "stall_fraction": (
            len(stalls)
            / len(intervals)
            if intervals
            else 0.0
        ),
        "negative_intervals": len(negative),
        "normal_mean_ms": (
            statistics.mean(normal)
            if normal
            else None
        ),
        "normal_median_ms": (
            statistics.median(normal)
            if normal
            else None
        ),
        "normal_p95_ms": (
            percentile(normal, 0.95)
            if normal
            else None
        ),
        "stall_mean_ms": (
            statistics.mean(stalls)
            if stalls
            else None
        ),
        "stall_median_ms": (
            statistics.median(stalls)
            if stalls
            else None
        ),
        "stall_p95_ms": (
            percentile(stalls, 0.95)
            if stalls
            else None
        ),
        "total_duration_ms": (
            elapsed[-1] - elapsed[0]
        ),
    }


def validity_report(rows, fields):

    result = {}

    candidates = [
        "bme_valid",
        "finger_detected",
        "raw_hr_valid",
        "raw_spo2_valid",
        "final_hr_valid",
        "final_spo2_valid",
        "charger_connected",
    ]

    for candidate in candidates:

        column = find_column(
            fields,
            [candidate],
        )

        if column is None:
            continue

        valid = 0
        invalid = 0
        unknown = 0

        for row in rows:

            value = row.get(column)

            if value in (
                "1",
                "1.0",
                "true",
                "True",
            ):

                valid += 1

            elif value in (
                "0",
                "0.0",
                "false",
                "False",
            ):

                invalid += 1

            else:

                unknown += 1

        total = len(rows)

        result[column] = {
            "valid": valid,
            "invalid": invalid,
            "unknown": unknown,
            "valid_fraction": (
                valid / total
                if total
                else 0.0
            ),
        }

    return result


def sentinel_report(rows, fields):

    result = {}

    for field in fields:

        count = 0

        for row in rows:

            value = row.get(field)

            if value is None:
                continue

            if str(value).strip() == "-999":
                count += 1

        if count:
            result[field] = count

    return result


def numeric_sanity(rows, fields):

    result = {}

    for field in fields:

        values = []

        for row in rows:

            value = number(
                row.get(field)
            )

            if value is not None:
                values.append(value)

        if not values:
            continue

        non_finite = 0

        for row in rows:

            raw = row.get(field)

            if raw is None:
                continue

            try:

                value = float(raw)

                if not math.isfinite(value):
                    non_finite += 1

            except (TypeError, ValueError):
                pass

        result[field] = {
            "numeric_count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.mean(values),
            "non_finite": non_finite,
        }

    return result


def duplicate_report(rows):

    serialized = []

    for row in rows:

        serialized.append(
            tuple(
                sorted(row.items())
            )
        )

    counts = {}

    for item in serialized:
        counts[item] = (
            counts.get(item, 0)
            + 1
        )

    duplicate_rows = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    duplicate_groups = sum(
        1
        for count in counts.values()
        if count > 1
    )

    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups,
    }


def activity_report(rows, fields):

    column = find_column(
        fields,
        ["activity"],
    )

    if column is None:
        return {
            "column": None,
            "values": {},
        }

    counts = {}

    for row in rows:

        value = str(
            row.get(column, "")
        ).strip()

        counts[value] = (
            counts.get(value, 0)
            + 1
        )

    return {
        "column": column,
        "values": counts,
    }


def audit_file(path):

    rows, fields = load_csv(path)

    name = path.name

    checks = {}

    checks["file_exists"] = True

    checks["row_count_positive"] = (
        len(rows) > 0
    )

    checks["column_count"] = (
        len(fields)
    )

    checks["schema_36_columns"] = (
        len(fields)
        == EXPECTED_COLUMNS
    )

    activity = activity_report(
        rows,
        fields,
    )

    expected_activity = (
        ACTIVITY_EXPECTATIONS.get(
            name
        )
    )

    actual_activities = (
        activity["values"]
    )

    checks["activity_expected"] = (
        expected_activity is not None
    )

    checks["activity_label_correct"] = (
        len(actual_activities) == 1
        and expected_activity in actual_activities
    )

    elapsed_column = find_column(
        fields,
        ["elapsed_ms"],
    )

    if elapsed_column is None:

        timing = {
            "interval_count": 0,
            "normal_count": 0,
            "stall_count": 0,
            "stall_fraction": 0.0,
            "negative_intervals": 0,
        }

        checks["elapsed_column_present"] = False
        checks["timing_valid"] = False

    else:

        timing = timing_report(
            rows,
            elapsed_column,
        )

        checks["elapsed_column_present"] = True

        checks["timing_valid"] = (
            timing["negative_intervals"]
            == 0
            and timing["interval_count"] > 0
        )

    sentinel = sentinel_report(
        rows,
        fields,
    )

    numeric = numeric_sanity(
        rows,
        fields,
    )

    duplicates = duplicate_report(
        rows
    )

    validity = validity_report(
        rows,
        fields,
    )

    checks["no_duplicate_rows"] = (
        duplicates["duplicate_rows"] == 0
    )

    checks["no_nonfinite_numeric_values"] = all(
        value["non_finite"] == 0
        for value in numeric.values()
    )

    return {
        "file": name,
        "path": str(path),
        "rows": len(rows),
        "columns": fields,
        "checks": checks,
        "activity": activity,
        "timing": timing,
        "sentinels": sentinel,
        "numeric": numeric,
        "duplicates": duplicates,
        "validity": validity,
    }


def compare_schemas(reports):

    if not reports:
        return {
            "pass": False,
            "reason": "No datasets found.",
        }

    reference = reports[0]["columns"]

    differences = {}

    for report in reports[1:]:

        current = report["columns"]

        if current != reference:

            differences[
                report["file"]
            ] = {
                "missing": [
                    field
                    for field in reference
                    if field not in current
                ],
                "extra": [
                    field
                    for field in current
                    if field not in reference
                ],
            }

    return {
        "pass": not differences,
        "reference_columns": reference,
        "differences": differences,
    }


def compare_timing(reports):

    real_reports = [
        report
        for report in reports
        if report["file"] !=
        "running_synthetic_dataset.csv"
    ]

    synthetic = next(
        (
            report
            for report in reports
            if report["file"]
            == "running_synthetic_dataset.csv"
        ),
        None,
    )

    walking = next(
        (
            report
            for report in reports
            if report["file"]
            == "walking.csv"
        ),
        None,
    )

    if synthetic is None or walking is None:

        return {
            "pass": False,
            "reason":
                "Missing walking or synthetic dataset.",
        }

    synthetic_fraction = (
        synthetic["timing"]
        ["stall_fraction"]
    )

    walking_fraction = (
        walking["timing"]
        ["stall_fraction"]
    )

    fraction_difference = abs(
        synthetic_fraction
        - walking_fraction
    )

    # Synthetic timing is accepted if its stall
    # fraction remains within 2 percentage points
    # of measured walking behavior.
    fraction_pass = (
        fraction_difference
        <= 0.02
    )

    normal_median = (
        walking["timing"]
        ["normal_median_ms"]
    )

    synthetic_normal_median = (
        synthetic["timing"]
        ["normal_median_ms"]
    )

    median_difference = (
        abs(
            synthetic_normal_median
            - normal_median
        )
        if (
            normal_median is not None
            and synthetic_normal_median is not None
        )
        else None
    )

    median_pass = (
        median_difference is not None
        and median_difference <= 50
    )

    return {
        "pass":
            fraction_pass
            and median_pass,

        "walking_stall_fraction":
            walking_fraction,

        "synthetic_stall_fraction":
            synthetic_fraction,

        "stall_fraction_difference":
            fraction_difference,

        "stall_fraction_pass":
            fraction_pass,

        "walking_normal_median_ms":
            normal_median,

        "synthetic_normal_median_ms":
            synthetic_normal_median,

        "normal_median_difference_ms":
            median_difference,

        "normal_median_pass":
            median_pass,
    }


def main():

    reports = []

    missing_files = []

    for path in FILES:

        if not path.exists():

            missing_files.append(
                str(path)
            )
            continue

        reports.append(
            audit_file(path)
        )

    if missing_files:

        print()
        print(
            "Missing datasets:"
        )

        for path in missing_files:
            print(
                f"  - {path}"
            )

        raise SystemExit(1)

    schema = compare_schemas(
        reports
    )

    timing = compare_timing(
        reports
    )

    all_file_checks_pass = all(
        all(
            report["checks"].values()
        )
        for report in reports
    )

    synthetic = next(
        report
        for report in reports
        if report["file"]
        == "running_synthetic_dataset.csv"
    )

    provenance_path = (
        GENERATED_DIR
        / "running_synthetic_dataset.provenance.json"
    )

    provenance_pass = False

    if provenance_path.exists():

        try:

            with provenance_path.open(
                "r",
                encoding="utf-8",
            ) as handle:

                provenance = json.load(
                    handle
                )

            provenance_pass = (
                provenance.get(
                    "status"
                )
                == "SYNTHETIC_DEVELOPMENT_FIXTURE"
                and provenance.get(
                    "validated_against_real_running"
                )
                is False
            )

        except Exception:
            provenance_pass = False

    else:

        provenance = None

    overall_pass = (
        all_file_checks_pass
        and schema["pass"]
        and timing["pass"]
        and provenance_pass
    )

    report = {
        "stage":
            "MILESTONE_2.9_DATASET_ACCEPTANCE",

        "status":
            "PASS"
            if overall_pass
            else "FAIL",

        "datasets":
            reports,

        "schema_comparison":
            schema,

        "synthetic_timing_comparison":
            timing,

        "synthetic_provenance_check":
            provenance_pass,

        "limitations": [
            "Synthetic RUNNING remains unvalidated against real RUNNING hardware data.",
            "Synthetic RUNNING must not be presented as empirical RUNNING evidence.",
            "Real RUNNING data will be used later for calibration and final validation.",
        ],
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        REPORT_DIR
        / "milestone_2_9_dataset_acceptance.json"
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
    print("=" * 100)
    print(
        "SAFEband MILESTONE 2.9 — DATASET ACCEPTANCE AUDIT"
    )
    print("=" * 100)
    print()

    for report_item in reports:

        print(
            f"{report_item['file']:<36}"
            f" rows={report_item['rows']:<5}"
            f" cols={len(report_item['columns']):<3}"
        )

        print(
            f"  activity : "
            f"{report_item['activity']['values']}"
        )

        print(
            f"  stalls   : "
            f"{report_item['timing']['stall_count']}"
            f" / "
            f"{report_item['timing']['interval_count']}"
            f" = "
            f"{report_item['timing']['stall_fraction']:.4f}"
        )

        print(
            f"  duplicate rows : "
            f"{report_item['duplicates']['duplicate_rows']}"
        )

        print()

    print(
        "Schema consistency       : "
        + (
            "PASS"
            if schema["pass"]
            else "FAIL"
        )
    )

    print(
        "Per-file integrity       : "
        + (
            "PASS"
            if all_file_checks_pass
            else "FAIL"
        )
    )

    print(
        "Timing consistency       : "
        + (
            "PASS"
            if timing["pass"]
            else "FAIL"
        )
    )

    print(
        "Synthetic provenance     : "
        + (
            "PASS"
            if provenance_pass
            else "FAIL"
        )
    )

    print()

    print(
        "Walking stall fraction   : "
        f"{timing['walking_stall_fraction']:.4f}"
    )

    print(
        "Synthetic stall fraction : "
        f"{timing['synthetic_stall_fraction']:.4f}"
    )

    print(
        "Difference               : "
        f"{timing['stall_fraction_difference']:.4f}"
    )

    print()

    print(
        "MILESTONE 2.9 STATUS     : "
        + (
            "PASS"
            if overall_pass
            else "FAIL"
        )
    )

    print()

    print(
        f"Report saved: {output}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
