from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "generated"
REPORT_DIR = ROOT / "reports" / "dataset_expansion" / "acceptance"

EXPECTED_STANDARD_COLUMNS = [
    "activity",
    "elapsed_ms",
    "sample_index",
    "heading_deg",
    "roll_deg",
    "pitch_deg",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "temperature_C",
    "humidity_pct",
    "pressure_hPa",
    "gas_kOhm",
    "bme_valid",
    "mic_rms",
    "mic_dc",
    "mic_average_rms",
    "mic_max_rms",
    "mic_blocks",
    "max_ir",
    "max_red",
    "finger_detected",
    "raw_hr",
    "raw_hr_valid",
    "raw_spo2",
    "raw_spo2_valid",
    "final_hr",
    "final_hr_valid",
    "final_spo2",
    "final_spo2_valid",
    "battery_voltage_V",
    "battery_percent",
    "charger_connected",
]

EXPECTED_HAZARD_COLUMNS = EXPECTED_STANDARD_COLUMNS + [
    "hazard_label",
    "hazard_type",
    "hazard_event_id",
    "is_hazard",
]


def classify(name: str) -> str:
    return "hazard" if name.lower().startswith("hazard_") else "standard"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATASET_DIR.glob("*.csv"))
    dataset_results = []
    all_pass = True

    for path in files:
        category = classify(path.name)

        expected = (
            EXPECTED_HAZARD_COLUMNS
            if category == "hazard"
            else EXPECTED_STANDARD_COLUMNS
        )

        try:
            df = pd.read_csv(path)

            actual = list(df.columns)

            missing = [
                column
                for column in expected
                if column not in actual
            ]

            unexpected = [
                column
                for column in actual
                if column not in expected
            ]

            duplicate_columns = (
                df.columns[df.columns.duplicated()]
                .tolist()
            )

            order_match = actual == expected
            row_count_valid = len(df) > 0

            result_pass = (
                len(missing) == 0
                and len(unexpected) == 0
                and not duplicate_columns
                and order_match
                and row_count_valid
            )

            result = {
                "dataset": path.name,
                "category": category,
                "rows": len(df),
                "columns": len(actual),
                "expected_columns": len(expected),
                "missing_columns": missing,
                "unexpected_columns": unexpected,
                "duplicate_columns": duplicate_columns,
                "column_order_match": order_match,
                "row_count_valid": row_count_valid,
                "status": "PASS" if result_pass else "FAIL",
            }

        except Exception as exc:
            result_pass = False

            result = {
                "dataset": path.name,
                "category": category,
                "status": "FAIL",
                "error": str(exc),
            }

        dataset_results.append(result)

        if not result_pass:
            all_pass = False

    report = {
        "acceptance_stage": "2.11.2",
        "stage_name": "Schema Consistency",
        "standard_column_count": len(EXPECTED_STANDARD_COLUMNS),
        "hazard_column_count": len(EXPECTED_HAZARD_COLUMNS),
        "dataset_count": len(files),
        "datasets": dataset_results,
        "overall_status": "PASS" if all_pass else "FAIL",
    }

    output = REPORT_DIR / "schema_consistency_2.11.2.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("SAFEband MILESTONE 2.11.2 — SCHEMA CONSISTENCY")
    print()
    print(f"Datasets checked           : {len(files)}")
    print(
        f"Standard schema            : "
        f"{len(EXPECTED_STANDARD_COLUMNS)} columns"
    )
    print(
        f"Hazard schema              : "
        f"{len(EXPECTED_HAZARD_COLUMNS)} columns"
    )

    passed = sum(
        item["status"] == "PASS"
        for item in dataset_results
    )

    failed = len(dataset_results) - passed

    print(f"Datasets PASS              : {passed}")
    print(f"Datasets FAIL              : {failed}")

    if failed:
        print()
        print("Failures:")

        for item in dataset_results:
            if item["status"] != "PASS":
                print()
                print(f"  {item['dataset']}")

                if "missing_columns" in item:
                    print(
                        f"    Missing   : "
                        f"{item['missing_columns']}"
                    )
                    print(
                        f"    Unexpected: "
                        f"{item['unexpected_columns']}"
                    )
                    print(
                        f"    Duplicates: "
                        f"{item['duplicate_columns']}"
                    )
                    print(
                        f"    Order     : "
                        f"{item['column_order_match']}"
                    )

                if "error" in item:
                    print(
                        f"    Error     : {item['error']}"
                    )

    print()
    print(
        f"Overall status             : "
        f"{report['overall_status']}"
    )
    print(
        f"Report                     : "
        f"{output.relative_to(ROOT)}"
    )

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
