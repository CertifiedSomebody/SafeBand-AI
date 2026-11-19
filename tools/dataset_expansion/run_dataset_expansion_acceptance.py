from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_DIR = ROOT / "reports" / "dataset_expansion" / "acceptance"
OUTPUT_REPORT = (
    ACCEPTANCE_DIR / "dataset_expansion_acceptance_2.11.9.json"
)

STAGES = [
    (
        "2.11.1",
        "Dataset Inventory",
        "acceptance_inventory.py",
        "dataset_inventory_2.11.1.json",
    ),
    (
        "2.11.2",
        "Schema Consistency",
        "acceptance_schema.py",
        "schema_consistency_2.11.2.json",
    ),
    (
        "2.11.3",
        "Provenance Completeness",
        "acceptance_provenance.py",
        "provenance_completeness_2.11.3.json",
    ),
    (
        "2.11.4",
        "Synthetic-Status Integrity",
        "acceptance_synthetic_status.py",
        "synthetic_status_2.11.4.json",
    ),
    (
        "2.11.5",
        "Activity/Profile Coverage",
        "acceptance_activity_profile.py",
        "activity_profile_coverage_2.11.5.json",
    ),
    (
        "2.11.6",
        "Sentinel/Validity Integrity",
        "acceptance_sentinel_validity.py",
        "sentinel_validity_2.11.6.json",
    ),
    (
        "2.11.7",
        "Timing Integrity",
        "acceptance_timing_integrity.py",
        "timing_integrity_2.11.7.json",
    ),
    (
        "2.11.8",
        "Hazard/Hard-Negative Coverage",
        "acceptance_hazard_hard_negative.py",
        "hazard_hard_negative_coverage_2.11.8.json",
    ),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def run_stage(
    stage: str,
    name: str,
    script_name: str,
    report_name: str,
) -> dict:
    script_path = ROOT / "tools" / "dataset_expansion" / script_name
    report_path = ACCEPTANCE_DIR / report_name
    result = {
        "stage": stage,
        "name": name,
        "report": report_path.relative_to(ROOT).as_posix(),
        "status": "FAIL",
    }

    try:
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        result["error"] = f"Could not run stage script: {exc}"
        return result

    result["exit_code"] = process.returncode
    if process.stdout:
        printable_stdout = process.stdout.encode(
            "ascii",
            "backslashreplace",
        ).decode("ascii")
        print(printable_stdout, end="")
    if process.stderr:
        printable_stderr = process.stderr.encode(
            "ascii",
            "backslashreplace",
        ).decode("ascii")
        print(printable_stderr, file=sys.stderr, end="")

    try:
        stage_report = load_json(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"Stage report could not be read: {exc}"
        return result

    report_status = stage_report.get(
        "overall_status",
        stage_report.get("status"),
    )
    result["report_status"] = report_status
    result["status"] = (
        "PASS"
        if process.returncode == 0 and report_status == "PASS"
        else "FAIL"
    )
    return result


def main() -> int:
    ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
    validation_steps = [
        run_stage(*stage)
        for stage in STAGES
    ]
    overall_status = (
        "PASS"
        if all(step["status"] == "PASS" for step in validation_steps)
        else "FAIL"
    )
    steps = validation_steps + [
        {
            "stage": "2.11.9",
            "name": "Final Dataset Expansion Acceptance",
            "report": OUTPUT_REPORT.relative_to(ROOT).as_posix(),
            "status": overall_status,
        }
    ]
    report = {
        "acceptance_stage": "2.11.9",
        "stage_name": "Final Dataset Expansion Acceptance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step_count": len(steps),
        "validation_stage_count": len(validation_steps),
        "steps_passed": sum(
            step["status"] == "PASS" for step in steps
        ),
        "steps_failed": sum(
            step["status"] == "FAIL" for step in steps
        ),
        "overall_status": overall_status,
        "steps": steps,
    }
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("SAFEband MILESTONE 2.11.9 — FINAL DATASET ACCEPTANCE")
    print(f"Stages passed : {report['steps_passed']}/{report['step_count']}")
    print(f"Overall status: {overall_status}")
    print(f"Report        : {OUTPUT_REPORT.relative_to(ROOT)}")
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
