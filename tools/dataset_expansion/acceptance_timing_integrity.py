from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "generated"
REPORT_DIR = ROOT / "reports" / "dataset_expansion"
ACCEPTANCE_DIR = REPORT_DIR / "acceptance"
VALIDATOR = (
    ROOT
    / "tools"
    / "dataset_expansion"
    / "timing_profiles"
    / "validate_timing_profile.py"
)
OUTPUT_REPORT = (
    ACCEPTANCE_DIR / "timing_integrity_2.11.7.json"
)

EXPECTED_DATASETS = {
    "timing_bursty_001.csv",
    "timing_jittered_001.csv",
    "timing_normal_001.csv",
    "timing_slow_001.csv",
    "timing_stall_heavy_001.csv",
    "timing_stall_light_001.csv",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if not path.is_absolute():
        path = ROOT / path
    return path


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def find_source(dataset_path: Path, provenance: dict) -> Path:
    provenance_sources = [
        provenance.get("source"),
        provenance.get("source_path"),
    ]

    for value in provenance_sources:
        if isinstance(value, str) and value.strip():
            source_path = resolve_path(value)
            if source_path.is_file():
                return source_path

    validation_reports = sorted(
        REPORT_DIR.rglob(
            f"{dataset_path.stem}_validation.json"
        )
    )
    for report_path in validation_reports:
        report = load_json(report_path)
        value = report.get("source")
        if isinstance(value, str) and value.strip():
            source_path = resolve_path(value)
            if source_path.is_file():
                return source_path

    raise FileNotFoundError(
        f"No existing source dataset found for {dataset_path.name}"
    )


def validate_dataset(dataset_path: Path) -> dict:
    result = {
        "dataset": dataset_path.name,
        "status": "FAIL",
        "validator_report": None,
    }
    provenance_path = Path(f"{dataset_path}.provenance.json")
    if not provenance_path.is_file():
        result["error"] = "Native provenance file is missing"
        return result

    try:
        provenance = load_json(provenance_path)
        source_path = find_source(dataset_path, provenance)
    except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
        result["error"] = str(exc)
        return result

    validation_report = REPORT_DIR / (
        f"{dataset_path.stem}_validation.json"
    )
    result["source"] = relative_path(source_path)
    result["validator_report"] = (
        relative_path(validation_report)
    )

    try:
        process = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--source",
                str(source_path),
                "--augmented",
                str(dataset_path),
                "--output",
                str(validation_report),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        result["error"] = f"Could not run timing validator: {exc}"
        return result

    result["validator_exit_code"] = process.returncode

    try:
        validator_report = load_json(validation_report)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"Invalid timing validation report: {exc}"
        return result

    result["checks"] = validator_report.get("checks", {})
    result["timing"] = validator_report.get("timing", {})
    passed = (
        process.returncode == 0
        and validator_report.get("status") == "PASS"
    )
    result["status"] = "PASS" if passed else "FAIL"
    if not passed:
        result["validator_output"] = (
            process.stdout + process.stderr
        ).encode("ascii", "backslashreplace").decode("ascii")
    print(f"  {dataset_path.name}: {result['status']}")
    return result


def main() -> int:
    ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
    datasets = sorted(DATASET_DIR.glob("timing_*.csv"))
    results = [validate_dataset(path) for path in datasets]

    actual_datasets = {path.name for path in datasets}
    checks = {
        "expected_timing_datasets": actual_datasets == EXPECTED_DATASETS,
        "all_timing_validators_pass": (
            len(results) == len(EXPECTED_DATASETS)
            and all(result["status"] == "PASS" for result in results)
        ),
    }
    report = {
        "acceptance_stage": "2.11.7",
        "stage_name": "Timing Integrity",
        "expected_dataset_count": len(EXPECTED_DATASETS),
        "dataset_count": len(datasets),
        "missing_datasets": sorted(EXPECTED_DATASETS - actual_datasets),
        "unexpected_datasets": sorted(actual_datasets - EXPECTED_DATASETS),
        "checks": checks,
        "overall_status": (
            "PASS" if all(checks.values()) else "FAIL"
        ),
        "datasets": results,
    }
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("SAFEband MILESTONE 2.11.7 — TIMING INTEGRITY")
    print(f"Timing datasets checked : {len(datasets)}")
    print(f"Expected profiles       : {'PASS' if checks['expected_timing_datasets'] else 'FAIL'}")
    print(f"Timing validators       : {'PASS' if checks['all_timing_validators_pass'] else 'FAIL'}")
    print(f"Overall status          : {report['overall_status']}")
    print(f"Report                  : {OUTPUT_REPORT.relative_to(ROOT)}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
