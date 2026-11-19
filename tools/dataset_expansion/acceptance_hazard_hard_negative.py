from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "generated"
SOURCE_DIR = ROOT / "datasets" / "real sensor data"
REPORT_DIR = ROOT / "reports" / "dataset_expansion"
ACCEPTANCE_DIR = REPORT_DIR / "acceptance"
HARD_NEGATIVE_VALIDATOR = (
    ROOT
    / "tools"
    / "dataset_expansion"
    / "hard_negatives"
    / "validate_hard_negative.py"
)
HAZARD_VALIDATOR = (
    ROOT
    / "tools"
    / "dataset_expansion"
    / "hazard_scenarios"
    / "validate_hazard_scenario.py"
)
OUTPUT_REPORT = (
    ACCEPTANCE_DIR / "hazard_hard_negative_coverage_2.11.8.json"
)

EXPECTED_HARD_NEGATIVE_PROFILES = {
    "ABRUPT_ACCELERATION",
    "ABRUPT_DECELERATION",
    "SHARP_TURN",
    "MOTION_SPIKE",
    "STOP_START",
    "PPG_ARTIFACT",
    "TIMING_STALL_NO_HAZARD",
    "NEAR_FALL_RECOVERY",
}
EXPECTED_HAZARD_TYPES = {
    "FALL_FORWARD",
    "FALL_BACKWARD",
    "FALL_SIDE",
    "TRIP_RECOVERY",
    "SUDDEN_COLLAPSE",
    "HIGH_IMPACT",
    "LOSS_OF_BALANCE",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if not path.is_absolute():
        path = ROOT / path
    return path


def preserved_source_signature(
    source: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    hazard: bool,
) -> bool:
    if len(source) != len(generated):
        return False
    if hazard:
        if not set(source.columns).issubset(generated.columns):
            return False
    elif list(source.columns) != list(generated.columns):
        return False

    for column in ("sample_index", "activity"):
        if column in source.columns and not source[column].equals(
            generated[column]
        ):
            return False

    validity_columns = [
        column
        for column in source.columns
        if "valid" in column.lower()
        or column.lower() == "finger_detected"
    ]
    return all(
        source[column].equals(generated[column])
        for column in validity_columns
    )


def find_source(
    dataset_path: Path,
    provenance: dict,
    *,
    hazard: bool,
) -> Path:
    for key in ("source", "source_path", "source_dataset"):
        value = provenance.get(key)
        if isinstance(value, str) and value.strip():
            source_path = resolve_path(value)
            if source_path.is_file():
                return source_path

    generated = pd.read_csv(dataset_path)
    matching_sources = []
    for source_path in sorted(SOURCE_DIR.glob("*.csv")):
        source = pd.read_csv(source_path)
        if preserved_source_signature(
            source,
            generated,
            hazard=hazard,
        ):
            matching_sources.append(source_path)

    if len(matching_sources) == 1:
        return matching_sources[0]
    if not matching_sources:
        raise FileNotFoundError(
            f"No source dataset matches preserved fields for "
            f"{dataset_path.name}"
        )
    raise ValueError(
        f"Source dataset is ambiguous for {dataset_path.name}: "
        + ", ".join(path.name for path in matching_sources)
    )


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def validate_dataset(
    dataset_path: Path,
    *,
    hazard: bool,
) -> dict:
    result = {
        "dataset": dataset_path.name,
        "kind": "hazard" if hazard else "hard_negative",
        "status": "FAIL",
        "validator_report": None,
    }
    provenance_path = Path(f"{dataset_path}.provenance.json")
    if not provenance_path.is_file():
        result["error"] = "Native provenance file is missing"
        return result

    try:
        provenance = load_json(provenance_path)
        source_path = find_source(
            dataset_path,
            provenance,
            hazard=hazard,
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        result["error"] = str(exc)
        return result

    profile = provenance.get("profile")
    if isinstance(profile, str):
        result["profile"] = profile.strip().upper()

    report_dir = (
        REPORT_DIR / ("hazard_scenarios" if hazard else "hard_negatives")
    )
    validation_report = report_dir / (
        f"{dataset_path.stem}_validation.json"
    )
    result["source"] = relative_path(source_path)
    result["validator_report"] = relative_path(validation_report)
    validator = HAZARD_VALIDATOR if hazard else HARD_NEGATIVE_VALIDATOR
    command = [
        sys.executable,
        str(validator),
        "--source",
        str(source_path),
        "--augmented",
        str(dataset_path),
        "--output",
        str(validation_report),
    ]

    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        result["error"] = f"Could not run dataset validator: {exc}"
        return result

    result["validator_exit_code"] = process.returncode

    try:
        validator_report = load_json(validation_report)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"Invalid dataset validation report: {exc}"
        return result

    passed = (
        validator_report.get("overall")
        if not hazard
        else validator_report.get("status") == "PASS"
    )
    result["checks"] = (
        validator_report.get("checks")
        if hazard
        else {
            key: value
            for key, value in validator_report.items()
            if isinstance(value, bool)
        }
    )
    result["status"] = (
        "PASS"
        if process.returncode == 0 and passed
        else "FAIL"
    )
    if result["status"] == "FAIL":
        result["validator_output"] = (
            process.stdout + process.stderr
        ).encode("ascii", "backslashreplace").decode("ascii")
    print(f"  {dataset_path.name}: {result['status']}")
    return result


def main() -> int:
    ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
    hard_negative_paths = sorted(DATASET_DIR.glob("hardneg_*.csv"))
    hazard_paths = sorted(DATASET_DIR.glob("hazard_*.csv"))
    results = [
        validate_dataset(path, hazard=False)
        for path in hard_negative_paths
    ] + [
        validate_dataset(path, hazard=True)
        for path in hazard_paths
    ]

    hard_negative_profiles = {
        result["profile"]
        for result in results
        if result["kind"] == "hard_negative"
        and result.get("profile")
    }
    hazard_profiles = {
        result["profile"]
        for result in results
        if result["kind"] == "hazard"
        and result.get("profile")
    }
    checks = {
        "hard_negative_profile_coverage": (
            hard_negative_profiles == EXPECTED_HARD_NEGATIVE_PROFILES
        ),
        "hazard_type_coverage": (
            hazard_profiles == EXPECTED_HAZARD_TYPES
        ),
        "all_dataset_validators_pass": (
            len(hard_negative_paths) == len(EXPECTED_HARD_NEGATIVE_PROFILES)
            and len(hazard_paths) == len(EXPECTED_HAZARD_TYPES)
            and all(result["status"] == "PASS" for result in results)
        ),
    }
    report = {
        "acceptance_stage": "2.11.8",
        "stage_name": "Hazard/Hard-Negative Coverage",
        "hard_negative_count": len(hard_negative_paths),
        "hazard_count": len(hazard_paths),
        "expected_hard_negative_profiles": sorted(
            EXPECTED_HARD_NEGATIVE_PROFILES
        ),
        "found_hard_negative_profiles": sorted(hard_negative_profiles),
        "missing_hard_negative_profiles": sorted(
            EXPECTED_HARD_NEGATIVE_PROFILES - hard_negative_profiles
        ),
        "unexpected_hard_negative_profiles": sorted(
            hard_negative_profiles - EXPECTED_HARD_NEGATIVE_PROFILES
        ),
        "expected_hazard_types": sorted(EXPECTED_HAZARD_TYPES),
        "found_hazard_types": sorted(hazard_profiles),
        "missing_hazard_types": sorted(
            EXPECTED_HAZARD_TYPES - hazard_profiles
        ),
        "unexpected_hazard_types": sorted(
            hazard_profiles - EXPECTED_HAZARD_TYPES
        ),
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

    print("SAFEband MILESTONE 2.11.8 — HAZARD/HARD-NEGATIVE COVERAGE")
    print(f"Hard-negative datasets : {len(hard_negative_paths)}")
    print(f"Hazard datasets        : {len(hazard_paths)}")
    print(f"Profile coverage       : {'PASS' if checks['hard_negative_profile_coverage'] and checks['hazard_type_coverage'] else 'FAIL'}")
    print(f"Dataset validators     : {'PASS' if checks['all_dataset_validators_pass'] else 'FAIL'}")
    print(f"Overall status         : {report['overall_status']}")
    print(f"Report                 : {OUTPUT_REPORT.relative_to(ROOT)}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
