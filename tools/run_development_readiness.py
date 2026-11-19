from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "development_readiness_2.28.json"
PYTHON = sys.executable


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def print_process_output(output: str) -> None:
    if output.strip():
        safe_output = output.strip().encode(
            "ascii",
            "backslashreplace",
        ).decode("ascii")
        sys.stdout.write(safe_output + "\n")


def run_step(
    name: str,
    command: list[str],
    expected_exit_codes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    expected = process.returncode in expected_exit_codes
    result = {
        "name": name,
        "command": command,
        "exit_code": process.returncode,
        "expected_exit_codes": list(expected_exit_codes),
        "status": "PASS" if expected else "FAIL",
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }
    if not expected:
        print(f"FAIL: {name} (exit {process.returncode})")
        print_process_output(process.stdout)
        print_process_output(process.stderr)
    else:
        print(f"PASS: {name}")
        print_process_output(process.stdout)
        print_process_output(process.stderr)
    return result


def test_count(output: str) -> int | None:
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    return int(match.group(1)) if match else None


def dotnet_test_count(output: str) -> int | None:
    match = re.search(r"Total:\s*(\d+)", output)
    return int(match.group(1)) if match else None


def build_report(
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    acceptance = load_json(
        ROOT
        / "reports"
        / "dataset_expansion"
        / "acceptance"
        / "dataset_expansion_acceptance_2.11.9.json"
    )
    lineage = load_json(ROOT / "datasets" / "training" / "lineage_manifest.json")
    training = load_json(
        ROOT
        / "reports"
        / "dataset_training"
        / "training_dataset_acceptance_2.12.json"
    )
    features = load_json(
        ROOT
        / "reports"
        / "dataset_training"
        / "development_feature_matrix_2.13.json"
    )
    replay = load_json(
        ROOT
        / "reports"
        / "runtime_replay"
        / "development_replay_2.22.json"
    )
    step_by_name = {step["name"]: step for step in steps}
    expected_lineage_gate = (
        training.get("status") == "BLOCKED"
        and training.get("classification")
        == "DELIBERATE_DATA_INTEGRITY_GATE"
        and training.get("checks", {}).get("all_dataset_lineage_resolved")
        and not training.get("checks", {}).get("minimum_independent_source_sessions")
        and not training.get("train_validation_test_created")
    )
    checks_passed = (
        all(step["status"] == "PASS" for step in steps)
        and acceptance.get("overall_status") == "PASS"
        and expected_lineage_gate
        and features.get("status") == "DEVELOPMENT ONLY"
        and replay.get("status") == "DEVELOPMENT ONLY"
    )
    artifact_suffixes = {".joblib", ".pt", ".keras", ".onnx", ".tflite"}
    model_artifacts = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "models").rglob("*")
        if path.is_file() and path.suffix.lower() in artifact_suffixes
    )
    python_tests = step_by_name["feature engineering unit tests"]
    dotnet_tests = step_by_name[".NET AI/core tests"]
    report_status = (
        "DEVELOPMENT_PACKAGE_COMPLETE_WITH_BLOCKED_VALIDATION_GATES"
        if checks_passed
        else "INCOMPLETE_VALIDATION_FAILURE"
    )
    return {
        "milestone": "2.28",
        "name": "Development Readiness and Final Status",
        "status": report_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "submission_scope": (
            "Development-only implementation and reproducible evidence; "
            "not a validated fall-detection product."
        ),
        "dataset_acceptance": {
            "milestone": "2.11.9",
            "status": acceptance.get("overall_status"),
            "accepted_dataset_count": lineage.get("accepted_dataset_count"),
            "resolved_dataset_count": lineage.get("resolved_dataset_count"),
        },
        "source_lineage_gate": {
            "milestone": "2.12",
            "status": training.get("status"),
            "independent_source_session_count": training.get(
                "independent_source_session_count"
            ),
            "required_independent_source_session_count": training.get(
                "required_independent_source_session_count"
            ),
            "source_sessions": training.get("source_sessions"),
            "train_validation_test_created": False,
            "leakage_rule_weakened": False,
            "reason": training.get("reason"),
        },
        "stage_statuses": [
            {
                "milestones": "2.11.1-2.11.9",
                "work": "Dataset expansion acceptance",
                "status": acceptance.get("overall_status"),
                "evidence": "Freshly rerun inventory, schema, provenance, validity, timing, hazard, and coverage checks.",
            },
            {
                "milestones": "2.12",
                "work": "Source-session lineage audit and split eligibility",
                "status": "BLOCKED",
                "evidence": "42 accepted derivatives resolve to one source session; no partitions created.",
            },
            {
                "milestones": "2.13",
                "work": "Development feature engineering",
                "status": features.get("status"),
                "evidence": (
                    f"{features.get('feature_row_count')} rows, "
                    f"{features.get('feature_count')} deterministic causal features, "
                    "sensor availability masks; no scaling."
                ),
            },
            {
                "milestones": "2.14-2.16",
                "work": "Final partitions, held-out model metrics, and generalization",
                "status": "BLOCKED",
                "evidence": "Requires independent source sessions; no test metrics generated.",
            },
            {
                "milestones": "2.17",
                "work": "Physiological error/clinical validation",
                "status": "BLOCKED",
                "evidence": "No aligned independent reference ground truth.",
            },
            {
                "milestones": "2.18-2.20",
                "work": "Model release contract, runtime input quality, and temporal fall logic",
                "status": "DEVELOPMENT ONLY",
                "evidence": (
                    ".NET version/lineage gate and temporal state machine are implemented "
                    "and tested; no trained model is approved by this work."
                ),
            },
            {
                "milestones": "2.21",
                "work": "Validated model inference integration",
                "status": "BLOCKED",
                "evidence": (
                    "Existing model artifacts were not loaded or scored; compatibility "
                    "with this feature schema and independent training lineage is unverified."
                ),
            },
            {
                "milestones": "2.22",
                "work": "Offline feature-pipeline replay",
                "status": replay.get("status"),
                "evidence": (
                    f"{replay.get('dataset_count')} datasets / {replay.get('row_count')} rows; "
                    "no model inference or label-based prediction."
                ),
            },
            {
                "milestones": "2.23",
                "work": "Software robustness tests",
                "status": (
                    "DEVELOPMENT ONLY"
                    if python_tests["status"] == "PASS"
                    and dotnet_tests["status"] == "PASS"
                    else "FAIL"
                ),
                "evidence": (
                    f"{test_count(python_tests['stdout'] + python_tests['stderr']) or 0} "
                    f"Python feature tests and "
                    f"{dotnet_test_count(dotnet_tests['stdout'] + dotnet_tests['stderr']) or 0} .NET tests."
                ),
            },
            {
                "milestones": "2.24",
                "work": "Host feature-extraction performance",
                "status": "DEVELOPMENT ONLY",
                "evidence": (
                    f"{replay.get('aggregate_rows_per_second')} rows/s on the current "
                    "development host; not wearable or end-to-end inference latency."
                ),
            },
            {
                "milestones": "2.25",
                "work": "Wearable/hardware and field verification",
                "status": "BLOCKED",
                "evidence": "Hardware is unavailable; no live-device or field test claimed.",
            },
            {
                "milestones": "2.26-2.28",
                "work": "Reproducibility, limitations, and final status package",
                "status": (
                    "DEVELOPMENT ONLY"
                    if checks_passed
                    else "INCOMPLETE"
                ),
                "evidence": "This report preserves blocked gates and distinguishes development from validation.",
            },
        ],
        "model_artifacts": {
            "discovered_count": len(model_artifacts),
            "discovered_paths": model_artifacts,
            "loaded_by_this_work": False,
            "compatibility_with_development_feature_schema": "NOT_ASSESSED",
            "production_approval": False,
        },
        "performance_claims": {
            "feature_extraction_rows_per_second_current_host": replay.get(
                "aggregate_rows_per_second"
            ),
            "wearable_latency_measured": False,
            "fall_detection_accuracy_claimed": False,
            "clinical_claimed": False,
        },
        "validation_commands": steps,
        "overall_claim": (
            "Development components and their software tests are complete. "
            "Independent model evaluation, clinical validation, and hardware "
            "verification remain blocked."
        ),
        "next_required_evidence": [
            "Collect at least three independent real source sessions and rerun the group-level split gate.",
            "Establish aligned reference ground truth before physiological error claims.",
            "Audit any pre-existing model artifact for exact feature schema and independent training lineage before deployment consideration.",
            "Restore hardware for live-device latency, sensor-quality, and field tests.",
        ],
        "all_automation_steps_succeeded": checks_passed,
    }


def main() -> int:
    steps = [
        run_step(
            "2.11.1-2.11.9 dataset acceptance",
            [
                PYTHON,
                "tools/dataset_expansion/run_dataset_expansion_acceptance.py",
            ],
        ),
        run_step(
            "2.12 lineage audit",
            [PYTHON, "tools/dataset_training/build_training_dataset.py"],
            expected_exit_codes=(2,),
        ),
        run_step(
            "2.12 partition validation gate",
            [PYTHON, "tools/dataset_training/validate_training_dataset.py"],
            expected_exit_codes=(2,),
        ),
        run_step(
            "feature engineering unit tests",
            [
                PYTHON,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tools/feature_engineering/tests",
                "-v",
            ],
        ),
        run_step(
            "development feature matrix",
            [PYTHON, "tools/feature_engineering/build_development_feature_matrix.py"],
        ),
        run_step(
            "offline development replay",
            [PYTHON, "tools/runtime_replay/replay_development_datasets.py"],
        ),
        run_step(
            ".NET AI/core tests",
            [
                "dotnet",
                "test",
                "software/SafeBand/tests/SafeBand.Core.Tests/SafeBand.Core.Tests.csproj",
                "--no-restore",
            ],
        ),
    ]
    report = build_report(steps)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Final report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Status      : {report['status']}")
    return 0 if report["all_automation_steps_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
