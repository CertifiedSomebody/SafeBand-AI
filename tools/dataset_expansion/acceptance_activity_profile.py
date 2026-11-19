from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "generated"
REPORT_DIR = ROOT / "reports" / "dataset_expansion" / "acceptance"


EXPECTED_GENERATED_ACTIVITIES = {
    "WALKING",
    "RUNNING",
}

EXPECTED_MOTION_PROFILES = {
    "WALKING_SLOW",
    "WALKING_NORMAL",
    "WALKING_BRISK",
    "WALKING_ACCELERATION",
    "WALKING_DECELERATION",
    "WALKING_START",
    "WALKING_STOP",
    "WALKING_IRREGULAR",
    "WALKING_TURNING",
}

EXPECTED_TIMING_PROFILES = {
    "TIMING_NORMAL",
    "TIMING_JITTERED",
    "TIMING_SLOW",
    "TIMING_BURSTY",
    "TIMING_STALL_LIGHT",
    "TIMING_STALL_HEAVY",
}

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

EXPECTED_PPG_PROFILES = {
    "GOOD_CONTACT",
}

EXPECTED_SENSOR_NOISE_PROFILES = {
    "NOISE_LOW",
}


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_native_provenance(path: Path) -> dict:
    candidates = [
        Path(str(path) + ".provenance.json"),
        path.with_suffix(".provenance.json"),
    ]

    for provenance_path in candidates:
        if provenance_path.exists():
            with provenance_path.open(
                "r",
                encoding="utf-8-sig",
            ) as handle:
                return json.load(handle)

    return {}


def classify(name: str) -> str:
    n = name.lower()

    if n.startswith("hardneg_"):
        return "hard_negative"
    if n.startswith("hazard_"):
        return "hazard"
    if n.startswith("motion_"):
        return "motion"
    if n.startswith("walking_aug_"):
        return "augmentation"
    if n.startswith("walking_profile_"):
        return "motion_profile"
    if n.startswith("ppg_"):
        return "ppg"
    if n.startswith("sensor_noise_"):
        return "sensor_noise"
    if n.startswith("timing_"):
        return "timing"
    if n.startswith("running_synthetic"):
        return "synthetic_running"

    return "unknown"


def main() -> int:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        DATASET_DIR.glob("*.csv")
    )

    failures: dict[str, list[str]] = {}

    activities: dict[str, list[str]] = {}
    motion_profiles: dict[str, list[str]] = {}
    timing_profiles: dict[str, list[str]] = {}
    hard_negative_profiles: dict[str, list[str]] = {}
    hazard_types: dict[str, list[str]] = {}
    ppg_profiles: dict[str, list[str]] = {}
    sensor_noise_profiles: dict[str, list[str]] = {}

    def add_failure(
        filename: str,
        message: str,
    ) -> None:
        failures.setdefault(
            filename,
            [],
        ).append(message)

    for path in files:
        filename = path.name
        category = classify(filename)

        try:
            df = load_csv(path)
        except Exception as exc:
            add_failure(
                filename,
                f"CSV read failure: {exc}",
            )
            continue

        if "activity" not in df.columns:
            add_failure(
                filename,
                "Missing activity column",
            )
            continue

        activity_values = (
            df["activity"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist()
        )

        if not activity_values:
            add_failure(
                filename,
                "Activity column contains no values",
            )
        else:
            for activity in activity_values:
                activities.setdefault(
                    activity,
                    [],
                ).append(filename)

                if activity not in EXPECTED_GENERATED_ACTIVITIES:
                    add_failure(
                        filename,
                        f"Unexpected activity: {activity}",
                    )

        native = load_native_provenance(path)

        profile = native.get("profile")

        if category in ("motion", "motion_profile"):
            if isinstance(profile, str):
                profile_upper = profile.strip().upper()

                motion_profiles.setdefault(
                    profile_upper,
                    [],
                ).append(filename)

        elif category == "timing":
            if isinstance(profile, str):
                profile_upper = profile.strip().upper()

                timing_profiles.setdefault(
                    profile_upper,
                    [],
                ).append(filename)

        elif category == "hard_negative":
            if isinstance(profile, str):
                profile_upper = profile.strip().upper()

                hard_negative_profiles.setdefault(
                    profile_upper,
                    [],
                ).append(filename)

        elif category == "ppg":
            if isinstance(profile, dict):
                profile_identity = (
                    Path(filename)
                    .stem
                    .replace("_001", "")
                    .replace("ppg_", "")
                    .upper()
                )

                ppg_profiles.setdefault(
                    profile_identity,
                    [],
                ).append(filename)

        elif category == "sensor_noise":
            profile_name = native.get("profile_name")

            if isinstance(profile_name, str):
                profile_upper = (
                    profile_name.strip().upper()
                )

                sensor_noise_profiles.setdefault(
                    profile_upper,
                    [],
                ).append(filename)

        if category == "hazard":
            if "hazard_type" not in df.columns:
                add_failure(
                    filename,
                    "Hazard dataset missing hazard_type column",
                )
            else:
                values = (
                    df["hazard_type"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .unique()
                    .tolist()
                )

                for hazard_type in values:
                    hazard_types.setdefault(
                        hazard_type,
                        [],
                    ).append(filename)

                    if (
                        hazard_type != "NONE"
                        and hazard_type
                        not in EXPECTED_HAZARD_TYPES
                    ):
                        add_failure(
                            filename,
                            f"Unexpected hazard type: {hazard_type}",
                        )

            if "is_hazard" not in df.columns:
                add_failure(
                    filename,
                    "Hazard dataset missing is_hazard column",
                )

    discovered_activities = set(
        activities
    )

    discovered_motion_profiles = set(
        motion_profiles
    )

    discovered_timing_profiles = set(
        timing_profiles
    )

    discovered_hard_negative_profiles = set(
        hard_negative_profiles
    )

    discovered_hazard_types = set(
        hazard_types
    )

    discovered_ppg_profiles = set(
        ppg_profiles
    )

    discovered_sensor_noise_profiles = set(
        sensor_noise_profiles
    )

    missing_activities = sorted(
        EXPECTED_GENERATED_ACTIVITIES
        - discovered_activities
    )

    missing_motion_profiles = sorted(
        EXPECTED_MOTION_PROFILES
        - discovered_motion_profiles
    )

    missing_timing_profiles = sorted(
        EXPECTED_TIMING_PROFILES
        - discovered_timing_profiles
    )

    missing_hard_negative_profiles = sorted(
        EXPECTED_HARD_NEGATIVE_PROFILES
        - discovered_hard_negative_profiles
    )

    missing_hazard_types = sorted(
        EXPECTED_HAZARD_TYPES
        - discovered_hazard_types
    )

    missing_ppg_profiles = sorted(
        EXPECTED_PPG_PROFILES
        - discovered_ppg_profiles
    )

    missing_sensor_noise_profiles = sorted(
        EXPECTED_SENSOR_NOISE_PROFILES
        - discovered_sensor_noise_profiles
    )

    checks = {
        "dataset_count": len(files) == 42,
        "all_datasets_readable": not any(
            "CSV read failure" in message
            for messages in failures.values()
            for message in messages
        ),
        "activities_valid": not any(
            "Unexpected activity" in message
            or "Missing activity column" in message
            or "Activity column contains no values" in message
            for messages in failures.values()
            for message in messages
        ),
        "activity_coverage": not missing_activities,
        "motion_profile_coverage": not missing_motion_profiles,
        "timing_profile_coverage": not missing_timing_profiles,
        "hard_negative_profile_coverage": not missing_hard_negative_profiles,
        "hazard_type_coverage": not missing_hazard_types,
        "ppg_profile_coverage": not missing_ppg_profiles,
        "sensor_noise_profile_coverage": not missing_sensor_noise_profiles,
        "hazard_schema": not any(
            "Hazard dataset missing" in message
            for messages in failures.values()
            for message in messages
        ),
        "no_other_failures": not failures,
    }

    report = {
        "acceptance_stage": "2.11.5",
        "stage_name": "Activity and Profile Coverage",
        "dataset_count": len(files),
        "activities": {
            key: sorted(value)
            for key, value in sorted(
                activities.items()
            )
        },
        "motion_profiles": {
            key: sorted(value)
            for key, value in sorted(
                motion_profiles.items()
            )
        },
        "timing_profiles": {
            key: sorted(value)
            for key, value in sorted(
                timing_profiles.items()
            )
        },
        "hard_negative_profiles": {
            key: sorted(value)
            for key, value in sorted(
                hard_negative_profiles.items()
            )
        },
        "hazard_types": {
            key: sorted(value)
            for key, value in sorted(
                hazard_types.items()
            )
        },
        "ppg_profiles": {
            key: sorted(value)
            for key, value in sorted(
                ppg_profiles.items()
            )
        },
        "sensor_noise_profiles": {
            key: sorted(value)
            for key, value in sorted(
                sensor_noise_profiles.items()
            )
        },
        "missing": {
            "activities": missing_activities,
            "motion_profiles": missing_motion_profiles,
            "timing_profiles": missing_timing_profiles,
            "hard_negative_profiles": missing_hard_negative_profiles,
            "hazard_types": missing_hazard_types,
            "ppg_profiles": missing_ppg_profiles,
            "sensor_noise_profiles": missing_sensor_noise_profiles,
        },
        "failures": failures,
        "checks": checks,
        "overall_status": (
            "PASS"
            if all(checks.values())
            else "FAIL"
        ),
    }

    output = (
        REPORT_DIR
        / "activity_profile_coverage_2.11.5.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SAFEband MILESTONE 2.11.5 — "
        "ACTIVITY/PROFILE COVERAGE"
    )
    print()

    print(
        f"Dataset count            : "
        f"{'PASS' if checks['dataset_count'] else 'FAIL'}"
    )
    print(
        f"Datasets readable        : "
        f"{'PASS' if checks['all_datasets_readable'] else 'FAIL'}"
    )
    print(
        f"Activities valid         : "
        f"{'PASS' if checks['activities_valid'] else 'FAIL'}"
    )
    print(
        f"Activity coverage        : "
        f"{'PASS' if checks['activity_coverage'] else 'FAIL'}"
    )
    print(
        f"Motion profiles          : "
        f"{'PASS' if checks['motion_profile_coverage'] else 'FAIL'}"
    )
    print(
        f"Timing profiles          : "
        f"{'PASS' if checks['timing_profile_coverage'] else 'FAIL'}"
    )
    print(
        f"Hard-negative profiles   : "
        f"{'PASS' if checks['hard_negative_profile_coverage'] else 'FAIL'}"
    )
    print(
        f"Hazard types             : "
        f"{'PASS' if checks['hazard_type_coverage'] else 'FAIL'}"
    )
    print(
        f"PPG profiles             : "
        f"{'PASS' if checks['ppg_profile_coverage'] else 'FAIL'}"
    )
    print(
        f"Sensor-noise profiles    : "
        f"{'PASS' if checks['sensor_noise_profile_coverage'] else 'FAIL'}"
    )
    print(
        f"Hazard schema            : "
        f"{'PASS' if checks['hazard_schema'] else 'FAIL'}"
    )
    print()

    print("Activities:")
    for activity, values in sorted(
        activities.items()
    ):
        print(
            f"  {activity:<20}: {len(values)}"
        )

    print()
    print(
        f"Overall status           : "
        f"{report['overall_status']}"
    )
    print(
        f"Report                   : "
        f"{output.relative_to(ROOT)}"
    )

    if failures:
        print()
        print("Failures:")
        for filename, messages in sorted(
            failures.items()
        ):
            print(f"  {filename}")
            for message in messages:
                print(f"    - {message}")

    return (
        0
        if report["overall_status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
