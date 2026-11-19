from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "generated"
REPORT_DIR = ROOT / "reports" / "dataset_expansion" / "acceptance"


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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATASET_DIR.glob("*.csv"))

    categories: dict[str, list[str]] = {}

    for path in files:
        category = classify(path.name)
        categories.setdefault(category, []).append(path.name)

    expected_categories = {
        "hard_negative",
        "hazard",
        "motion",
        "augmentation",
        "motion_profile",
        "ppg",
        "sensor_noise",
        "timing",
        "synthetic_running",
    }

    missing_categories = sorted(
        expected_categories - set(categories)
    )

    unknown_files = categories.get("unknown", [])

    checks = {
        "dataset_directory_exists": DATASET_DIR.exists(),
        "dataset_count": len(files) == 42,
        "expected_categories_present": not missing_categories,
        "no_unknown_files": not unknown_files,
    }

    report = {
        "acceptance_stage": "2.11.1",
        "stage_name": "Dataset Inventory",
        "dataset_directory": str(DATASET_DIR.relative_to(ROOT)).replace("\\", "/"),
        "dataset_count": len(files),
        "categories": {
            key: {
                "count": len(value),
                "files": value,
            }
            for key, value in sorted(categories.items())
        },
        "missing_categories": missing_categories,
        "unknown_files": unknown_files,
        "checks": checks,
        "overall_status": (
            "PASS"
            if all(checks.values())
            else "FAIL"
        ),
    }

    output = REPORT_DIR / "dataset_inventory_2.11.1.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("SAFEband MILESTONE 2.11.1 — DATASET INVENTORY")
    print()
    print(f"Dataset directory exists : {'PASS' if checks['dataset_directory_exists'] else 'FAIL'}")
    print(f"Dataset count            : {'PASS' if checks['dataset_count'] else 'FAIL'}")
    print(f"Expected categories      : {'PASS' if checks['expected_categories_present'] else 'FAIL'}")
    print(f"Unknown files            : {'PASS' if checks['no_unknown_files'] else 'FAIL'}")
    print()
    print(f"Datasets                 : {len(files)}")

    for category, values in sorted(categories.items()):
        print(f"{category:<25}: {len(values)}")

    if missing_categories:
        print()
        print("Missing categories:")
        for category in missing_categories:
            print(f"  - {category}")

    if unknown_files:
        print()
        print("Unknown files:")
        for filename in unknown_files:
            print(f"  - {filename}")

    print()
    print(f"Overall status           : {report['overall_status']}")
    print(f"Report                   : {output.relative_to(ROOT)}")

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
