from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

GENERATED_DIR = ROOT / "datasets" / "generated"
REPORT_DIR = ROOT / "reports" / "dataset_expansion"

SCHEMA_PATH = (
    ROOT
    / "tools"
    / "dataset_expansion"
    / "provenance"
    / "session_metadata_schema.json"
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def normalize_relative(path: Path) -> str:
    return (
        str(path.relative_to(ROOT))
        .replace("\\", "/")
    )


def get_first(
    data: dict,
    keys: list[str],
    default=None,
):
    for key in keys:
        if key in data:
            value = data[key]

            if value is not None:
                return value

    return default


def find_native_provenance(
    dataset_path: Path,
) -> Path | None:
    candidates = [
        Path(
            str(dataset_path)
            + ".provenance.json"
        ),
        dataset_path.parent
        / f"{dataset_path.stem}.provenance.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def find_validation_report(
    dataset_path: Path,
) -> str | None:
    dataset_name = dataset_path.stem

    candidates = sorted(
        REPORT_DIR.rglob(
            f"{dataset_name}*validation.json"
        )
    )

    if not candidates:
        return None

    return normalize_relative(
        candidates[0]
    )


def infer_stage(
    dataset_path: Path,
    provenance: dict,
) -> str:
    value = get_first(
        provenance,
        [
            "generation_stage",
            "stage",
            "expansion_stage",
        ],
    )

    if value:
        return str(value)

    name = dataset_path.stem.lower()

    if name.startswith("hazard_"):
        return "2.10.7"

    if name.startswith("hardneg_"):
        return "2.10.6"

    if name.startswith("timing_"):
        return "2.10.5"

    if name.startswith("sensor_noise_"):
        return "2.10.4"

    if name.startswith("ppg_"):
        return "2.10.3"

    if name.startswith("motion_"):
        return "2.10.2"

    if name.startswith("walking_profile_"):
        return "2.10.2"

    if name.startswith("walking_aug_"):
        return "2.10.1"

    if name == "running_synthetic_dataset":
        return "2.9"

    return "UNKNOWN"


def infer_profile(
    dataset_path: Path,
    provenance: dict,
) -> str:
    value = get_first(
        provenance,
        [
            "profile_name",
        ],
    )

    if value:
        return str(value)

    value = provenance.get(
        "profile"
    )

    if isinstance(value, str):
        return value

    name = dataset_path.stem

    prefixes = [
        "hazard_",
        "hardneg_",
        "timing_",
        "sensor_noise_",
        "ppg_",
        "motion_",
        "walking_profile_",
    ]

    for prefix in prefixes:
        if name.startswith(prefix):
            profile = name[len(prefix):]

            if profile.endswith("_001"):
                profile = profile[:-4]

            if profile.endswith("_002"):
                profile = profile[:-4]

            return profile.upper()

    if name == "running_synthetic_dataset":
        return "RUNNING_SYNTHETIC"

    if name == "walking_aug_001":
        return "WALKING_AUGMENTED"

    return "BASE_DATASET"


def infer_activity(
    df: pd.DataFrame,
    provenance: dict,
) -> str:
    value = get_first(
        provenance,
        [
            "activity",
            "label",
            "source_activity_values",
        ],
    )

    if isinstance(value, list):
        return "|".join(
            str(item)
            for item in value
        )

    if value:
        return str(value)

    if "activity" in df.columns and not df.empty:
        values = (
            df["activity"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(values) == 1:
            return values[0]

        if len(values) > 1:
            return "|".join(
                sorted(values)
            )

    return "UNKNOWN"


def infer_source_dataset(
    provenance: dict,
) -> str:
    value = get_first(
        provenance,
        [
            "source_dataset",
            "source",
            "input_dataset",
            "source_file",
        ],
    )

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in [
            "dataset",
            "path",
            "file",
            "source",
        ]:
            nested = value.get(key)

            if nested:
                return str(nested)

    return "UNKNOWN"


def infer_source_path(
    provenance: dict,
) -> str:
    value = get_first(
        provenance,
        [
            "source_path",
            "input_path",
        ],
    )

    if value:
        return str(value)

    value = provenance.get(
        "source"
    )

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in [
            "path",
            "file",
        ]:
            nested = value.get(key)

            if nested:
                return str(nested)

    return "UNKNOWN"


def infer_seed(
    provenance: dict,
):
    value = get_first(
        provenance,
        [
            "seed",
            "random_seed",
        ],
    )

    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def infer_tool(
    provenance: dict,
) -> str:
    value = get_first(
        provenance,
        [
            "generation_tool",
            "generator",
            "tool",
        ],
    )

    if value:
        return str(value)

    return "UNKNOWN"


def infer_status(
    provenance: dict,
    dataset_path: Path,
) -> str:
    value = get_first(
        provenance,
        [
            "synthetic_status",
            "status",
        ],
    )

    if value:
        return str(value)

    if dataset_path.stem == (
        "running_synthetic_dataset"
    ):
        return (
            "SYNTHETIC_DEVELOPMENT_FIXTURE"
        )

    return "SYNTHETIC_UNVALIDATED"


def infer_parent_dataset(
    provenance: dict,
) -> str:
    value = get_first(
        provenance,
        [
            "parent_dataset",
            "parent",
            "source_dataset",
        ],
    )

    if value:
        return str(value)

    source = provenance.get(
        "source"
    )

    if isinstance(source, str):
        return source

    return "UNKNOWN"


def infer_schema_version(
    provenance: dict,
) -> str:
    value = provenance.get(
        "schema_version"
    )

    if value is None:
        return "UNKNOWN"

    return str(value)


def infer_parameters(
    provenance: dict,
) -> dict:
    value = get_first(
        provenance,
        [
            "augmentation_parameters",
            "parameters",
            "config",
            "profile_parameters",
        ],
        {},
    )

    if isinstance(value, dict):
        return value

    return {
        "value": value
    }


def build_entry(
    dataset_path: Path,
) -> dict:
    provenance_path = (
        find_native_provenance(
            dataset_path
        )
    )

    provenance = {}

    if provenance_path is not None:
        provenance = load_json(
            provenance_path
        )

    df = pd.read_csv(
        dataset_path
    )

    dataset_id = get_first(
        provenance,
        [
            "dataset_id",
            "id",
        ],
        dataset_path.stem,
    )

    validation_report = (
        find_validation_report(
            dataset_path
        )
    )

    timestamp = get_first(
        provenance,
        [
            "timestamp",
            "generated_at",
            "created_at",
        ],
    )

    if timestamp is None:
        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

    return {
        "dataset_id": str(dataset_id),

        "source_dataset": (
            infer_source_dataset(
                provenance
            )
        ),

        "source_path": (
            infer_source_path(
                provenance
            )
        ),

        "generation_stage": (
            infer_stage(
                dataset_path,
                provenance,
            )
        ),

        "generation_tool": (
            infer_tool(
                provenance
            )
        ),

        "profile": (
            infer_profile(
                dataset_path,
                provenance,
            )
        ),

        "seed": infer_seed(
            provenance
        ),

        "timestamp": str(
            timestamp
        ),

        "row_count": int(
            len(df)
        ),

        "column_count": int(
            len(df.columns)
        ),

        "activity": (
            infer_activity(
                df,
                provenance,
            )
        ),

        "synthetic_status": (
            infer_status(
                provenance,
                dataset_path,
            )
        ),

        "parent_dataset": (
            infer_parent_dataset(
                provenance
            )
        ),

        "schema_version": (
            infer_schema_version(
                provenance
            )
        ),

        "augmentation_parameters": (
            infer_parameters(
                provenance
            )
        ),

        "validation_report": (
            validation_report
        ),

        "provenance_file": (
            normalize_relative(
                provenance_path
            )
            if provenance_path is not None
            else None
        ),
    }


def build_session(
    output_path: Path,
) -> None:
    if not GENERATED_DIR.exists():
        raise RuntimeError(
            "Generated dataset directory "
            f"not found: {GENERATED_DIR}"
        )

    dataset_paths = sorted(
        path
        for path in GENERATED_DIR.glob(
            "*.csv"
        )
        if not path.name.startswith(
            "~$"
        )
    )

    entries = [
        build_entry(
            dataset_path
        )
        for dataset_path in dataset_paths
    ]

    session = {
        "session_name": (
            "SafeBand Dataset Expansion "
            "2.10.8"
        ),
        "session_schema_version": "1.0",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "repository_root": str(
            ROOT
        ),
        "generated_dataset_count": (
            len(entries)
        ),
        "datasets": entries,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            session,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "SESSION METADATA BUILD PASS"
    )

    print(
        f"Datasets discovered : "
        f"{len(entries)}"
    )

    print(
        f"Output              : "
        f"{output_path.relative_to(ROOT)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    build_session(
        ROOT / args.output
    )


if __name__ == "__main__":
    main()
