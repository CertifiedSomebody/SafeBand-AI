import json
from pathlib import Path


BASE = Path("reports/real_data_characterization")

CHARACTERIZATION = BASE / "stage_a_characterization.json"
TEMPORAL = BASE / "stage_a_temporal_sensor_analysis.json"
VALIDITY = BASE / "stage_a_validity_transition_analysis.json"

OUTPUT = BASE / "stage_a4_empirical_specification.json"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_file_report(report, filename):
    for item in report.get("files", []):
        if Path(item.get("file", "")).name == filename:
            return item

    raise KeyError(
        f"{filename} not found in characterization report"
    )


def find_temporal_report(report, filename):
    for item in report.get("files", []):
        if Path(item.get("file", "")).name == filename:
            return item

    raise KeyError(
        f"{filename} not found in temporal report"
    )


def find_validity_report(report, filename):
    for item in report.get("files", []):
        if Path(item.get("file", "")).name == filename:
            return item

    raise KeyError(
        f"{filename} not found in validity report"
    )


def extract_activity_name(activity):
    if isinstance(activity, str):
        return activity

    if isinstance(activity, dict):
        for key in [
            "unique",
            "label",
            "name",
            "value",
            "activity",
        ]:
            value = activity.get(key)

            if isinstance(value, str):
                return value

            if isinstance(value, list) and value:
                return str(value[0])

        return "UNKNOWN"

    if isinstance(activity, list):
        if activity:
            return str(activity[0])

        return "UNKNOWN"

    return "UNKNOWN"


def extract_numeric_statistics(file_report):
    result = {}

    numeric_summary = file_report.get(
        "numeric_summary",
        {}
    )

    if not isinstance(numeric_summary, dict):
        return result

    for column, values in numeric_summary.items():

        if not isinstance(values, dict):
            continue

        selected = {}

        for key in [
            "count",
            "missing",
            "min",
            "q01",
            "q05",
            "q25",
            "median",
            "q75",
            "q95",
            "q99",
            "max",
            "mean",
            "std",
        ]:
            if key in values:
                selected[key] = values[key]

        result[column] = selected

    return result


def build_activity_spec(
    filename,
    characterization,
    temporal,
    validity,
):
    char = find_file_report(
        characterization,
        filename,
    )

    temp = find_temporal_report(
        temporal,
        filename,
    )

    valid = find_validity_report(
        validity,
        filename,
    )

    activity_name = extract_activity_name(
        char.get("activity")
    )

    timing = char.get(
        "timing",
        {}
    )

    return {
        "source_file": filename,

        "rows": char.get(
            "rows"
        ),

        "columns": char.get(
            "columns"
        ),

        "activity": activity_name,

        "activity_metadata": char.get(
            "activity",
            {}
        ),

        "timing": {
            "mean_delta_ms": timing.get(
                "mean_delta_ms"
            ),

            "std_delta_ms": timing.get(
                "std_delta_ms"
            ),

            "min_delta_ms": timing.get(
                "min_delta_ms"
            ),

            "median_delta_ms": timing.get(
                "median_delta_ms"
            ),

            "p95_delta_ms": timing.get(
                "p95_delta_ms"
            ),

            "p99_delta_ms": timing.get(
                "p99_delta_ms"
            ),

            "max_delta_ms": timing.get(
                "max_delta_ms"
            ),

            "gaps_over_1s": timing.get(
                "gaps_over_1s"
            ),

            "gaps_over_5s": timing.get(
                "gaps_over_5s"
            ),
        },

        "numeric_statistics": (
            extract_numeric_statistics(
                char
            )
        ),

        "temporal_structure": {
            "acceleration_magnitude": temp.get(
                "acceleration_magnitude",
                {}
            ),

            "gyro_magnitude": temp.get(
                "gyro_magnitude",
                {}
            ),

            "lag1_autocorrelation": temp.get(
                "lag1_autocorrelation",
                {}
            ),

            "strongest_correlations": temp.get(
                "strongest_correlations",
                []
            ),
        },

        "validity_structure": {
            "state_transitions": valid.get(
                "state_transitions",
                {}
            ),

            "run_lengths": valid.get(
                "run_lengths",
                {}
            ),

            "pairwise_states": valid.get(
                "pairwise_states",
                {}
            ),

            "conditional_probabilities": valid.get(
                "conditional_probabilities",
                []
            ),

            "timing_gaps": valid.get(
                "timing_gaps",
                {}
            ),
        },
    }


def build_cross_session_summary(
    activity_specs
):
    timing_by_activity = {}

    for spec in activity_specs:

        activity = spec["activity"]

        timing_by_activity[
            activity
        ] = spec["timing"]

    return {
        "activities": [
            spec["activity"]
            for spec in activity_specs
        ],

        "timing_by_activity":
            timing_by_activity,

        "known_real_sessions": [
            spec["source_file"]
            for spec in activity_specs
        ],
    }


def build_synthetic_constraints():

    return {
        "general": [
            "Preserve the 36-column SafeBand firmware CSV schema.",
            "Preserve firmware validity flags.",
            "Preserve firmware -999 sentinel semantics where applicable.",
            "Preserve temporal dependence; do not independently sample rows.",
            "Preserve approximately observed acquisition timing behavior.",
            "Model acquisition stalls explicitly rather than deleting them.",
            "Keep synthetic provenance separate from real hardware provenance.",
            "Do not present synthetic RUNNING as real experimental evidence.",
        ],

        "timing": [
            "Normal observed inter-sample intervals are concentrated around approximately 387-430 ms.",
            "Observed acquisition stalls are approximately 5.4 seconds.",
            "Observed stall spacing is approximately 10.28 seconds.",
            "Timing behavior should remain stateful and session-like.",
        ],

        "motion": [
            "Acceleration and gyro signals must exhibit temporal smoothness.",
            "Walking has substantially greater gyro variability than stationary sessions.",
            "Walking acceleration magnitude has greater variability than stationary sessions.",
            "Synthetic RUNNING should be derived from measured walking motion dynamics rather than invented independently.",
        ],

        "ppg": [
            "PPG validity must be generated as a state process.",
            "Finger detection must be modeled separately from HR and SpO2 validity.",
            "HR and SpO2 validity must not be sampled independently.",
            "Activity-conditioned validity behavior is required.",
            "Stationary no-finger/no-PPG behavior must remain possible.",
            "Walking may have continuous finger detection while SpO2 validity remains poor.",
            "Raw and final validity states must remain distinct.",
        ],

        "audio": [
            "Audio features should preserve temporal smoothness.",
            "RMS-derived audio features should retain their observed internal coupling.",
            "Audio should not be treated as a deterministic activity classifier.",
        ],

        "environment": [
            "BME680 validity should remain highly reliable based on observed sessions.",
            "Temperature, humidity, pressure, and gas should retain observed cross-sensor relationships.",
        ],

        "power": [
            "Battery fields should preserve empirical firmware behavior.",
            "Battery voltage and battery percentage are correlated but should not be treated as causal ground truth.",
            "Charger state must remain represented explicitly.",
        ],

        "provenance": [
            "Synthetic RUNNING must be marked as derived/synthetic/unvalidated.",
            "Future real RUNNING data must use the same schema and ingestion path.",
            "Synthetic assumptions must be replaceable without changing downstream consumers.",
        ],
    }


def main():

    characterization = load_json(
        CHARACTERIZATION
    )

    temporal = load_json(
        TEMPORAL
    )

    validity = load_json(
        VALIDITY
    )

    filenames = [
        "stationary.csv",
        "stationary_2.csv",
        "walking.csv",
    ]

    activity_specs = []

    for filename in filenames:

        activity_specs.append(
            build_activity_spec(
                filename,
                characterization,
                temporal,
                validity,
            )
        )

    output = {

        "document":
            "SafeBand Stage A.4 Empirical Characterization Specification",

        "version":
            "1.1",

        "status":
            "FINAL_CHARACTERIZATION_REVIEW",

        "scope": {

            "real_sessions":
                filenames,

            "synthetic_running":
                "NOT_YET_GENERATED",

            "purpose":
                "Empirical specification for Stage A.5 synthetic RUNNING model design.",
        },

        "activities":
            activity_specs,

        "cross_session_summary":
            build_cross_session_summary(
                activity_specs
            ),

        "synthetic_model_constraints":
            build_synthetic_constraints(),

        "known_limitations": [

            "Only three real SafeBand activity sessions are currently available.",

            "RUNNING real hardware data is unavailable because the hardware acquisition path is currently unavailable.",

            "The observed acquisition timing contains recurring acquisition stalls.",

            "Synthetic RUNNING must therefore remain a development fixture until real RUNNING data is collected.",

            "Observed battery behavior may contain session-specific confounding.",

            "PPG behavior is strongly session- and activity-dependent.",
        ],

        "next_stage": {

            "stage":
                "A.5",

            "name":
                "Synthetic RUNNING model design",

            "input":
                "This empirical specification",

            "output":
                "Documented stateful synthetic RUNNING generation model",
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            output,
            handle,
            indent=2,
        )

    print()
    print("=" * 100)
    print(
        "SAFEband STAGE A.4 — EMPIRICAL SPECIFICATION"
    )
    print("=" * 100)
    print()

    print(
        f"Real sessions : {len(activity_specs)}"
    )

    for spec in activity_specs:

        print(
            f"  {spec['activity']:<12} "
            f"{spec['source_file']:<20} "
            f"rows={spec['rows']}"
        )

    print()

    print(
        "Synthetic RUNNING : NOT YET GENERATED"
    )

    print(
        "Status             : FINAL_CHARACTERIZATION_REVIEW"
    )

    print()

    print(
        f"Saved: {OUTPUT}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
