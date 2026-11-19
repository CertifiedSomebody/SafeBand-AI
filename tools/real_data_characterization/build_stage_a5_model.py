import json
from pathlib import Path


BASE = Path("reports/real_data_characterization")

INPUT = BASE / "stage_a4_empirical_specification.json"
OUTPUT = BASE / "stage_a5_synthetic_running_model.json"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():

    empirical = load_json(INPUT)

    model = {
        "document": "SafeBand Stage A.5 Synthetic RUNNING Model Specification",
        "version": "1.0",
        "status": "DESIGN_COMPLETE_GENERATOR_NOT_YET_RUN",

        "purpose": (
            "Define a stateful synthetic RUNNING data-generation model "
            "derived from measured SafeBand stationary and walking sessions."
        ),

        "provenance": {
            "source_specification": str(INPUT),
            "source_real_sessions": [
                "stationary.csv",
                "stationary_2.csv",
                "walking.csv",
            ],
            "synthetic_activity": "RUNNING",
            "validation_status": "UNVALIDATED_AGAINST_REAL_RUNNING",
        },

        "source_strategy": {
            "primary_motion_source": "walking.csv",
            "stationary_reference": [
                "stationary.csv",
                "stationary_2.csv",
            ],
            "method": (
                "Transform temporally ordered walking observations into "
                "higher-intensity RUNNING observations while preserving "
                "measured temporal structure."
            ),
            "independent_row_sampling": False,
            "independent_feature_sampling": False,
        },

        "session_model": {
            "activity": "RUNNING",

            "state_sequence": [
                "WALKING_TRANSITION",
                "RUNNING",
                "RUNNING_RECOVERY",
            ],

            "transition_behavior": {
                "walking_to_running": {
                    "type": "smooth_progression",
                    "duration_rows": {
                        "minimum": 8,
                        "maximum": 20,
                    },
                },

                "running_steady": {
                    "type": "stateful",
                    "minimum_duration_rows": 120,
                },

                "running_to_recovery": {
                    "type": "smooth_progression",
                    "duration_rows": {
                        "minimum": 8,
                        "maximum": 20,
                    },
                },
            },
        },

        "timing_model": {
            "source": "empirical_stage_a4",

            "normal_interval_ms": {
                "target_range": [
                    387,
                    430,
                ],
                "sampling": "empirical_walking_distribution",
            },

            "acquisition_stall": {
                "enabled": True,
                "duration_ms": {
                    "target": 5400,
                    "tolerance_ms": 100,
                },
                "spacing_ms": {
                    "target": 10280,
                    "tolerance_ms": 500,
                },
            },

            "implementation_rule": (
                "Timing must be generated as a sequential session process; "
                "do not generate elapsed_ms independently for each row."
            ),
        },

        "motion_model": {

            "acceleration": {
                "source": "walking.csv",
                "method": "stateful_intensity_transform",

                "running_intensity": {
                    "relative_to_walking": {
                        "minimum": 1.15,
                        "nominal": 1.35,
                        "maximum": 1.60,
                    }
                },

                "transition": {
                    "type": "smooth_ramp",
                    "preserve_sign_changes": True,
                    "preserve_temporal_correlation": True,
                },

                "noise": {
                    "type": "correlated",
                    "independent_row_noise": False,
                },
            },

            "gyro": {
                "source": "walking.csv",
                "method": "stateful_intensity_transform",

                "running_intensity": {
                    "relative_to_walking": {
                        "minimum": 1.20,
                        "nominal": 1.50,
                        "maximum": 1.90,
                    }
                },

                "transition": {
                    "type": "smooth_ramp",
                    "preserve_temporal_correlation": True,
                },

                "noise": {
                    "type": "correlated",
                    "independent_row_noise": False,
                },
            },

            "orientation": {
                "source": "walking.csv",
                "method": "temporally_smoothed_transform",
                "avoid_unbounded_drift": True,
            },

            "derived_magnitude": {
                "acceleration_magnitude": "recalculate_from_xyz",
                "gyro_magnitude": "recalculate_from_xyz",
            },
        },

        "activity_dynamics": {
            "running_characteristics": [
                "higher acceleration variability than walking",
                "higher gyro variability than walking",
                "temporally smooth motion",
                "persistent running state",
                "smooth transition into running",
                "smooth recovery from running",
            ],

            "forbidden_behavior": [
                "independent random rows",
                "white-noise-only motion",
                "instantaneous activity switching",
                "unbounded orientation drift",
                "arbitrary independent sensor values",
            ],
        },

        "ppg_model": {

            "architecture": "state_machine",

            "states": [
                "NO_FINGER",
                "FINGER_DETECTED",
                "HR_VALID",
                "SPO2_VALID",
                "DEGRADED",
            ],

            "dependencies": {
                "finger_detection": "activity_and_session_state",
                "raw_hr_valid": "finger_detection_plus_signal_quality",
                "raw_spo2_valid": "finger_detection_plus_signal_quality_plus_motion",
                "final_hr_valid": "raw_hr_valid_plus_filtering",
                "final_spo2_valid": "raw_spo2_valid_plus_filtering",
            },

            "running_behavior": {
                "finger_detection": {
                    "target": "high_probability",
                },

                "heart_rate": {
                    "target": "mostly_valid",
                    "motion_degradation": "moderate",
                },

                "spo2": {
                    "target": "less_reliable_than_hr",
                    "motion_degradation": "strong",
                },
            },

            "rules": [
                "HR validity must not be independent of finger detection.",
                "SpO2 validity must not be independent of finger detection.",
                "SpO2 must be more sensitive to motion degradation than HR.",
                "Raw validity and final validity remain separate states.",
                "The -999 sentinel must be preserved where firmware semantics require invalid values.",
            ],
        },

        "audio_model": {
            "source": "walking.csv",
            "method": "temporally_correlated_intensity_transform",
            "preserve_feature_coupling": True,
            "activity_effect": {
                "running": "higher but bounded activity-associated energy",
            },
            "avoid": [
                "independent random RMS features",
                "deterministic activity labels from audio alone",
            ],
        },

        "environment_model": {
            "source": "walking.csv",
            "method": "empirical_resampling_with_temporal_smoothing",
            "bme_valid": {
                "target": "near_100_percent",
            },
            "preserve_cross_sensor_relationships": True,
        },

        "power_model": {
            "source": "walking.csv",
            "method": "session_continuation",
            "battery_voltage": "temporally_smooth",
            "battery_percentage": "correlated_with_voltage",
            "charger_state": "explicit_firmware_state",
        },

        "schema_rules": {
            "preserve_original_column_count": True,
            "preserve_original_column_order": True,
            "preserve_firmware_field_names": True,
            "preserve_validity_flags": True,
            "preserve_minus_999_semantics": True,
        },

        "generation_parameters": {
            "default_rows": 1000,
            "random_seed": 20261118,

            "running_intensity": {
                "acceleration": 1.35,
                "gyro": 1.50,
            },

            "transition_rows": {
                "minimum": 8,
                "maximum": 20,
            },

            "correlation_window_rows": 12,

            "timing": {
                "normal_interval_distribution": "walking_empirical",
                "stall_duration_ms": 5400,
                "stall_spacing_ms": 10280,
            },
        },

        "validation_plan": {

            "schema": [
                "36-column compatibility",
                "column order compatibility",
                "firmware sentinel compatibility",
            ],

            "timing": [
                "normal interval distribution",
                "median interval",
                "p95 interval",
                "stall frequency",
                "stall duration",
                "stall spacing",
            ],

            "motion": [
                "acceleration magnitude distribution",
                "gyro magnitude distribution",
                "lag-1 autocorrelation",
                "temporal smoothness",
            ],

            "ppg": [
                "finger detection rate",
                "raw HR validity rate",
                "raw SpO2 validity rate",
                "final HR validity rate",
                "final SpO2 validity rate",
                "validity transition behavior",
            ],

            "environment": [
                "BME validity",
                "temperature distribution",
                "humidity distribution",
                "pressure distribution",
                "gas distribution",
            ],

            "power": [
                "battery voltage continuity",
                "battery percentage continuity",
                "charger state behavior",
            ],
        },

        "future_real_running_calibration": {
            "required": True,

            "when_available": [
                "Compare real RUNNING motion distributions.",
                "Compare timing distributions.",
                "Compare PPG validity behavior.",
                "Estimate systematic synthetic bias.",
                "Replace empirical assumptions with measured RUNNING parameters.",
                "Regenerate synthetic RUNNING fixtures.",
            ],

            "synthetic_data_status_after_real_data": (
                "development_fixture_only"
            ),
        },

        "next_stage": {
            "stage": "A.5.1",
            "name": "Implement synthetic RUNNING generator",
            "input": str(OUTPUT),
            "output": (
                "datasets/generated/"
                "running_synthetic_dataset.csv"
            ),
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
            model,
            handle,
            indent=2,
        )

    print()
    print("=" * 100)
    print("SAFEband STAGE A.5 — SYNTHETIC RUNNING MODEL")
    print("=" * 100)
    print()
    print("Status : DESIGN_COMPLETE_GENERATOR_NOT_YET_RUN")
    print()
    print("Source motion      : walking.csv")
    print("Acceleration scale : 1.35x")
    print("Gyro scale         : 1.50x")
    print("Transition model   : Stateful")
    print("Timing model       : Empirical walking + acquisition stalls")
    print("PPG model          : Activity/state dependent")
    print("Schema             : Original firmware schema preserved")
    print()
    print(f"Saved: {OUTPUT}")
    print()
    print("Next: A.5.1 — Implement synthetic RUNNING generator")
    print("=" * 100)


if __name__ == "__main__":
    main()
