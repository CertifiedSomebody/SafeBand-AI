import json
from pathlib import Path


PATH = Path(
    "reports/real_data_characterization/"
    "stage_a_validity_transition_analysis.json"
)

with PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)


print("=" * 100)
print("SAFEband STAGE A.3 — VALIDITY TRANSITIONS / GAP STRUCTURE")
print("=" * 100)


for item in data["files"]:

    print()
    print(Path(item["file"]).name)
    print("-" * 100)

    print("TIMING GAPS")

    gaps = item["timing_gaps"]

    print(
        f"  Count              : {gaps['gap_count']}"
    )

    print(
        f"  Mean gap           : "
        f"{gaps['gap_mean_ms']:.2f} ms"
        if gaps["gap_mean_ms"] is not None
        else "  Mean gap           : unavailable"
    )

    print(
        f"  Median gap         : "
        f"{gaps['gap_median_ms']:.2f} ms"
        if gaps["gap_median_ms"] is not None
        else "  Median gap         : unavailable"
    )

    print(
        f"  Std gap            : "
        f"{gaps['gap_std_ms']:.2f} ms"
        if gaps["gap_std_ms"] is not None
        else "  Std gap            : unavailable"
    )

    print(
        f"  Min / Max          : "
        f"{gaps['gap_min_ms']:.2f} / "
        f"{gaps['gap_max_ms']:.2f} ms"
        if gaps["gap_min_ms"] is not None
        else "  Min / Max          : unavailable"
    )

    print(
        f"  Mean gap spacing   : "
        f"{gaps['gap_spacing_mean_ms']:.2f} ms"
        if gaps["gap_spacing_mean_ms"] is not None
        else "  Mean gap spacing   : unavailable"
    )

    print(
        f"  Median gap spacing : "
        f"{gaps['gap_spacing_median_ms']:.2f} ms"
        if gaps["gap_spacing_median_ms"] is not None
        else "  Median gap spacing : unavailable"
    )

    print()
    print("STATE TRANSITIONS")

    for column, transition in item["state_transitions"].items():

        p = transition["probabilities"]

        print()
        print(f"  {column}")

        print(
            f"    V→V : {transition['valid_to_valid']:4d} "
            f"({p.get('valid_to_valid', 0):.2%})"
        )

        print(
            f"    V→I : {transition['valid_to_invalid']:4d} "
            f"({p.get('valid_to_invalid', 0):.2%})"
        )

        print(
            f"    I→V : {transition['invalid_to_valid']:4d} "
            f"({p.get('invalid_to_valid', 0):.2%})"
        )

        print(
            f"    I→I : {transition['invalid_to_invalid']:4d} "
            f"({p.get('invalid_to_invalid', 0):.2%})"
        )

    print()
    print("RUN LENGTHS")

    for column, runs in item["run_lengths"].items():

        valid = runs["valid"]
        invalid = runs["invalid"]

        print()
        print(f"  {column}")

        print(
            f"    VALID   count={valid['count']:3d} "
            f"median={valid['median']} "
            f"p95={valid['p95']} "
            f"max={valid['max']}"
        )

        print(
            f"    INVALID count={invalid['count']:3d} "
            f"median={invalid['median']} "
            f"p95={invalid['p95']} "
            f"max={invalid['max']}"
        )

    print()
    print("FINGER → PPG VALIDITY")

    for item2 in item["conditional_probabilities"]:

        if item2["condition"] != "finger_detected":
            continue

        print(
            f"  P(f({item2['target']}) | finger) = "
            f"{item2['probability']:.2%}"
            if item2["probability"] is not None
            else
            f"  P(f({item2['target']}) | finger) = unavailable"
        )

    print()
    print("PPG STATE COUPLING")

    for key, table in item["pairwise_states"].items():

        if not (
            "hr_valid" in key
            or "spo2_valid" in key
        ):
            continue

        print(f"  {key}")

        fractions = table["fractions"]

        print(
            f"    both valid       : "
            f"{fractions.get('first_valid_second_valid', 0):.2%}"
        )

        print(
            f"    first valid only : "
            f"{fractions.get('first_valid_second_invalid', 0):.2%}"
        )

        print(
            f"    second valid only: "
            f"{fractions.get('first_invalid_second_valid', 0):.2%}"
        )

        print(
            f"    both invalid     : "
            f"{fractions.get('first_invalid_second_invalid', 0):.2%}"
        )


print()
print("=" * 100)
