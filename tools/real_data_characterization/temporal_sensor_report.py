import json
from pathlib import Path


path = Path(
    "reports/real_data_characterization/"
    "stage_a_temporal_sensor_analysis.json"
)

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)


print("=" * 90)
print("SAFEband TEMPORAL / SENSOR ANALYSIS")
print("=" * 90)


for item in data["files"]:

    print()
    print(Path(item["file"]).name)
    print("-" * 90)

    print(
        f"Timing gaps >1s  : "
        f"{item['timing']['gaps_gt_1000ms']}"
    )

    print(
        f"Timing gaps >5s  : "
        f"{item['timing']['gaps_gt_5000ms']}"
    )

    print()
    print("Motion magnitude:")

    for name, stats in item["motion_magnitude"].items():
        print(
            f"  {name:24s} "
            f"mean={stats['mean']:.4f} "
            f"std={stats['std']:.4f} "
            f"median={stats['median']:.4f} "
            f"p95={stats['p95']:.4f} "
            f"p99={stats['p99']:.4f}"
        )

    print()
    print("Lag-1 autocorrelation:")

    important = [
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2",
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
        "mic_rms",
        "max_ir",
        "max_red",
        "final_hr",
        "final_spo2",
    ]

    for signal in important:
        value = item["lag1_autocorrelation"].get(signal)

        if value is None:
            print(f"  {signal:24s} unavailable")
        else:
            print(f"  {signal:24s} {value:+.5f}")

    print()
    print("Top cross-sensor correlations:")

    for pair in item["strongest_correlations"][:10]:
        print(
            f"  {pair['first']:20s} <-> "
            f"{pair['second']:20s} "
            f"{pair['correlation']:+.5f}"
        )


print()
print("=" * 90)
