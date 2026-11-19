import json
from pathlib import Path

path = Path("reports/real_data_characterization/stage_a_characterization.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("SAFEband REAL-DATA CHARACTERIZATION — COMPACT SUMMARY")
print("=" * 80)

for item in data["files"]:
    timing = item["timing"]
    percentiles = timing["percentiles_ms"]

    print()
    print(Path(item["file"]).name)
    print("-" * 80)
    print(f"Rows              : {item['rows']}")
    print(f"Columns           : {item['columns']}")
    print(f"Activity          : {', '.join(item['activity']['unique'])}")
    print(f"Timing mean       : {timing['mean_ms']:.2f} ms")
    print(f"Timing median     : {percentiles['p50']:.2f} ms")
    print(f"Timing p95        : {percentiles['p95']:.2f} ms")
    print(f"Timing p99        : {percentiles['p99']:.2f} ms")
    print(f"Timing max        : {timing['max_ms']:.2f} ms")
    print(f"Gaps > 1 s        : {timing['intervals_gt_1000ms']}")
    print(f"Gaps > 5 s        : {timing['intervals_gt_5000ms']}")
    print(f"Fraction > 5 s    : {timing['fraction_gt_5000ms']:.2%}")

print()
print("=" * 80)
