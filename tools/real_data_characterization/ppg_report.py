import json
from pathlib import Path

path = Path("reports/real_data_characterization/stage_a_characterization.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("PPG VALIDITY / DROPOUT CHARACTERIZATION")
print("=" * 80)

for item in data["files"]:
    print()
    print(Path(item["file"]).name)
    print("-" * 80)

    for name, info in item["validity"].items():
        print(
            f"{name:20s} "
            f"valid={info['valid_count']:4d} "
            f"invalid={info['invalid_count']:4d} "
            f"valid_fraction={info['valid_fraction']:.2%} "
            f"max_valid_run={info['runs']['valid']['max']:3d} "
            f"max_invalid_run={info['runs']['invalid']['max']:3d}"
        )

    print()
    print("Sentinels:")

    for name, info in item["sentinels"].items():
        print(
            f"{name:20s} "
            f"-999_count={info['minus_999_count']:4d} "
            f"fraction={info['minus_999_fraction']:.2%}"
        )

print()
print("=" * 80)
