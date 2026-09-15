from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.sisfall_features import (
    FEATURE_COLUMNS,
    adxl_counts_to_g,
    mma_counts_to_g,
    downsample_200_to_20,
    extract_features,
)

PAT = re.compile(r"^(D|F)(\d{2})_(SA|SE)(\d{2})_R(\d{2})\.txt$", re.I)


def parse_raw(z: zipfile.ZipFile, name: str, sensor: str):
    """Fast SisFall parser.

    The previous implementation converted every one of ~15.9M rows through
    Python int() calls. That is unnecessarily slow. numpy.fromstring parses
    the complete numeric text in C and is substantially faster.
    """
    with z.open(name) as fh:
        raw = fh.read()

    # SisFall terminates EVERY row with a semicolon, e.g.
    #   9, -18, 240, -12, 74, -2, 45, -77, 1042;
    #
    # np.fromstring(..., sep=",") needs a comma between the last value of
    # one row and the first value of the next row. Therefore convert the
    # row terminator ";<newline>" into ",<newline>". Remove the final
    # terminator so we do not create one extra empty field.
    text = raw.replace(b";\r\n", b",\n").replace(b";\n", b",\n")
    text = text.rstrip(b",\r\n")
    arr = np.fromstring(text, sep=",", dtype=np.int32)

    # A small number of released SisFall recordings contain an incomplete
    # final line (only one value after the last complete 9-value sample).
    # Do not discard the recording; discard only that incomplete tail sample.
    remainder = arr.size % 9
    if remainder:
        if remainder != 1:
            raise ValueError(
                f"Parsed {arr.size} values from {name}; "
                f"unexpected remainder {remainder} (expected 0 or 1)."
            )
        arr = arr[:-remainder]

    if arr.size == 0:
        return []

    usable = (arr.size // 9) * 9
    if usable != arr.size:
        arr = arr[:usable]

    arr = arr.reshape(-1, 9)

    if sensor == "adxl":
        xyz = arr[:, :3].astype(np.float64) * ((2.0 * 16.0) / (2 ** 13))
    elif sensor == "mma":
        xyz = arr[:, 6:9].astype(np.float64) * ((2.0 * 8.0) / (2 ** 14))
    else:
        raise ValueError(sensor)

    # 200 Hz -> 20 Hz, exactly 10 raw samples per output sample.
    n = (len(xyz) // 10) * 10
    if n < 10:
        return []

    xyz = xyz[:n].reshape(-1, 10, 3).mean(axis=1)
    return [tuple(row) for row in xyz]


def main():
    ap = argparse.ArgumentParser(
        description="Prepare SisFall for BITS-2 V6 cross-dataset testing."
    )
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensor", choices=["adxl", "mma"], default="adxl")
    ap.add_argument("--window-samples", type=int, default=60)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    if not 0 <= args.overlap < 1:
        ap.error("--overlap must be >= 0 and < 1")
    if args.window_samples <= 0:
        ap.error("--window-samples must be > 0")

    step = max(1, int(round(args.window_samples * (1 - args.overlap))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "subject_id",
        "source_file",
        "activity_code",
        "recording_type",
        "trial",
        "start_sample",
        "window_samples",
        *FEATURE_COLUMNS,
    ]

    total = 0
    files = 0
    bad = 0

    with zipfile.ZipFile(args.zip) as z, out.open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for name in z.namelist():
            base = os.path.basename(name)
            m = PAT.match(base)
            if not m:
                continue

            typ, code_num, prefix, sid, trial = m.groups()

            samples = parse_raw(z, name, args.sensor)

            if len(samples) < args.window_samples:
                bad += 1
                continue

            subject = prefix.upper() + sid
            recording_type = "FALL" if typ.upper() == "F" else "NONFALL"
            activity_code = typ.upper() + code_num

            for start in range(
                0,
                len(samples) - args.window_samples + 1,
                step,
            ):
                feat = extract_features(
                    samples[start : start + args.window_samples]
                )

                writer.writerow(
                    {
                        "subject_id": subject,
                        "source_file": name,
                        "activity_code": activity_code,
                        "recording_type": recording_type,
                        "trial": int(trial),
                        "start_sample": start,
                        "window_samples": args.window_samples,
                        **feat,
                    }
                )
                total += 1

            files += 1
            if files % 250 == 0:
                print(
                    f"[SisFall] recordings={files}, windows={total}",
                    flush=True,
                )

    print(
        f"[SisFall] complete: recordings={files}, windows={total}, "
        f"short_or_invalid={bad}, sensor={args.sensor}, rate=20Hz, "
        f"window={args.window_samples}, step={step}, units=g"
    )


if __name__ == "__main__":
    main()
