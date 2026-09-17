from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


BVP_FS = 64
ACC_FS = 32
WINDOW_SEC = 8
SHIFT_SEC = 2
WINDOW_BVP = BVP_FS * WINDOW_SEC
SHIFT_BVP = BVP_FS * SHIFT_SEC
WINDOW_ACC = ACC_FS * WINDOW_SEC
SHIFT_ACC = ACC_FS * SHIFT_SEC


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare PPG-DaLiA synchronized BVP+ACC+HR windows."
    )
    p.add_argument("--root", default=r"datasets/raw/PPG-DaLiA/ppg_dalia/data/PPG_FieldStudy")
    p.add_argument("--out", default=r"datasets/processed/ppgdalia_windows.npz")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--include-activity", action="store_true")
    return p.parse_args()


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def validate_subject(obj, sid):
    required = {"activity", "label", "signal", "rpeaks", "subject"}
    if not isinstance(obj, dict):
        raise ValueError(f"{sid}: pickle is not a dict")

    missing = required - set(obj)
    if missing:
        raise ValueError(f"{sid}: missing keys {sorted(missing)}")

    signal = obj["signal"]
    if not isinstance(signal, dict):
        raise ValueError(f"{sid}: signal is not a dict")

    wrist = signal.get("wrist")
    chest = signal.get("chest")
    if not isinstance(wrist, dict) or not isinstance(chest, dict):
        raise ValueError(f"{sid}: missing wrist/chest dictionaries")

    for key in ("BVP", "ACC"):
        if key not in wrist:
            raise ValueError(f"{sid}: missing wrist/{key}")

    bvp = np.asarray(wrist["BVP"])
    acc = np.asarray(wrist["ACC"])
    labels = np.asarray(obj["label"]).reshape(-1)
    activity = np.asarray(obj["activity"]).reshape(-1)

    if bvp.ndim != 1:
        bvp = bvp.reshape(-1)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"{sid}: wrist ACC must have shape (N,3), got {acc.shape}")

    if labels.ndim != 1:
        raise ValueError(f"{sid}: label must be 1-D")

    expected = max(0, (len(bvp) - WINDOW_BVP) // SHIFT_BVP + 1)
    if len(labels) != expected:
        raise ValueError(
            f"{sid}: label count {len(labels)} != expected {expected} "
            f"from BVP 8s/2s framing"
        )

    # The 32 Hz ACC should represent the same 8/2-second framing.
    expected_acc = max(0, (len(acc) - WINDOW_ACC) // SHIFT_ACC + 1)
    if expected_acc != len(labels):
        raise ValueError(
            f"{sid}: ACC framing count {expected_acc} != label count {len(labels)}"
        )

    # Official activity is 4 Hz. HR windows are 8 s long with a 2 s shift.
    # The exact activity span needed to cover all HR windows is:
    #   (n - 1) * (2 s * 4 Hz) + (8 s * 4 Hz)
    # Some PPG-DaLiA recordings contain a few trailing activity samples beyond
    # that span because the activity stream and physiological streams can have
    # slightly different endpoint conventions. Those trailing samples cannot
    # contribute to any HR window, so we safely ignore them.
    activity_fs = 4
    expected_activity = max(
        0, (len(labels) - 1) * (SHIFT_SEC * activity_fs) + (WINDOW_SEC * activity_fs)
    )
    activity_extra = len(activity) - expected_activity
    if activity_extra < 0:
        raise ValueError(
            f"{sid}: activity length {len(activity)} is shorter than the required "
            f"4 Hz activity span {expected_activity} for {len(labels)} HR windows"
        )
    # Permit only a small endpoint discrepancy. One 2-second activity shift
    # (8 samples at 4 Hz) is a conservative upper bound; larger differences
    # indicate a real framing/schema problem and must not be silently cropped.
    if activity_extra > SHIFT_SEC * activity_fs:
        raise ValueError(
            f"{sid}: activity length {len(activity)} exceeds required span "
            f"{expected_activity} by {activity_extra} samples; this is too large "
            "to treat as an endpoint discrepancy"
        )
    if activity_extra:
        activity = activity[:expected_activity]
        print(
            f"[{sid}] activity has {activity_extra} trailing 4 Hz samples; "
            f"cropping to {expected_activity} for HR-window alignment"
        )

    if not np.all(np.isfinite(bvp)):
        raise ValueError(f"{sid}: BVP contains non-finite values")
    if not np.all(np.isfinite(acc)):
        raise ValueError(f"{sid}: ACC contains non-finite values")
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"{sid}: labels contain non-finite values")

    return bvp.astype(np.float32, copy=False), acc.astype(np.float32, copy=False), labels.astype(np.float32, copy=False), activity


def activity_for_windows(activity, n_windows):
    # Activity is 4 Hz; each HR window is 8 s long and shifts by 2 s.
    # Therefore each HR window maps to a 32-sample activity span, with
    # successive spans starting 8 activity samples apart. Use the midpoint
    # (4 s into the HR window) as the representative activity label.
    activity_fs = 4
    window_activity = WINDOW_SEC * activity_fs
    shift_activity = SHIFT_SEC * activity_fs
    expected = max(0, (n_windows - 1) * shift_activity + window_activity)
    if len(activity) < expected:
        raise ValueError(
            f"Unexpected activity length {len(activity)}; expected at least {expected}"
        )
    if len(activity) > expected:
        # validate_subject() normally crops small trailing endpoint differences.
        # Keep this guard so the mapping function remains safe if called alone.
        if len(activity) - expected > shift_activity:
            raise ValueError(
                f"Unexpected activity length {len(activity)}; expected {expected}"
            )
        activity = activity[:expected]

    midpoint_offset = window_activity // 2
    starts = np.arange(n_windows, dtype=np.int64) * shift_activity
    return activity[starts + midpoint_offset].astype(np.int16, copy=False)


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)

    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")

    discovered = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("S") and p.name[1:].isdigit()
    )
    if args.subjects:
        wanted = set(args.subjects)
        subjects = [p for p in discovered if p.name in wanted]
        missing = wanted - {p.name for p in subjects}
        if missing:
            raise SystemExit(f"Subjects not found: {sorted(missing)}")
    else:
        subjects = discovered

    all_bvp = []
    all_acc = []
    all_hr = []
    all_subject = []
    all_window = []
    all_activity = []

    print("Preparing PPG-DaLiA")
    print("Root:", root.resolve())
    print(f"Window: {WINDOW_SEC}s | shift: {SHIFT_SEC}s")
    print(f"BVP: {BVP_FS} Hz | ACC: {ACC_FS} Hz")

    for sdir in subjects:
        sid = sdir.name
        path = sdir / f"{sid}.pkl"
        print(f"\n[{sid}] loading {path.name} ...")

        obj = load_pickle(path)
        bvp, acc, labels, activity = validate_subject(obj, sid)

        n = len(labels)
        bvp_windows = np.stack(
            [bvp[i * SHIFT_BVP : i * SHIFT_BVP + WINDOW_BVP] for i in range(n)]
        )
        acc_windows = np.stack(
            [acc[i * SHIFT_ACC : i * SHIFT_ACC + WINDOW_ACC] for i in range(n)]
        )

        if args.include_activity:
            act = activity_for_windows(activity, n)
            all_activity.append(act)

        all_bvp.append(bvp_windows)
        all_acc.append(acc_windows)
        all_hr.append(labels)
        all_subject.append(np.full(n, sid))
        all_window.append(np.arange(n, dtype=np.int32))

        print(
            f"[{sid}] windows={n:,} "
            f"BVP={bvp_windows.shape} ACC={acc_windows.shape} "
            f"HR=[{labels.min():.2f}, {labels.max():.2f}]"
        )

        del obj, bvp, acc, labels, activity, bvp_windows, acc_windows
        gc.collect()

    BVP = np.concatenate(all_bvp, axis=0)
    ACC = np.concatenate(all_acc, axis=0)
    HR = np.concatenate(all_hr, axis=0)
    SUBJECT = np.concatenate(all_subject, axis=0)
    WINDOW = np.concatenate(all_window, axis=0)

    payload = {
        "bvp": BVP,
        "acc": ACC,
        "hr": HR,
        "subject": SUBJECT,
        "window_index": WINDOW,
        "bvp_fs": np.array(BVP_FS, dtype=np.int16),
        "acc_fs": np.array(ACC_FS, dtype=np.int16),
        "window_sec": np.array(WINDOW_SEC, dtype=np.int16),
        "shift_sec": np.array(SHIFT_SEC, dtype=np.int16),
    }
    if args.include_activity:
        payload["activity"] = np.concatenate(all_activity, axis=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    print("\n" + "=" * 72)
    print("PREPARATION COMPLETE")
    print("=" * 72)
    print("Output:", out.resolve())
    print("Windows:", len(HR))
    print("BVP:", BVP.shape, BVP.dtype)
    print("ACC:", ACC.shape, ACC.dtype)
    print("HR:", HR.shape, HR.dtype)
    print("Subjects:", sorted(set(SUBJECT)))
    print("Output size MB:", round(out.stat().st_size / 1024 / 1024, 2))


if __name__ == "__main__":
    main()
