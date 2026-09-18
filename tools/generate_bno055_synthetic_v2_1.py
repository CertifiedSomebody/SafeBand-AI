"""
SafeBand AI — BNO055 Synthetic Dataset V2.1
FINAL SYNTHETIC GENERATOR

V2.1 is the final planned synthetic-generator revision for the current
software/ML development phase. It is intentionally advanced but bounded.

Design goals:
- 100 Hz BNO055-like raw ACC/GYRO/MAG streams
- subject/session variability
- physically coupled orientation, gravity and magnetometer
- gyro generated as bounded body angular velocity and integrated to attitude
- varied gait frequency/amplitude/phase and cross-axis coupling
- explicit, metadata-marked posture transitions
- explicit fall sequence: pre-motion -> rotation -> impact -> settling
- realistic sensor bias/noise/drift
- magnetic environment independent of activity label
- deterministic seed
- no class-specific fixed sensor orientation
- bounded motion rates to avoid trivial synthetic shortcuts

IMPORTANT:
This is synthetic data for software integration and controlled ML
experimentation. It is NOT validation of real BNO055 hardware performance.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bno055_v2_common import ACTIVITIES

FS = 100.0
G = 9.80665
DEG = np.pi / 180.0


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)


def quat_mul(q1, q2):
    """Hamilton product for [w,x,y,z] quaternions."""
    w1,x1,y1,z1 = np.moveaxis(q1, -1, 0)
    w2,x2,y2,z2 = np.moveaxis(q2, -1, 0)
    return np.stack([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2-z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ], axis=-1)


def quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return quat_normalize(np.stack([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ], axis=-1))


def quat_to_rotmat(q):
    """Body-to-world rotation matrix for [w,x,y,z]."""
    q = quat_normalize(q)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], axis=-1),
        np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], axis=-1),
        np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1),
    ], axis=-2)


def integrate_body_gyro(q0, omega_dps, dt):
    """Integrate body-frame angular velocity into body-to-world attitude."""
    n = len(omega_dps)
    q = np.empty((n,4), dtype=float)
    q[0] = quat_normalize(q0)
    for i in range(1,n):
        omega = omega_dps[i-1] * DEG
        theta = np.linalg.norm(omega) * dt
        if theta < 1e-10:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = omega / np.linalg.norm(omega)
            half = theta / 2.0
            dq = np.concatenate(([np.cos(half)], axis*np.sin(half)))
        q[i] = quat_normalize(quat_mul(q[i-1], dq))
    return q


def smooth_noise(rng, n, scale, block=40):
    if n <= 1:
        return np.zeros(n)
    k = max(2, int(np.ceil(n / block)) + 1)
    xp = np.linspace(0, n-1, k)
    yp = rng.normal(0, scale, k)
    return np.interp(np.arange(n), xp, yp)


def lowpass_random_walk(rng, n, step_scale, block=100):
    return smooth_noise(rng, n, step_scale, block)


def make_subject_params(rng):
    """Stable subject-specific characteristics; activity does not determine them."""
    return {
        "roll": rng.uniform(-35, 35) * DEG,
        "pitch": rng.uniform(-25, 25) * DEG,
        "yaw": rng.uniform(-180, 180) * DEG,
        "acc_bias": rng.normal(0, 0.035, 3),
        "gyro_bias": rng.normal(0, 0.20, 3),
        "mag_scale": rng.uniform(0.94, 1.06, 3),
        "mag_bias": rng.normal(0, 0.8, 3),
        "temp": rng.uniform(24, 30),
        "gait_phase": rng.uniform(0, 2*np.pi),
        "body_field_strength": rng.uniform(38, 52),
        "body_field_inc": rng.uniform(-55, 55) * DEG,
        "body_field_decl": rng.uniform(-180, 180) * DEG,
    }


def base_orientation(params, rng):
    # Every activity gets the same subject-level base orientation distribution.
    jitter = np.array([
        rng.normal(0, 10) * DEG,
        rng.normal(0, 10) * DEG,
        rng.normal(0, 18) * DEG,
    ])
    return quat_from_euler(
        params["roll"] + jitter[0],
        params["pitch"] + jitter[1],
        params["yaw"] + jitter[2],
    )


def posture_reference(activity, params, rng):
    """
    Broad posture priors. These are not fixed class fingerprints because
    subject pose + session jitter dominate the absolute orientation.
    """
    templates = {
        "SITTING": (8, -6, 0),
        "STANDING": (2, -2, 0),
        "LYING": (72, 4, 15),
    }
    rr, pp, yy = templates[activity]
    return quat_from_euler(
        (rr + rng.normal(0, 16))*DEG + params["roll"],
        (pp + rng.normal(0, 16))*DEG + params["pitch"],
        (yy + rng.normal(0, 30))*DEG + params["yaw"],
    )


def q_to_euler_deg(q):
    """ZYX Euler angles in degrees, wrapped heading to [-180,180]."""
    w,x,y,z = np.moveaxis(quat_normalize(q), -1, 0)
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2*(w*y - z*x), -1, 1)
    pitch = np.arcsin(sinp)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return np.column_stack([yaw, roll, pitch]) / DEG


def generate_motion(activity, n, t, rng, params):
    """
    Return bounded body angular velocity [dps] and non-gravitational linear
    acceleration [m/s^2]. The two are correlated, but not identical.
    """
    omega = np.zeros((n,3), dtype=float)
    lin = np.zeros((n,3), dtype=float)

    if activity in ("SITTING", "STANDING", "LYING"):
        # Micro-motion: several weak components + slow drift.
        f1 = rng.uniform(0.07, 0.18)
        f2 = rng.uniform(0.18, 0.32)
        phase = rng.uniform(0, 2*np.pi, 3)
        amps = rng.uniform(0.8, 2.6, 3)
        for k in range(3):
            omega[:,k] = (
                amps[k]*np.sin(2*np.pi*f1*t + phase[k])
                + 0.45*amps[k]*np.sin(2*np.pi*f2*t + phase[k]+0.8)
                + smooth_noise(rng, n, 0.45, 80)
            )
        omega += rng.normal(0, 0.35, omega.shape)

        lin = rng.normal(0, 0.035, (n,3))
        lin += np.column_stack([
            0.08*np.sin(2*np.pi*rng.uniform(.10,.22)*t+rng.uniform(0,6.28)),
            0.07*np.sin(2*np.pi*rng.uniform(.10,.25)*t+rng.uniform(0,6.28)),
            0.06*np.sin(2*np.pi*rng.uniform(.08,.20)*t+rng.uniform(0,6.28)),
        ])

    elif activity in ("WALKING", "RUNNING", "STAIRS"):
        ranges = {
            "WALKING": (1.15, 1.75, 5.0, 18.0, 0.35, 0.95),
            "RUNNING": (2.0, 3.0, 10.0, 38.0, 0.65, 1.45),
            "STAIRS": (0.95, 1.50, 7.0, 30.0, 0.50, 1.25),
        }
        f_lo, f_hi, w_lo, w_hi, a_lo, a_hi = ranges[activity]
        f = rng.uniform(f_lo, f_hi)
        phase = params["gait_phase"] + rng.uniform(-0.8, 0.8)
        w = 2*np.pi*f
        amp = rng.uniform(a_lo, a_hi)

        # Angular-rate gait signature, deliberately bounded.
        base = np.sin(w*t + phase)
        h2 = np.sin(2*w*t + phase*0.7 + 0.9)
        h3 = np.sin(3*w*t + phase*1.1 + 1.8)

        omega[:,0] = (0.65 + rng.uniform(-.12,.12))*w_hi/2*base + 0.18*w_hi*h2
        omega[:,1] = (0.45 + rng.uniform(-.10,.10))*w_hi/2*np.sin(w*t+phase+1.1) + 0.16*w_hi*h2
        omega[:,2] = (0.35 + rng.uniform(-.08,.08))*w_hi/2*np.sin(w*t+phase+2.0) + 0.10*w_hi*h3

        # Normalize each component jointly to requested peak range.
        peak = np.max(np.linalg.norm(omega, axis=1))
        target = rng.uniform(w_lo, w_hi)
        omega *= target / max(peak, 1e-9)

        # Subject-specific gait variation.
        omega += smooth_noise(rng, n, rng.uniform(0.5, 1.5), 60)[:,None] * rng.normal(0,1,(1,3))
        omega += rng.normal(0, 0.45, omega.shape)

        # Linear acceleration: periodic but with different phase/coupling.
        lin[:,0] = amp*(0.48*base + 0.15*h2)
        lin[:,1] = amp*(0.18*np.sin(w*t+phase+1.0) + 0.08*h2)
        lin[:,2] = amp*(0.40*np.sin(w*t+phase+0.5) + 0.13*h2)
        if activity == "STAIRS":
            lin[:,2] += 0.28*amp*np.maximum(0, np.sin(w*t+phase+0.2))**2
        if activity == "RUNNING":
            lin[:,2] += 0.20*amp*np.sin(2*w*t+phase)
        lin += rng.normal(0, 0.07 if activity != "RUNNING" else 0.11, (n,3))

    elif activity in ("SIT_TO_STAND", "STAND_TO_SIT"):
        # Event centered in the recording, with a deterministic event marker.
        event_start = rng.uniform(5.0, 6.0)
        event_dur = rng.uniform(1.6, 2.2)
        event_end = event_start + event_dur
        u = np.clip((t-event_start)/event_dur, 0, 1)
        s = u*u*(3-2*u)
        ds = np.gradient(s, 1/FS)
        d2s = np.gradient(ds, 1/FS)

        # Posture rotation magnitude varies by subject/session.
        sign = 1.0 if activity == "SIT_TO_STAND" else -1.0
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        peak_rate = rng.uniform(35, 65)
        omega += sign * axis[None,:] * peak_rate * np.sin(np.pi*u)[:,None]
        omega += rng.normal(0, 0.65, omega.shape)

        lin[:,0] += sign * rng.uniform(0.15,0.30) * ds
        lin[:,1] += rng.uniform(0.08,0.18) * ds
        lin[:,2] += sign * rng.uniform(0.25,0.55) * d2s / max(np.max(np.abs(d2s)), 1e-9)
        lin += rng.normal(0, 0.055, (n,3))

        # Keep a weak post-event settling oscillation.
        settle = np.exp(-np.maximum(t-event_end, 0)/0.8)
        omega += (2.0*settle*np.sin(2*np.pi*0.8*t))[:,None] * rng.normal(0,1,(1,3))
        event_type = activity
        return omega, lin, event_start, event_end, event_type

    elif activity == "FALL":
        event_start = rng.uniform(6.0, 7.0)
        rot_dur = rng.uniform(0.45, 0.75)
        impact_t = event_start + rot_dur
        settle_dur = rng.uniform(0.7, 1.2)

        u = np.clip((t-event_start)/rot_dur, 0, 1)
        pulse = np.sin(np.pi*u) * (u > 0) * (u < 1)

        axis = rng.normal(0,1,3)
        axis /= np.linalg.norm(axis)
        peak_rate = rng.uniform(140, 220)
        omega += peak_rate * pulse[:,None] * axis[None,:]

        # Small pre-fall destabilization.
        pre = np.clip((t-(event_start-0.7))/0.7, 0, 1)
        omega += (5.0*pre*np.sin(2*np.pi*1.8*t))[:,None] * rng.normal(0,1,(1,3))

        # Impact spike and damped post-impact rotation.
        impact = np.exp(-((t-impact_t)/0.055)**2)
        omega += rng.uniform(20,45) * impact[:,None] * rng.normal(0,1,(1,3))
        post = np.maximum(t-impact_t, 0)
        omega += (
            rng.uniform(5,12)*np.exp(-post/settle_dur)
            * np.sin(2*np.pi*rng.uniform(1.0,2.0)*post)
        )[:,None] * rng.normal(0,1,(1,3))
        omega += rng.normal(0, 0.65, omega.shape)

        # Impact acceleration is multi-axis and short.
        lin += rng.normal(0, 0.06, (n,3))
        lin += rng.normal(0,1,(1,3)) * (rng.uniform(4.5,7.5)*impact)[:,None]
        rebound = np.exp(-((t-(impact_t+0.18))/0.10)**2)
        lin += rng.normal(0,1,(1,3)) * (rng.uniform(1.0,2.5)*rebound)[:,None]
        event_type = "FALL"
        return omega, lin, event_start, impact_t + 0.25, event_type

    else:
        raise ValueError(f"Unsupported activity: {activity}")

    return omega, lin, np.nan, np.nan, ""


def generate_activity(activity, subject, session, duration, rng, params):
    n = int(round(duration * FS))
    if n < 200:
        raise ValueError("duration must be at least 2 seconds for the default window contract.")
    t = np.arange(n, dtype=float) / FS

    q0 = (
        posture_reference(activity, params, rng)
        if activity in ("SITTING","STANDING","LYING")
        else base_orientation(params, rng)
    )

    omega_true, lin, event_start, event_end, event_type = generate_motion(
        activity, n, t, rng, params
    )

    # Integrate the body-frame angular rate. This makes attitude, gravity and
    # magnetometer mutually consistent with the generated gyro.
    q = integrate_body_gyro(q0, omega_true, 1/FS)
    R = quat_to_rotmat(q)

    world_g = np.array([0.0, 0.0, G])
    gravity = np.einsum("nij,j->ni", np.swapaxes(R, -1, -2), world_g)
    accel = gravity + lin + params["acc_bias"] + rng.normal(0, 0.045, (n,3))

    # Measured gyro = true body rate + bias + realistic noise + slow drift.
    gyro_drift = np.column_stack([
        lowpass_random_walk(rng, n, 0.035, 100),
        lowpass_random_walk(rng, n, 0.035, 100),
        lowpass_random_walk(rng, n, 0.035, 100),
    ])
    gyro = omega_true + params["gyro_bias"] + gyro_drift + rng.normal(0, 0.55, (n,3))

    # Magnetic field exists in the world frame and is independent of activity.
    inc = params["body_field_inc"] + rng.normal(0, 2.0)*DEG
    decl = params["body_field_decl"] + rng.normal(0, 5.0)*DEG
    strength = params["body_field_strength"] + rng.normal(0, 1.5)
    mag_world = strength*np.array([
        np.cos(inc)*np.cos(decl),
        np.cos(inc)*np.sin(decl),
        np.sin(inc)
    ])
    mag = np.einsum("nij,j->ni", np.swapaxes(R, -1, -2), mag_world)

    env = np.column_stack([
        smooth_noise(rng,n,0.7,90),
        smooth_noise(rng,n,0.7,90),
        smooth_noise(rng,n,0.7,90),
    ])
    mag = mag*params["mag_scale"] + params["mag_bias"] + env + rng.normal(0,0.55,(n,3))

    euler = q_to_euler_deg(q)
    temp = (
        params["temp"]
        + 0.20*np.sin(2*np.pi*t/max(duration,1))
        + 0.04*smooth_noise(rng,n,0.5,120)
        + rng.normal(0,0.035,n)
    )

    # Re-zero event times are relative to the activity recording.
    return pd.DataFrame({
        "subject_id": subject,
        "session_id": session,
        "timestamp_s": t,
        "activity_label": activity,
        "event_type": event_type,
        "event_start_s": event_start,
        "event_end_s": event_end,
        "accel_x_mps2": accel[:,0], "accel_y_mps2": accel[:,1], "accel_z_mps2": accel[:,2],
        "gyro_x_dps": gyro[:,0], "gyro_y_dps": gyro[:,1], "gyro_z_dps": gyro[:,2],
        "mag_x_uT": mag[:,0], "mag_y_uT": mag[:,1], "mag_z_uT": mag[:,2],
        "euler_heading_deg": euler[:,0], "euler_roll_deg": euler[:,1], "euler_pitch_deg": euler[:,2],
        "quat_w": q[:,0], "quat_x": q[:,1], "quat_y": q[:,2], "quat_z": q[:,3],
        "linear_accel_x_mps2": lin[:,0], "linear_accel_y_mps2": lin[:,1], "linear_accel_z_mps2": lin[:,2],
        "gravity_x_mps2": gravity[:,0], "gravity_y_mps2": gravity[:,1], "gravity_z_mps2": gravity[:,2],
        "temperature_c": temp,
    })


def parse_args():
    p = argparse.ArgumentParser(description="Generate SafeBand BNO055 Synthetic V2.1.")
    p.add_argument("--subjects", type=int, default=20)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=ROOT/"datasets/synthetic/bno055/bno055_synthetic_v2_1.csv")
    return p.parse_args()


def main():
    a = parse_args()
    if a.subjects < 1 or a.sessions < 1 or a.duration < 2:
        raise ValueError("subjects/sessions must be >=1 and duration must be >=2 seconds.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(a.seed)
    frames = []

    for subject in range(1, a.subjects+1):
        for session in range(1, a.sessions+1):
            params = make_subject_params(master)
            for activity in ACTIVITIES:
                seed = int(master.integers(0, 2**63-1))
                rng = np.random.default_rng(seed)
                frames.append(generate_activity(
                    activity, subject, session, a.duration, rng, params
                ))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False, float_format="%.6f")

    print(f"[PASS] Created: {a.out}")
    print(f"[PASS] Rows: {len(df):,}")
    print(f"[PASS] Subjects: {df.subject_id.nunique()}")
    print(f"[PASS] Sessions: {df[['subject_id','session_id']].drop_duplicates().shape[0]}")
    print(f"[PASS] Activities: {df.activity_label.nunique()}")
    print(f"[PASS] Sampling: {FS:.0f} Hz target")
    print(f"[PASS] Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
