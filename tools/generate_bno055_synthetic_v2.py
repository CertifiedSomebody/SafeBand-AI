"""
SafeBand AI — BNO055 Synthetic Dataset V2

V2 goals:
- 100 Hz BNO055-like output streams
- subject/session variability
- randomized wrist orientation
- realistic-ish periodic gait variation
- physically coupled gravity/linear acceleration
- gyro derived from orientation dynamics
- magnetometer generated from a world field, independent of activity label
- explicit transition and fall temporal events
- deterministic generation from a seed

This is synthetic integration/benchmark data, not hardware validation.
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
ACTIVITIES = ACTIVITIES

def rotmat_from_euler(roll, pitch, yaw):
    """World-to-body rotation for ZYX yaw-pitch-roll convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R_bw = Rz(yaw) Ry(pitch) Rx(roll); body vector = R_bw.T @ world
    r00 = cy*cp
    r01 = cy*sp*sr - sy*cr
    r02 = cy*sp*cr + sy*sr
    r10 = sy*cp
    r11 = sy*sp*sr + cy*cr
    r12 = sy*sp*cr - cy*sr
    r20 = -sp
    r21 = cp*sr
    r22 = cp*cr
    return np.stack([
        np.stack([r00,r01,r02], axis=-1),
        np.stack([r10,r11,r12], axis=-1),
        np.stack([r20,r21,r22], axis=-1),
    ], axis=-2)

def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    q = np.stack([w,x,y,z], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q

def smooth_noise(rng, n, scale, block=20):
    """Piecewise-linear low-frequency noise."""
    if n <= 1:
        return np.zeros(n)
    k = max(2, int(np.ceil(n / block)) + 1)
    xp = np.linspace(0, n-1, k)
    yp = rng.normal(0, scale, k)
    return np.interp(np.arange(n), xp, yp)

def make_subject_params(rng):
    # Per-person wrist pose and sensor characteristics.
    return {
        "roll": rng.uniform(-25,25),
        "pitch": rng.uniform(-20,20),
        "yaw": rng.uniform(-180,180),
        "acc_bias": rng.normal(0, 0.035, 3),
        "gyro_bias": rng.normal(0, 0.20, 3),
        "mag_scale": rng.uniform(0.92,1.08,3),
        "temp": rng.uniform(24,30),
        "gait_phase": rng.uniform(0,2*np.pi),
    }

def posture_angles(activity, p, rng):
    # Templates are intentionally broad; random wrist offsets prevent
    # one fixed orientation from becoming a class identifier.
    templates = {
        "SITTING":  (8, -6, 0),
        "STANDING": (2, -2, 0),
        "LYING":    (72, 4, 15),
    }
    r,pit,y = templates[activity]
    return (
        r + p["roll"] + rng.normal(0, 14),
        pit + p["pitch"] + rng.normal(0, 14),
        y + p["yaw"] + rng.normal(0, 25),
    )

def generate_activity(activity, subject, session, duration, rng, params):
    n = int(round(duration*FS))
    t = np.arange(n)/FS
    roll0, pitch0, yaw0 = posture_angles(
        activity if activity in ("SITTING","STANDING","LYING") else
        ("STANDING" if activity in ("WALKING","RUNNING","STAIRS") else "SITTING"),
        params, rng
    )
    r = np.full(n, np.deg2rad(roll0))
    p = np.full(n, np.deg2rad(pitch0))
    y = np.full(n, np.deg2rad(yaw0))

    lin = np.zeros((n,3), dtype=float)

    if activity in ("SITTING","STANDING","LYING"):
        # Quiet but nonzero wrist micro-motion.
        for k, arr in enumerate((r,p,y)):
            arr += np.deg2rad(
                1.2*np.sin(2*np.pi*rng.uniform(.08,.22)*t+rng.uniform(0,2*np.pi))
                + smooth_noise(rng,n,0.7,30)
            )
        lin += rng.normal(0,0.045,(n,3))
        lin += np.column_stack([
            0.08*np.sin(2*np.pi*0.15*t+rng.uniform(0,6.28)),
            0.06*np.sin(2*np.pi*0.19*t+rng.uniform(0,6.28)),
            0.05*np.sin(2*np.pi*0.12*t+rng.uniform(0,6.28)),
        ])

    elif activity in ("WALKING","RUNNING","STAIRS"):
        freq = {"WALKING":rng.uniform(1.25,1.85),
                "RUNNING":rng.uniform(2.2,3.2),
                "STAIRS":rng.uniform(1.05,1.65)}[activity]
        amp = {"WALKING":rng.uniform(.7,1.25),
               "RUNNING":rng.uniform(1.5,2.6),
               "STAIRS":rng.uniform(1.0,1.9)}[activity]
        phase = params["gait_phase"] + rng.uniform(-.7,.7)
        w = 2*np.pi*freq
        # Multiple harmonics and cross-axis motion.
        lin[:,0] = amp*(0.55*np.sin(w*t+phase)+0.18*np.sin(2*w*t+phase*0.7))
        lin[:,1] = amp*(0.20*np.sin(w*t+phase+1.1)+0.10*np.sin(2*w*t+0.3))
        lin[:,2] = amp*(0.45*np.sin(w*t+phase+0.5)+0.16*np.sin(2*w*t+1.4))
        if activity == "STAIRS":
            # Asymmetric step-like vertical component.
            lin[:,2] += 0.35*amp*np.maximum(0, np.sin(w*t+phase))**2
        lin += rng.normal(0,0.10 if activity=="RUNNING" else 0.07,(n,3))
        # Wrist orientation follows gait but not identically to acceleration.
        r += np.deg2rad((4 if activity=="WALKING" else 7)*np.sin(w*t+phase+.8))
        p += np.deg2rad((3 if activity=="WALKING" else 5)*np.sin(w*t+phase+1.7))
        y += np.deg2rad((5 if activity=="WALKING" else 8)*np.sin(w*t+phase+2.1))
        # Slow drift / natural posture changes.
        r += np.deg2rad(smooth_noise(rng,n,2.0,60))
        p += np.deg2rad(smooth_noise(rng,n,1.5,60))
        y += np.deg2rad(smooth_noise(rng,n,2.5,60))

    elif activity in ("SIT_TO_STAND","STAND_TO_SIT"):
        start, end = ("SITTING","STANDING") if activity=="SIT_TO_STAND" else ("STANDING","SITTING")
        a0 = posture_angles(start, params, rng)
        a1 = posture_angles(end, params, rng)
        # Transition occupies a random central 1.4–2.4 seconds.
        t0 = rng.uniform(3.5,5.0)
        dur = rng.uniform(1.4,2.4)
        u = np.clip((t-t0)/dur,0,1)
        # Smoothstep gives a continuous transition with realistic acceleration.
        s = u*u*(3-2*u)
        for j, arr in enumerate((r,p,y)):
            arr[:] = np.deg2rad(a0[j] + (a1[j]-a0[j])*s)
        lin += rng.normal(0,0.05,(n,3))
        lin[:,2] += 0.65*np.gradient(np.gradient(s,1/FS),1/FS)
        lin[:,0] += 0.22*np.gradient(s,1/FS)
        lin += rng.normal(0,0.04,(n,3))
        # Small post-transition settling.
        settling = np.maximum(0, 1-u)
        r += np.deg2rad(1.5*np.sin(2*np.pi*.8*t)*settling)
        p += np.deg2rad(1.2*np.sin(2*np.pi*.7*t+.5)*settling)

    elif activity == "FALL":
        # Explicit sequence: normal -> rotation/acceleration -> impact -> rest.
        event = rng.uniform(5.0,7.0)
        pre = np.clip((t-(event-0.9))/0.9,0,1)
        impact = np.exp(-((t-event)/0.10)**2)
        post = np.clip((t-event)/1.0,0,1)
        # Rapid multi-axis orientation change.
        direction = rng.normal(0,1,3)
        direction /= np.linalg.norm(direction)
        angle = np.deg2rad(rng.uniform(70,150))
        smooth = pre*pre*(3-2*pre)
        r += direction[0]*angle*smooth
        p += direction[1]*angle*smooth
        y += direction[2]*angle*smooth
        # Acceleration disturbance + impact impulse.
        lin[:,0] += direction[0]*(5.0*impact + 1.2*np.sin(2*np.pi*4*t)*impact)
        lin[:,1] += direction[1]*(4.0*impact)
        lin[:,2] += direction[2]*(6.0*impact)
        lin += rng.normal(0,0.08,(n,3))
        # Post-fall settling.
        r += np.deg2rad(4*np.sin(2*np.pi*.9*t)*np.exp(-post*2))
        p += np.deg2rad(3*np.sin(2*np.pi*.7*t+.4)*np.exp(-post*2))

    # Gravity in body frame from orientation.
    R = rotmat_from_euler(r,p,y)
    world_g = np.array([0.0,0.0,G])
    gravity = np.einsum("nij,j->ni", np.swapaxes(R,-1,-2), world_g)

    # Raw accelerometer = gravity + non-gravitational linear acceleration.
    accel = gravity + lin + params["acc_bias"] + rng.normal(0,0.045,(n,3))

    # Euler derivatives are an approximation to angular velocity; add
    # cross-axis coupling, bias and sensor noise.
    e = np.column_stack([r,p,y])
    gyro = np.rad2deg(np.gradient(e, 1/FS, axis=0))
    coupling = np.column_stack([
        0.08*gyro[:,1] + 0.03*gyro[:,2],
        0.06*gyro[:,0] + 0.04*gyro[:,2],
        0.05*gyro[:,0] + 0.04*gyro[:,1],
    ])
    gyro = gyro + coupling + params["gyro_bias"] + rng.normal(0,0.55,(n,3))

    # A world magnetic vector varies by subject/session/environment, not label.
    mag_strength = rng.uniform(38,52)
    inc = np.deg2rad(rng.uniform(-55,55))
    decl = np.deg2rad(rng.uniform(-180,180))
    mag_world = mag_strength*np.array([
        np.cos(inc)*np.cos(decl),
        np.cos(inc)*np.sin(decl),
        np.sin(inc)
    ])
    mag = np.einsum("nij,j->ni", np.swapaxes(R,-1,-2), mag_world)
    # Slowly changing environmental field + measurement noise.
    env = np.column_stack([
        smooth_noise(rng,n,0.9,80),
        smooth_noise(rng,n,0.9,80),
        smooth_noise(rng,n,0.9,80),
    ])
    mag = mag*params["mag_scale"] + env + rng.normal(0,0.55,(n,3))

    q = euler_to_quaternion(r,p,y)
    temp = params["temp"] + 0.25*np.sin(2*np.pi*t/duration) + rng.normal(0,0.035,n)

    return pd.DataFrame({
        "subject_id": subject,
        "session_id": session,
        "timestamp_s": t,
        "activity_label": activity,
        "accel_x_mps2": accel[:,0], "accel_y_mps2": accel[:,1], "accel_z_mps2": accel[:,2],
        "gyro_x_dps": gyro[:,0], "gyro_y_dps": gyro[:,1], "gyro_z_dps": gyro[:,2],
        "mag_x_uT": mag[:,0], "mag_y_uT": mag[:,1], "mag_z_uT": mag[:,2],
        "euler_heading_deg": np.rad2deg(y), "euler_roll_deg": np.rad2deg(r),
        "euler_pitch_deg": np.rad2deg(p),
        "quat_w": q[:,0], "quat_x": q[:,1], "quat_y": q[:,2], "quat_z": q[:,3],
        "linear_accel_x_mps2": lin[:,0], "linear_accel_y_mps2": lin[:,1], "linear_accel_z_mps2": lin[:,2],
        "gravity_x_mps2": gravity[:,0], "gravity_y_mps2": gravity[:,1], "gravity_z_mps2": gravity[:,2],
        "temperature_c": temp,
    })

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, default=20)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT/"datasets/synthetic/bno055/bno055_synthetic_v2.csv")
    return p.parse_args()

def main():
    a = parse_args()
    if a.subjects < 1 or a.sessions < 1 or a.duration <= 0:
        raise ValueError("subjects/sessions must be >=1 and duration must be >0")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(a.seed)
    frames=[]
    for subject in range(1,a.subjects+1):
        for session in range(1,a.sessions+1):
            params=make_subject_params(master)
            for activity in ACTIVITIES:
                seed=int(master.integers(0,2**63-1))
                rng=np.random.default_rng(seed)
                frames.append(generate_activity(activity,subject,session,a.duration,rng,params))
    df=pd.concat(frames,ignore_index=True)
    df.to_csv(a.out,index=False,float_format="%.6f")
    print(f"[PASS] Created: {a.out}")
    print(f"[PASS] Rows: {len(df):,}")
    print(f"[PASS] Subjects: {df.subject_id.nunique()}")
    print(f"[PASS] Sessions: {df[['subject_id','session_id']].drop_duplicates().shape[0]}")
    print(f"[PASS] Activities: {df.activity_label.nunique()}")
    print(f"[PASS] Sampling: 100 Hz target")
    print(f"[PASS] Columns: {len(df.columns)}")

if __name__=="__main__":
    main()
