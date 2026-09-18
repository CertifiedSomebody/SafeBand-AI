
"""BNO055-oriented SafeBand synthetic sensor generator.

This simulator produces engineering-unit telemetry matching the semantic
outputs documented in the Bosch BNO055 datasheet. It is NOT a physical
replacement for real BNO055 measurements and must not be used as hardware
validation evidence.
"""
from __future__ import annotations
import math
import numpy as np

FS = 100.0
DT = 1.0 / FS
G = 9.80665

ACTIVITIES = [
    "SITTING", "STANDING", "LYING", "WALKING",
    "RUNNING", "STAIRS", "SIT_TO_STAND", "STAND_TO_SIT", "FALL"
]

# Nominal magnetic field used only to make synthetic magnetometer values
# physically coherent enough for software integration tests.
MAG_FIELD_UT = np.array([20.0, 0.0, 42.0], dtype=float)

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def quat_mul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2
    ])

def quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

def quat_to_euler(q):
    w,x,y,z = quat_normalize(q)
    sinr=2*(w*x+y*z); cosr=1-2*(x*x+y*y)
    roll=math.atan2(sinr, cosr)
    sinp=2*(w*y-z*x)
    pitch=math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
    siny=2*(w*z+x*y); cosy=1-2*(y*y+z*z)
    yaw=math.atan2(siny, cosy)
    return roll, pitch, yaw

def quat_rotate(q, v):
    qv=np.array([0.0, *v])
    return quat_mul(quat_mul(q, qv), np.array([q[0],-q[1],-q[2],-q[3]]))[1:]

def rotation_matrix(q):
    w,x,y,z=quat_normalize(q)
    return np.array([
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]
    ])

def smooth_noise(rng, n, scale, alpha=0.92):
    raw=rng.normal(0, scale, n)
    out=np.empty(n); out[0]=raw[0]
    for i in range(1,n):
        out[i]=alpha*out[i-1]+(1-alpha)*raw[i]
    return out

def activity_motion(activity, t, rng):
    """Return body-frame linear acceleration and angular velocity (dps),
    plus nominal orientation offsets in radians.

    These are intentionally scenario models, not learned distributions.
    """
    n=len(t)
    lin=np.zeros((n,3)); omg=np.zeros((n,3))
    roll=np.zeros(n); pitch=np.zeros(n); yaw=np.zeros(n)

    # small subject/session perturbation
    phase=rng.uniform(0,2*np.pi)
    if activity=="SITTING":
        pitch[:] = math.radians(8)
        roll[:] = math.radians(3)
        lin += rng.normal(0,0.08,(n,3))
        omg += rng.normal(0,0.7,(n,3))
    elif activity=="STANDING":
        pitch[:] = math.radians(2)
        roll[:] = math.radians(1)
        lin += rng.normal(0,0.10,(n,3))
        omg += rng.normal(0,0.8,(n,3))
    elif activity=="LYING":
        roll[:] = math.radians(78)
        pitch[:] = math.radians(4)
        lin += rng.normal(0,0.06,(n,3))
        omg += rng.normal(0,0.45,(n,3))
    elif activity in ("WALKING","RUNNING","STAIRS"):
        f = {"WALKING":1.8,"RUNNING":3.0,"STAIRS":1.6}[activity]
        amp = {"WALKING":1.15,"RUNNING":2.5,"STAIRS":1.5}[activity]
        phase2=2*np.pi*f*t+phase
        lin[:,0] += amp*np.sin(phase2)
        lin[:,1] += 0.25*amp*np.sin(2*phase2+0.7)
        lin[:,2] += 0.45*amp*np.cos(phase2)
        omg[:,1] += (18 if activity=="WALKING" else 35 if activity=="RUNNING" else 25)*np.sin(phase2)
        omg[:,2] += 8*np.sin(2*phase2)
        if activity=="STAIRS":
            lin[:,2] += 0.7*np.maximum(0,np.sin(phase2))
            pitch += math.radians(8)*np.sin(phase2)
    elif activity in ("SIT_TO_STAND","STAND_TO_SIT"):
        dur=max(t[-1],DT)
        u=np.clip(t/dur,0,1)
        # Smooth transition, concentrated in the middle.
        s=3*u*u-2*u*u*u
        ds=6*u*(1-u)/dur
        if activity=="SIT_TO_STAND":
            pitch[:] = math.radians(8)*(1-s)
        else:
            pitch[:] = math.radians(8)*s
        lin[:,2] += 1.8*ds
        omg[:,1] += np.degrees(np.radians(25)*ds)
        lin += rng.normal(0,0.12,(n,3))
        omg += rng.normal(0,1.0,(n,3))
    elif activity=="FALL":
        # Quiet pre-event -> sharp impact -> orientation change -> inactivity.
        lin += rng.normal(0,0.08,(n,3))
        omg += rng.normal(0,0.6,(n,3))
        k=max(2,int(0.35*n))
        impact=min(n-1,int(0.42*n))
        idx=np.arange(n)
        spike=np.exp(-0.5*((idx-impact)/max(1,0.025*n))**2)
        lin[:,2] += -12.0*spike
        lin[:,0] += 8.0*spike
        post=idx>=impact
        roll[post] += math.radians(55)*(1-np.exp(-(idx[post]-impact)/(0.08*n)))
        omg[post,0] += 180*spike[post] + 35*np.exp(-(idx[post]-impact)/(0.12*n))
        # post-fall quiescence
        q=idx>int(0.62*n)
        lin[q]*=0.35; omg[q]*=0.25
    else:
        raise ValueError(activity)

    # gentle orientation drift for realistic non-static telemetry
    drift=0.4*np.sin(0.15*t+phase)
    yaw += np.radians(drift)
    return lin, omg, roll, pitch, yaw

def generate_session(subject_id, session_id, activity, duration, seed):
    rng=np.random.default_rng(seed)
    n=int(round(duration*FS))
    t=np.arange(n)*DT
    lin, omg_dps, r0,p0,y0=activity_motion(activity,t,rng)

    # Subject-specific sensor placement/orientation variation.
    sr=rng.normal(1.0,0.015)
    bias_acc=rng.normal(0,0.025,3)
    bias_gyr=rng.normal(0,0.7,3)
    bias_mag=rng.normal(0,0.8,3)

    q=np.zeros((n,4))
    euler=np.zeros((n,3))
    acc=np.zeros((n,3))
    gyro=np.zeros((n,3))
    mag=np.zeros((n,3))
    grav=np.zeros((n,3))
    lia=np.zeros((n,3))
    temp=np.zeros(n)

    for i in range(n):
        q[i]=quat_normalize(quat_from_euler(r0[i],p0[i],y0[i]))
        R=rotation_matrix(q[i])
        # gravity expressed in sensor frame; acceleration output includes gravity
        g_sensor=R.T@np.array([0,0,G])
        acc[i]=g_sensor + lin[i]*sr + rng.normal(0,0.035,3)+bias_acc
        gyro[i]=omg_dps[i]+bias_gyr+rng.normal(0,0.35,3)
        mag[i]=R.T@MAG_FIELD_UT+bias_mag+rng.normal(0,0.35,3)
        grav[i]=g_sensor
        lia[i]=lin[i]*sr+rng.normal(0,0.025,3)
        euler[i]=np.degrees([r0[i],p0[i],y0[i]])
        temp[i]=27.0 + 0.6*np.sin(2*np.pi*t[i]/60)+rng.normal(0,0.03)

    rows=[]
    for i in range(n):
        rows.append([
            subject_id,session_id,i*DT,activity,
            *acc[i],*gyro[i],*mag[i],
            *euler[i],*q[i],*lia[i],*grav[i],temp[i]
        ])
    return rows
