"""
SafeBand AI — Stationary-state features v3.

Adds features aimed at separating RESTING from SITTING while retaining
the same accelerometer-first, no-fabricated-synchronization principle.
"""

from __future__ import annotations
import math
from typing import Sequence, Dict

STATIONARY_FEATURES = [
    "gravity_mag_mean","gravity_mag_std","gravity_axis_ratio",
    "vertical_axis_mean_abs","horizontal_energy_ratio",
    "tilt_x_mean","tilt_y_mean","tilt_x_std","tilt_y_std",
    "static_energy_ratio","dynamic_energy_ratio",
]

def stationary_features(samples: Sequence[Sequence[float]]) -> Dict[str,float]:
    if not samples:
        return {k:0.0 for k in STATIONARY_FEATURES}
    ax=[float(x[0]) for x in samples]
    ay=[float(x[1]) for x in samples]
    az=[float(x[2]) for x in samples]
    n=len(ax)
    mx=sum(ax)/n; my=sum(ay)/n; mz=sum(az)/n
    mag=[math.sqrt(x*x+y*y+z*z) for x,y,z in zip(ax,ay,az)]
    mm=sum(mag)/n
    ms=math.sqrt(sum((v-mm)**2 for v in mag)/n)
    axis=[abs(mx),abs(my),abs(mz)]
    dominant=max(axis) if axis else 0.0
    others=sum(v*v for i,v in enumerate(axis) if v!=dominant)
    gravity_axis_ratio=dominant/(math.sqrt(others)+1e-9)
    vertical_axis_mean_abs=abs(mz)
    horizontal_energy_ratio=(mx*mx+my*my)/(mx*mx+my*my+mz*mz+1e-9)

    # Orientation proxies from gravity-dominated accelerometer means.
    tilt_x=math.atan2(my, math.sqrt(mx*mx+mz*mz))
    tilt_y=math.atan2(-mx, math.sqrt(my*my+mz*mz))
    tx=[]; ty=[]
    for x,y,z in zip(ax,ay,az):
        tx.append(math.atan2(y, math.sqrt(x*x+z*z)))
        ty.append(math.atan2(-x, math.sqrt(y*y+z*z)))
    txm=sum(tx)/n; tym=sum(ty)/n
    txs=math.sqrt(sum((v-txm)**2 for v in tx)/n)
    tys=math.sqrt(sum((v-tym)**2 for v in ty)/n)

    # Dynamic energy around the per-window gravity/static mean.
    dyn=sum((x-mx)**2+(y-my)**2+(z-mz)**2 for x,y,z in zip(ax,ay,az))/n
    static=mx*mx+my*my+mz*mz
    return {
        "gravity_mag_mean":mm,
        "gravity_mag_std":ms,
        "gravity_axis_ratio":gravity_axis_ratio,
        "vertical_axis_mean_abs":vertical_axis_mean_abs,
        "horizontal_energy_ratio":horizontal_energy_ratio,
        "tilt_x_mean":txm,"tilt_y_mean":tym,
        "tilt_x_std":txs,"tilt_y_std":tys,
        "static_energy_ratio":static/(static+dyn+1e-9),
        "dynamic_energy_ratio":dyn/(static+dyn+1e-9),
    }
