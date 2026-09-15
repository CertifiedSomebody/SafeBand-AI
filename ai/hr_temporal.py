from collections import deque
import numpy as np
class HRTemporalFusion:
    def __init__(self,history=3,max_change_bpm_per_sec=12,min_hr=42,max_hr=180):
        self.history=deque(maxlen=history);self.last=None
        self.max_rate=float(max_change_bpm_per_sec);self.lo=float(min_hr);self.hi=float(max_hr)
    def update(self,ml_hr,candidate_hr=np.nan,quality=0.0,dt_sec=2.0,motion=0.0):
        if not np.isfinite(ml_hr):return self.last
        ml=float(np.clip(ml_hr,self.lo,self.hi));q=float(np.clip(quality,0,1))
        if np.isfinite(candidate_hr):
            c=float(np.clip(candidate_hr,self.lo,self.hi))
            w=.20*q*np.exp(-.35*max(float(motion),0))
            x=(1-w)*ml+w*c
        else:x=ml
        if self.last is not None:
            step=self.max_rate*max(float(dt_sec),.1)
            x=self.last+np.sign(x-self.last)*min(abs(x-self.last),step)
        self.history.append(x);self.last=float(np.median(self.history));return self.last
    def reset(self):self.history.clear();self.last=None
