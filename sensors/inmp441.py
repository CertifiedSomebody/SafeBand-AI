"""SAFEBAND AI - INMP441 microphone interface.

This module is an acquisition adapter, not an emergency classifier.
The Python prototype currently simulates audio. Real ESP32-S3 I2S capture
will later populate the same fields.

Returned fields distinguish:
- signal measurements: RMS/peak/audio_level
- acquisition metadata: sample rate/frame length/clipping
- semantic AI output: optional audio_event/audio_event_confidence
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Dict,Optional,Sequence
import random
import numpy as np

SENSOR_NAME="INMP441"; SENSOR_TYPE="Audio"; INTERFACE="I2S"
DEFAULT_SAMPLE_RATE_HZ=16000
DEFAULT_FRAME_SAMPLES=3200
DEFAULT_AUDIO_LEVEL=0.08
MIN_AUDIO_LEVEL=0.0; MAX_AUDIO_LEVEL=1.0
QUIET_THRESHOLD=0.20; NORMAL_THRESHOLD=0.50; LOUD_THRESHOLD=0.75

@dataclass
class INMP441Reading:
    audio_level: float
    acoustic_state: str
    connected: bool
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    frame_samples: int = DEFAULT_FRAME_SAMPLES
    rms: float = 0.0
    peak: float = 0.0
    clipped: bool = False
    simulated: bool = True

class INMP441Sensor:
    def __init__(self,simulation:bool=True,sample_rate_hz:int=DEFAULT_SAMPLE_RATE_HZ,frame_samples:int=DEFAULT_FRAME_SAMPLES)->None:
        self.simulation=bool(simulation); self.connected=False
        self.sample_rate_hz=int(sample_rate_hz); self.frame_samples=int(frame_samples)
        self.audio_level=DEFAULT_AUDIO_LEVEL; self.rms=DEFAULT_AUDIO_LEVEL
        self.peak=min(1.0,DEFAULT_AUDIO_LEVEL*2); self.clipped=False; self.acoustic_state="QUIET"
    def connect(self)->bool:
        if self.simulation: self.connected=True; return True
        # Real ESP32-S3 I2S acquisition belongs in firmware. Python returns
        # False here rather than pretending that desktop I2S is connected.
        self.connected=False; return False
    def disconnect(self)->None: self.connected=False
    def _ensure_connected(self):
        if not self.connected: self.connect()
    def read_audio_level(self)->float:
        self._ensure_connected()
        if self.simulation:
            self.audio_level=max(MIN_AUDIO_LEVEL,min(MAX_AUDIO_LEVEL,self.audio_level+random.uniform(-.04,.04)))
        return round(self.audio_level,3)
    @staticmethod
    def classify_audio(audio_level:float)->str:
        try: level=float(audio_level)
        except (TypeError,ValueError): level=0.0
        level=max(0.0,min(1.0,level))
        if level<=QUIET_THRESHOLD:return "QUIET"
        if level<=NORMAL_THRESHOLD:return "NORMAL"
        if level<=LOUD_THRESHOLD:return "LOUD"
        return "VERY_LOUD"
    def read_frame(self,samples:Optional[Sequence[float]]=None)->np.ndarray:
        """Return a normalized PCM frame. Hardware integration can replace this."""
        self._ensure_connected()
        if samples is not None:
            x=np.asarray(samples,dtype=np.float32).reshape(-1)
        elif self.simulation:
            x=np.random.default_rng().normal(0.0,max(.005,self.audio_level*.25),self.frame_samples).astype(np.float32)
        else:
            raise RuntimeError("Real INMP441 frame acquisition is not implemented in the Python prototype.")
        if x.size==0: raise ValueError("Empty audio frame")
        self.rms=float(np.sqrt(np.mean(x*x))); self.peak=float(np.max(np.abs(x)))
        self.clipped=bool(np.any(np.abs(x)>=.999))
        self.audio_level=float(max(0.0,min(1.0,self.rms*4.0)))
        self.acoustic_state=self.classify_audio(self.audio_level)
        return x
    def read(self)->INMP441Reading:
        x=self.read_frame()
        return INMP441Reading(audio_level=round(self.audio_level,3),acoustic_state=self.acoustic_state,
            connected=self.connected,sample_rate_hz=self.sample_rate_hz,frame_samples=len(x),
            rms=round(self.rms,6),peak=round(self.peak,6),clipped=self.clipped,simulated=self.simulation)
    def read_dict(self)->Dict[str,Any]:
        r=self.read()
        return {"audio_level":r.audio_level,"acoustic_state":r.acoustic_state,
                "audio_rms":r.rms,"audio_peak":r.peak,"audio_clipped":r.clipped,
                "audio_sample_rate_hz":r.sample_rate_hz,"audio_frame_samples":r.frame_samples,
                "microphone_connected":r.connected,"simulated":r.simulated}
    def get_status(self)->Dict[str,Any]:
        return {"sensor":SENSOR_NAME,"name":"MEMS Microphone","type":SENSOR_TYPE,"interface":INTERFACE,
                "connected":self.connected,"simulation":self.simulation,"sample_rate_hz":self.sample_rate_hz,
                "frame_samples":self.frame_samples,"audio_level":round(self.audio_level,3),
                "audio_rms":round(self.rms,6),"audio_peak":round(self.peak,6),
                "clipped":self.clipped,"acoustic_state":self.acoustic_state}
    def reset(self)->None:
        self.audio_level=DEFAULT_AUDIO_LEVEL; self.rms=DEFAULT_AUDIO_LEVEL; self.peak=DEFAULT_AUDIO_LEVEL*2
        self.clipped=False; self.acoustic_state="QUIET"

_inmp441=INMP441Sensor(simulation=True)
def initialize_inmp441()->bool:return _inmp441.connect()
def read_inmp441()->Dict[str,Any]:return _inmp441.read_dict()
def get_inmp441_status()->Dict[str,Any]:return _inmp441.get_status()
def reset_inmp441()->None:_inmp441.reset()
__all__=["INMP441Reading","INMP441Sensor","initialize_inmp441","read_inmp441","get_inmp441_status","reset_inmp441"]
