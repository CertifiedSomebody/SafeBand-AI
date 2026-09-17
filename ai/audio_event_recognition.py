"""SafeBand audio-event inference layer.

Audio event recognition is optional and model-driven. Volume/acoustic level
alone is never treated as a semantic emergency event.
"""
from __future__ import annotations
from typing import Any,Dict,Optional,Sequence
from ai.ml_audio_model import MLAudioModel
from config.settings import AI_AUDIO_MODEL_ENABLED,AI_AUDIO_MODEL_PATH,AI_AUDIO_CONFIDENCE_THRESHOLD,AI_AUDIO_MIN_WINDOW_SAMPLES

_model=MLAudioModel(AI_AUDIO_MODEL_PATH,AI_AUDIO_MIN_WINDOW_SAMPLES)

def recognize_audio(samples: Optional[Sequence[float]], sample_rate_hz: float = 16000.0, fallback_level: Optional[float]=None)->Dict[str,Any]:
    if not samples:
        level=float(fallback_level or 0.0)
        return {"event":"UNKNOWN","confidence":0.0,"source":"LEVEL_ONLY","audio_level":level,"model_available":_model.available}
    if not AI_AUDIO_MODEL_ENABLED:
        return {"event":"UNAVAILABLE","confidence":0.0,"source":"DISABLED","audio_level":float(fallback_level or 0.0),"model_available":_model.available}
    try:
        event,conf,probs=_model.predict(samples,sample_rate_hz)
        if conf < AI_AUDIO_CONFIDENCE_THRESHOLD:
            event="UNKNOWN"
        return {"event":event,"confidence":conf,"probabilities":probs,"source":"ML","model_available":True,
                "model_name":_model.model_name,"model_version":_model.model_version,"window_samples":len(samples)}
    except Exception as exc:
        return {"event":"UNKNOWN","confidence":0.0,"source":"FALLBACK","model_available":_model.available,"error":str(exc)}
