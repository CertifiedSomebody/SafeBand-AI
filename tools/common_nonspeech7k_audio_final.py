from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf

def load_audio(path,target_sr=16000):
    x,sr=sf.read(path,always_2d=False); x=np.asarray(x,dtype=np.float32)
    if x.ndim==2: x=x.mean(axis=1)
    if x.ndim!=1 or x.size==0 or not np.isfinite(x).all(): raise ValueError(f'Invalid audio: {path}')
    if sr!=target_sr:
        import librosa
        x=librosa.resample(x,orig_sr=sr,target_sr=target_sr).astype(np.float32); sr=target_sr
    peak=float(np.max(np.abs(x)))
    if peak>1.0: x/=max(peak,1e-8)
    return x,int(sr)

def make_logmel(x,sr):
    import librosa
    S=librosa.feature.melspectrogram(y=x,sr=sr,n_fft=512,hop_length=160,n_mels=64,fmin=50,fmax=min(7600,sr//2),power=2.0)
    S=np.clip(librosa.power_to_db(S,ref=np.max),-80.0,0.0)
    if S.shape[1]<128: S=np.pad(S,((0,0),(0,128-S.shape[1])),constant_values=-80.0)
    elif S.shape[1]>128:
        start=(S.shape[1]-128)//2; S=S[:,start:start+128]
    if S.shape!=(64,128) or not np.isfinite(S).all(): raise RuntimeError(f'Bad logmel shape/data: {S.shape}')
    return S.astype(np.float32)
