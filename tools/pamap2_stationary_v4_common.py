"""
SafeBand AI — PAMAP2 Stationary V4 common utilities.
All paths are repository-root relative.
"""
from pathlib import Path
import sys, random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT/"datasets"/"processed"/"pamap2"/"stationary_accgyro_windows.npz"
OUT = ROOT/"models"/"pamap2_stationary_v4"
CLASSES = ["LYING","SITTING","STANDING"]

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)

def load_dataset(path=DATA):
    path = Path(path)
    if not path.is_absolute(): path = ROOT/path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Expected: datasets\\processed\\pamap2\\stationary_accgyro_windows.npz"
        )
    z=np.load(path,allow_pickle=True)
    missing={"X","y","groups"}-set(z.files)
    if missing: raise ValueError(f"Missing NPZ keys: {sorted(missing)}")
    X=np.asarray(z["X"],dtype=np.float32)
    y=np.asarray(z["y"],dtype=np.int64)
    g=np.asarray(z["groups"]).astype(str)
    if X.ndim!=3 or X.shape[1]!=6: raise ValueError(f"Expected X=(N,6,T), got {X.shape}")
    if not(len(X)==len(y)==len(g)): raise ValueError("X/y/groups length mismatch")
    if sorted(np.unique(y).tolist()) != [0,1,2]:
        raise ValueError(f"Expected labels 0,1,2; got {np.unique(y)}")
    if not np.isfinite(X).all(): raise ValueError("X contains NaN/Inf")
    return X,y,g

def handcrafted_features(X):
    """94-feature stationary representation: axis stats + magnitudes + gravity direction + ACC correlations."""
    acc=X[:,:3,:]; gyro=X[:,3:,:]
    am=np.sqrt(np.sum(acc**2,axis=1)); gm=np.sqrt(np.sum(gyro**2,axis=1))
    out=[]
    for sig in (acc,gyro,am[:,None,:],gm[:,None,:]):
        for i in range(sig.shape[1]):
            a=sig[:,i,:]
            out += [a.mean(1),a.std(1),a.min(1),a.max(1),np.sqrt(np.mean(a*a,axis=1)),
                    np.percentile(a,10,axis=1),np.percentile(a,25,axis=1),
                    np.percentile(a,50,axis=1),np.percentile(a,75,axis=1),
                    np.percentile(a,90,axis=1),np.ptp(a,axis=1)]
    m=acc.mean(axis=2); n=np.linalg.norm(m,axis=1)+1e-8
    out += [m[:,0]/n,m[:,1]/n,m[:,2]/n]
    for i,j in ((0,1),(0,2),(1,2)):
        ai,aj=acc[:,i,:],acc[:,j,:]
        ci=ai-ai.mean(1,keepdims=True); cj=aj-aj.mean(1,keepdims=True)
        out.append((ci*cj).mean(1)/(ai.std(1)*aj.std(1)+1e-8))
    F=np.column_stack(out).astype(np.float32)
    if F.shape[1]!=94: raise RuntimeError(f"Feature contract failure: {F.shape[1]} != 94")
    return F

def standardize_features(train,test):
    mu=train.mean(axis=0,keepdims=True); sd=train.std(axis=0,keepdims=True)
    sd=np.where(sd<1e-6,1,sd)
    return ((train-mu)/sd).astype(np.float32),((test-mu)/sd).astype(np.float32)
