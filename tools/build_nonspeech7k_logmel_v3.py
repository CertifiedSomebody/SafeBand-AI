from pathlib import Path
import sys, argparse, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_nonspeech7k_cnn import resolve_audio, load_audio, make_logmel, LABELS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default="datasets/processed/nonspeech7k/train_clean_manifest.csv")
    ap.add_argument("--out",default="datasets/processed/nonspeech7k/logmel_v3.npz")
    args=ap.parse_args()
    df=pd.read_csv(ROOT/args.manifest,dtype={"file_id":str})
    required={"file_id","label","audio_path"}
    missing=required-set(df.columns)
    if missing: raise RuntimeError(f"Manifest missing: {sorted(missing)}")
    label_to_id={v:i for i,v in enumerate(LABELS)}
    X=np.empty((len(df),1,64,128),dtype=np.float32)
    y=np.empty(len(df),dtype=np.int64)
    groups=np.empty(len(df),dtype="U64")
    paths=[]
    for i,row in df.iterrows():
        p=resolve_audio(row)
        x,sr=load_audio(p,16000)
        X[i,0]=make_logmel(x,sr)
        y[i]=label_to_id[str(row.label)]
        groups[i]=str(row.file_id).strip()
        paths.append(str(p.relative_to(ROOT)))
        if (i+1)%500==0: print(f"[INFO] processed {i+1}/{len(df)}")
    if not np.isfinite(X).all(): raise RuntimeError("Nonfinite cached log-mel values")
    if len(set(groups))!=1899: raise RuntimeError("Unexpected group count in cache")
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=X,y=y,groups=groups,labels=np.array(LABELS),audio_paths=np.array(paths,dtype=str))
    print(f"[PASS] cache: {out}")
    print(f"[PASS] X shape: {X.shape}")
    print(f"[PASS] groups: {len(set(groups))}")
    print(pd.Series([LABELS[i] for i in y]).value_counts().to_string())

if __name__=="__main__": main()
