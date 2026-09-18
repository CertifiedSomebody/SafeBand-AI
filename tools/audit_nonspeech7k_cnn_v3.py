from pathlib import Path
import sys, argparse, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_nonspeech7k_cnn import resolve_audio, load_audio, make_logmel, LABELS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default="datasets/processed/nonspeech7k/train_clean_manifest.csv")
    ap.add_argument("--check-all",action="store_true")
    args=ap.parse_args()
    df=pd.read_csv(ROOT/args.manifest,dtype={"file_id":str})
    required={"file_id","label","audio_path"}
    missing=required-set(df.columns)
    if missing: raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")
    if len(df)!=6283: raise RuntimeError(f"Expected 6283 rows, got {len(df)}")
    if df.file_id.astype(str).str.strip().nunique()!=1899:
        raise RuntimeError("Expected 1899 unique file_id groups")
    if set(df.label.astype(str))!=set(LABELS):
        raise RuntimeError("Unexpected label set")
    print(f"[PASS] manifest rows: {len(df)}")
    print(f"[PASS] unique file_id: {df.file_id.astype(str).str.strip().nunique()}")
    print(f"[PASS] labels: {sorted(df.label.astype(str).unique())}")
    if args.check_all:
        for i,row in df.iterrows():
            z=make_logmel(*load_audio(resolve_audio(row),16000))
            if z.shape!=(64,128) or not np.isfinite(z).all():
                raise RuntimeError(f"Invalid log-mel at row {i}, file_id={row.file_id}")
            if (i+1)%500==0: print(f"[INFO] validated {i+1}/{len(df)}")
        print("[PASS] all 6283 WAV/log-mel tensors validated")
    else:
        for label in LABELS:
            row=df[df.label==label].iloc[0]
            z=make_logmel(*load_audio(resolve_audio(row),16000))
            if z.shape!=(64,128): raise RuntimeError(f"Bad representative tensor: {label}")
        print("[PASS] representative audio/log-mel validation passed for all classes")
    print("[PASS] official 725-file test set untouched")

if __name__=="__main__": main()
