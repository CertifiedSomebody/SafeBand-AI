from pathlib import Path
import sys, argparse, pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.common_nonspeech7k_tf import resolve_audio, load_mono, logmel_mfcc_features

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", default="datasets/processed/nonspeech7k/train_clean_manifest.csv")
    ap.add_argument("--out", default="datasets/processed/nonspeech7k/train_tf_features_v2.csv")
    args=ap.parse_args()
    manifest=ROOT/args.manifest
    df=pd.read_csv(manifest)
    required={"file_id","label"}
    missing=required-set(df.columns)
    if missing: raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")
    if "audio_path" not in df.columns and "path" not in df.columns and "filepath" not in df.columns:
        raise RuntimeError("Manifest must contain audio_path/path/filepath; refusing basename guessing.")
    rows=[]; failures=[]
    for i,row in df.iterrows():
        try:
            p=resolve_audio(row)
            x,sr=load_mono(p,16000)
            f=logmel_mfcc_features(x,sr)
            f.update({"file_id":str(row["file_id"]), "label":str(row["label"]), "audio_path":str(p.relative_to(ROOT))})
            rows.append(f)
        except Exception as e:
            failures.append((i,str(row["file_id"]),str(e)))
    if failures:
        raise RuntimeError(f"{len(failures)} audio rows failed. First failure: {failures[0]}")
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True)
    out_df=pd.DataFrame(rows)
    out_df.to_csv(out,index=False)
    print(f"[PASS] wrote {len(out_df)} feature rows -> {out}")
    print(out_df["label"].value_counts().to_string())
    print(f"[PASS] feature columns: {len(out_df.columns)-3}")

if __name__=="__main__": main()
