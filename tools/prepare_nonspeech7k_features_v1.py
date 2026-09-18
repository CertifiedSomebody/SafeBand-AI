import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_nonspeech7k import CLASSES, extract_features

def resolve(p, root):
    p=Path(str(p))
    candidates=[p, root/p, root/"train"/"train"/p.name, root/"test"/"test"/p.name]
    for c in candidates:
        if c.exists(): return c.resolve()
    # final recursive lookup by basename
    hits=list(root.rglob(p.name))
    return hits[0].resolve() if hits else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default="datasets/processed/nonspeech7k/train_clean_manifest.csv")
    ap.add_argument("--data-root",default="datasets/raw/Nonspeech7k")
    ap.add_argument("--out",default="datasets/processed/nonspeech7k/train_features_v1.csv")
    args=ap.parse_args()
    root=Path(args.data_root)
    manifest_path=Path(args.manifest)
    out=Path(args.out)
    if not root.is_absolute(): root=ROOT/root
    if not manifest_path.is_absolute(): manifest_path=ROOT/manifest_path
    if not out.is_absolute(): out=ROOT/out
    m=pd.read_csv(manifest_path)
    if not root.exists(): raise FileNotFoundError(f"Dataset root not found: {root}")
    required={"file_id","label"}
    miss=required-set(m.columns)
    if miss: raise ValueError(f"Manifest missing columns: {sorted(miss)}")
    rows=[]; missing=[]
    for i,r in m.iterrows():
        p=resolve(r.get("audio_path",r.get("path",r.get("filepath",r.get("file_id","")))),root)
        if p is None: missing.append(i); continue
        try:
            feat=extract_features(p)
            feat.update({"file_id":str(r["file_id"]),"label":str(r["label"]),"path":str(p)})
            rows.append(feat)
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed at row {i}: {p}: {e}") from e
    if missing: raise RuntimeError(f"{len(missing)} manifest rows could not resolve to WAV files.")
    out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(out,index=False)
    print(f"[PASS] Wrote {len(df)} feature rows -> {out}")
    print("Class counts:")
    print(df["label"].value_counts().reindex(CLASSES,fill_value=0).to_string())
if __name__=="__main__": main()
