import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_nonspeech7k import CLASSES

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default="datasets/processed/nonspeech7k/train_clean_manifest.csv")
    args=ap.parse_args()
    p=Path(args.manifest)
    if not p.exists(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    print(f"[PASS] manifest: {p}")
    print(f"[PASS] rows: {len(df)}")
    print(f"[PASS] unique file_id: {df.file_id.nunique()}")
    print(f"[PASS] classes: {sorted(df.label.unique())}")
    bad=set(df.label)-set(CLASSES)
    if bad: raise ValueError(f"Unknown labels: {bad}")
    print("[PASS] labels valid")
    print(df.label.value_counts().reindex(CLASSES,fill_value=0).to_string())
    print("[PASS] feature benchmark input manifest ready.")
if __name__=="__main__": main()
