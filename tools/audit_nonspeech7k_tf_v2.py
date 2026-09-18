from pathlib import Path
import sys, argparse, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_nonspeech7k_tf import resolve_audio, load_mono

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default="datasets/processed/nonspeech7k/train_clean_manifest.csv")
    args=ap.parse_args()
    df=pd.read_csv(ROOT/args.manifest)
    required={"file_id","label","audio_path"}
    missing=required-set(df.columns)
    if missing: raise RuntimeError(f"Manifest missing: {sorted(missing)}")
    assert len(df)==6283, f"Expected clean train 6283, got {len(df)}"
    assert df.file_id.nunique()==1899, f"Expected 1899 unique file_id, got {df.file_id.nunique()}"
    assert set(df.label)=={"breath","cough","crying","laugh","screaming","sneeze","yawn"}
    bad=0; sr_counts={}; durations=[]
    for i,row in df.iterrows():
        try:
            p=resolve_audio(row); x,sr=load_mono(p,16000)
            sr_counts[sr]=sr_counts.get(sr,0)+1; durations.append(len(x)/sr)
        except Exception as e:
            bad+=1
            if bad==1: print("[FAIL] first audio:",e)
    if bad: raise RuntimeError(f"{bad} WAV rows failed validation.")
    print(f"[PASS] rows: {len(df)}")
    print(f"[PASS] unique file_id: {df.file_id.nunique()}")
    print(f"[PASS] WAVs resolved: {len(df)}")
    print(f"[PASS] resampled target rate: 16000 Hz")
    print(f"[PASS] original-rate counts: {sr_counts}")
    print(f"[PASS] duration min/median/max: {min(durations):.3f}/{np.median(durations):.3f}/{max(durations):.3f}s")
if __name__=="__main__": main()
