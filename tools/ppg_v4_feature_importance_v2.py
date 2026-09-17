#!/usr/bin/env python3
"""V4.2 feature importance on the validation set only.

Permutation importance is computed only on validation data, after the model
has been trained on the training subjects. The held-out test set is never read.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

META={"subject","activity","record","start_sample","start_time","hr_bpm"}

def subject_key(s):
    return int(str(s)[1:])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--n-repeats",type=int,default=8)
    a=ap.parse_args()
    df=pd.read_csv(a.input); art=joblib.load(a.model)
    feats=art["feature_names"]
    X=df[feats].apply(pd.to_numeric,errors="coerce").to_numpy(float); y=df.hr_bpm.to_numpy(float)
    val_subjects=set(map(str,art["validation_subjects"]))
    mask=df.subject.astype(str).isin(val_subjects).to_numpy()
    if not mask.any(): raise RuntimeError("No validation subjects found in input.")
    result=permutation_importance(art["model"],X[mask],y[mask],scoring="neg_mean_absolute_error",
                                  n_repeats=a.n_repeats,random_state=42,n_jobs=-1)
    out=pd.DataFrame({"feature":feats,"importance_mae_increase_bpm":result.importances_mean,
                      "std":result.importances_std})
    out=out.sort_values("importance_mae_increase_bpm",ascending=False)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
    print("=== VALIDATION-ONLY FEATURE IMPORTANCE ===")
    print(out.head(25).to_string(index=False))
    print(f"\nSaved: {a.out}")
if __name__=="__main__": main()
