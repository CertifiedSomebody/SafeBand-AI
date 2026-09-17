#!/usr/bin/env python3
"""One-command PTT reference build + final artifact validation."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    print('\n>>>',' '.join(map(str,cmd)))
    subprocess.run(cmd, cwd=ROOT, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ptt-root',default='datasets/raw/PTT')
    ap.add_argument('--out-dir',default='models/ppg_v6_final')
    args=ap.parse_args()
    proc=Path('datasets/processed/ptt_hr_v6.csv')
    summary=Path('datasets/processed/ptt_hr_v6_summary.json')
    out=Path(args.out_dir)
    model=out/'ppg_v6_max30101_reference.joblib'
    report=out/'ppg_v6_max30101_reference_report.json'
    pred=out/'ppg_v6_max30101_reference_predictions.csv'
    run([sys.executable,'tools/prepare_ptt_hr_v6.py','--ptt-root',args.ptt_root,'--out',str(proc),'--summary-out',str(summary)])
    run([sys.executable,'tools/train_ppg_v6.py','--input',str(proc),'--model-out',str(model),'--report-out',str(report),'--predictions-out',str(pred)])
    run([sys.executable,'tools/validate_ppg_v6_artifacts.py','--model',str(model),'--report',str(report),'--predictions',str(pred)])
    print('\nFINAL PPG REFERENCE PIPELINE: PASS')
    print('MAX30102 hardware validation remains a separate required stage.')
if __name__=='__main__': main()
