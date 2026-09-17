#!/usr/bin/env python3
"""Validate the reproducibility/integrity contract of the final PPG artifact."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--report',required=True)
    ap.add_argument('--predictions',required=True)
    a=ap.parse_args()
    model=joblib.load(a.model)
    report=json.loads(Path(a.report).read_text(encoding='utf-8'))
    pred=pd.read_csv(a.predictions)
    required=['model','feature_names','model_type','calibration','temporal','preprocessing','version','domain_warning']
    missing=[x for x in required if x not in model]
    if missing: raise SystemExit(f'Model artifact missing keys: {missing}')
    if not model['feature_names']: raise SystemExit('Empty feature_names')
    for c in ['subject','activity','record','hr_bpm','prediction_raw','prediction_temporal']:
        if c not in pred.columns: raise SystemExit(f'Predictions missing column: {c}')
    if pred.empty: raise SystemExit('Predictions file is empty')
    if report.get('split',{}).get('subject_overlap') is not False: raise SystemExit('Subject-overlap contract is not explicitly false')
    print(json.dumps({
        'status':'PASS', 'artifact_version':model['version'],
        'model_type':model['model_type'], 'feature_count':len(model['feature_names']),
        'prediction_rows':len(pred), 'test_subjects':report['split']['test_subjects'],
        'domain_warning':model['domain_warning']
    },indent=2))
if __name__=='__main__': main()
