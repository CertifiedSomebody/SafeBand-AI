"""Audit AICOS-HAR acquisitions that produced zero windows.

Compares metadata acquisitions against the window manifest, preserving the
metadata-side LiftDown/LiftUp labels through the raw Elevator aliases.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from common_aicos_har_v4 import metadata_to_raw_acquisition_id

def read_csv(p):
    with open(p,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metadata',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out',default='reports/aicos_har_zero_window_v6.json')
    args=ap.parse_args()
    meta=read_csv(args.metadata); man=read_csv(args.manifest)
    meta_ids=[]
    for r in meta:
        aid=str(r.get('AcquisitionID','')).replace('\\','/').strip()
        if aid: meta_ids.append(aid)
    win_ids={str(r.get('acquisition_id','')).replace('\\','/').strip() for r in man}
    missing=sorted(set(meta_ids)-win_ids)
    by_id={str(r.get('AcquisitionID','')).replace('\\','/').strip():r for r in meta}
    detail=[]
    for aid in missing:
        r=by_id[aid]
        detail.append({
            'acquisition_id':aid,
            'participant_id':r.get('participant_id',''),
            'activity':r.get('activity',aid.split('/')[1].rsplit('_',1)[0] if len(aid.split('/'))>1 else ''),
            'position':r.get('Position_FullName',r.get('device_position_id','')),
            'fold':r.get('Fold',''),
            'raw_alias_id':metadata_to_raw_acquisition_id(aid),
        })
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    report={'metadata_acquisitions':len(set(meta_ids)),'manifest_acquisitions':len(win_ids),'zero_window_acquisitions':len(missing),'zero_window_details':detail}
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('AICOS-HAR ZERO-WINDOW AUDIT V6')
    print('Metadata acquisitions:',len(set(meta_ids)))
    print('Manifest acquisitions:',len(win_ids))
    print('Zero-window acquisitions:',len(missing))
    if detail:
        from collections import Counter
        print('By activity:',dict(Counter(d['activity'] for d in detail)))
        print('By fold:',dict(Counter(d['fold'] for d in detail)))
        print('Details:')
        for d in detail: print(' ',d['acquisition_id'],'|',d['activity'],'|',d['fold'],'|',d['position'])
    else: print('All metadata acquisitions produced at least one window.')
    print('Report:',out)
if __name__=='__main__': main()
