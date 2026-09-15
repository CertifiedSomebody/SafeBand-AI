"""Audit a SisFall ZIP without extracting the full dataset."""
from __future__ import annotations
import argparse,csv,io,json,os,re,zipfile
from collections import Counter,defaultdict

PAT=re.compile(r'^(D|F)(\d{2})_(SA|SE)(\d{2})_R(\d{2})\.txt$',re.I)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True); rows=[]; code_counts=Counter(); subject_counts=Counter(); malformed=0; total_rows=0; bad_rows=0
    with zipfile.ZipFile(args.zip) as z:
      for name in z.namelist():
        base=os.path.basename(name)
        if not base.lower().endswith('.txt') or base.lower()=='readme.txt': continue
        m=PAT.match(base)
        if not m: continue
        typ,code,prefix,sid,trial=m.groups(); subject=prefix.upper()+sid; label=('FALL' if typ.upper()=='F' else 'NONFALL')
        info={'file':name,'code':typ.upper()+code,'subject_id':subject,'trial':int(trial),'recording_type':label,'rows':0,'bad_rows':0}
        with z.open(name) as fh:
          for raw in io.TextIOWrapper(fh,encoding='utf-8',errors='replace'):
            line=raw.strip()
            if not line: continue
            parts=[p.strip() for p in line.rstrip(';').split(',')]
            info['rows']+=1; total_rows+=1
            if len(parts)!=9:
              info['bad_rows']+=1; bad_rows+=1; continue
            try: [int(p) for p in parts]
            except ValueError: info['bad_rows']+=1; bad_rows+=1
        rows.append(info); code_counts[info['code']]+=1; subject_counts[subject]+=1
    rows.sort(key=lambda x:x['file'])
    with open(os.path.join(args.out,'sisfall_manifest.csv'),'w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['file']); w.writeheader(); w.writerows(rows)
    report={'dataset':'SisFall','recordings':len(rows),'fall_recordings':sum(1 for r in rows if r['recording_type']=='FALL'),'nonfall_recordings':sum(1 for r in rows if r['recording_type']=='NONFALL'),'subjects':len(subject_counts),'subjects_adult':sum(s.startswith('SA') for s in subject_counts),'subjects_elderly':sum(s.startswith('SE') for s in subject_counts),'total_rows':total_rows,'bad_rows':bad_rows,'code_counts':dict(sorted(code_counts.items())),'subject_counts':dict(sorted(subject_counts.items())),'sampling_hz':200,'columns':9}
    with open(os.path.join(args.out,'sisfall_audit.json'),'w',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
