from pathlib import Path
import sys, argparse, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LABEL_MAP = {"breath":"breath","cough":"cough","crying":"crying","laugh":"laugh","screaming":"screaming","sneeze":"sneeze","yawn":"yawn","yawm":"yawn"}


def find_wavs(root):
    return {p.name: p for p in Path(root).rglob('*.wav')}


def load_meta(path):
    df = pd.read_csv(path)
    lower = {c.lower().replace(' ','').replace('_',''): c for c in df.columns}
    def col(*names):
        for n in names:
            k=n.lower().replace(' ','').replace('_','')
            if k in lower: return lower[k]
        raise KeyError(names)
    return df, {
        'filename': col('Filename'), 'file_id': col('File ID','File_ID'),
        'duration_ms': col('Duration in ms','Durationin ms','Durationinms'),
        'class_id': col('Class ID','Class_id'), 'class_name': col('Classname'),
        'augmentation_id': col('augmentation id','Augment Id','Augment_Id'),
        'augmentation_type': col('Augmentation type'), 'source': col('source')}


def normalize(df, cols):
    out=df.copy()
    out['filename_norm']=out[cols['filename']].astype(str).str.strip()
    out['file_id_norm']=out[cols['file_id']].astype(str).str.strip()
    out['label_raw']=out[cols['class_name']].astype(str).str.strip().str.lower()
    out['label']=out['label_raw'].map(LABEL_MAP)
    out['source_norm']=out[cols['source']].astype(str).str.strip().str.lower()
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/raw/Nonspeech7k')
    ap.add_argument('--train-meta', default='datasets/raw/Nonspeech7k/metadata of train set .csv')
    ap.add_argument('--test-meta', default='datasets/raw/Nonspeech7k/metadata of test set.csv')
    ap.add_argument('--out', default='reports/nonspeech7k_dataset_audit_v1.json')
    args=ap.parse_args()
    root=Path(args.root); train_meta=Path(args.train_meta); test_meta=Path(args.test_meta)
    if not root.is_absolute(): root=REPO/root
    if not train_meta.is_absolute(): train_meta=REPO/train_meta
    if not test_meta.is_absolute(): test_meta=REPO/test_meta
    assert root.exists(), f'Missing dataset root: {root}'
    assert train_meta.exists(), f'Missing train metadata: {train_meta}'
    assert test_meta.exists(), f'Missing test metadata: {test_meta}'
    tr,tc=load_meta(train_meta); te,ec=load_meta(test_meta)
    tr=normalize(tr,tc); te=normalize(te,ec)
    train_wavs=find_wavs(root/'train'); test_wavs=find_wavs(root/'test')
    report={'dataset_root':str(root),'train_metadata_rows':len(tr),'test_metadata_rows':len(te),
            'train_wav_files':len(train_wavs),'test_wav_files':len(test_wavs),'errors':[], 'warnings':[]}
    for name,df,wavs in [('train',tr,train_wavs),('test',te,test_wavs)]:
        missing=sorted(set(df.filename_norm)-set(wavs))
        extra=sorted(set(wavs)-set(df.filename_norm))
        report[f'{name}_missing_wavs']=missing
        report[f'{name}_extra_wavs']=extra
        bad_labels=sorted(df.loc[df.label.isna(),'label_raw'].unique())
        report[f'{name}_unknown_labels']=bad_labels
        report[f'{name}_class_counts']=df.label.value_counts().sort_index().to_dict()
        report[f'{name}_sources']=df.source_norm.value_counts().to_dict()
        report[f'{name}_augmentation_ids']=sorted(df[tc['augmentation_id'] if name=='train' else ec['augmentation_id']].dropna().unique().tolist())
        if missing: report['errors'].append(f'{name}: {len(missing)} metadata rows have no matching WAV')
        if bad_labels: report['errors'].append(f'{name}: unknown labels {bad_labels}')
    train_keys=set(zip(tr.source_norm,tr.file_id_norm)); test_keys=set(zip(te.source_norm,te.file_id_norm))
    overlap=sorted(train_keys & test_keys)
    report['source_file_id_overlap_count']=len(overlap)
    report['source_file_id_overlaps']=[{'source':s,'file_id':i} for s,i in overlap]
    report['exact_filename_overlap_count']=len(set(tr.filename_norm)&set(te.filename_norm))
    if overlap: report['warnings'].append('Train/test share source+File ID groups; these groups must be excluded from training when preserving the official test set.')
    # Duration consistency against metadata only; audio decoding is deferred to preparation to avoid a costly full decode here.
    for name,df,cols in [('train',tr,tc),('test',te,ec)]:
        d=pd.to_numeric(df[cols['duration_ms']],errors='coerce')
        report[f'{name}_duration_ms']={'min':float(d.min()),'median':float(d.median()),'max':float(d.max()),'nonfinite':int(d.isna().sum())}
    report['status']='PASS' if not report['errors'] else 'FAIL'
    out=Path(args.out); out=out if out.is_absolute() else REPO/out; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"[{report['status']}] Nonspeech7k audit")
    print(f"Train metadata: {len(tr)} | train WAV: {len(train_wavs)}")
    print(f"Test metadata:  {len(te)} | test WAV:  {len(test_wavs)}")
    print(f"Classes: {sorted(set(tr.label.dropna()) | set(te.label.dropna()))}")
    print(f"Source+File-ID overlap groups: {len(overlap)}")
    if overlap:
        print('WARNING: overlapping groups will be quarantined from clean training preparation.')
    print(f"Report: {out}")
    if report['errors']: raise SystemExit(1)

if __name__=='__main__': main()
