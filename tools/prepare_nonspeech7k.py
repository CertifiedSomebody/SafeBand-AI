from pathlib import Path
import sys, argparse
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
LABEL_MAP={"breath":"breath","cough":"cough","crying":"crying","laugh":"laugh","screaming":"screaming","sneeze":"sneeze","yawn":"yawn","yawm":"yawn"}

def load(path):
    df=pd.read_csv(path)
    lower={c.lower().replace(' ','').replace('_',''):c for c in df.columns}
    def col(*names):
        for n in names:
            k=n.lower().replace(' ','').replace('_','')
            if k in lower:return lower[k]
        raise KeyError(names)
    m={'filename':col('Filename'),'file_id':col('File ID','File_ID'),'duration_ms':col('Duration in ms','Durationin ms','Durationinms'),'class_id':col('Class ID','Class_id'),'class_name':col('Classname'),'augmentation_id':col('augmentation id','Augment Id','Augment_Id'),'augmentation_type':col('Augmentation type'),'source':col('source')}
    df['filename']=df[m['filename']].astype(str).str.strip(); df['file_id']=df[m['file_id']].astype(str).str.strip()
    df['label_raw']=df[m['class_name']].astype(str).str.strip().str.lower(); df['label']=df.label_raw.map(LABEL_MAP)
    df['source']=df[m['source']].astype(str).str.strip(); df['duration_ms']=pd.to_numeric(df[m['duration_ms']],errors='coerce')
    df['augmentation_id']=pd.to_numeric(df[m['augmentation_id']],errors='coerce')
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='datasets/raw/Nonspeech7k')
    ap.add_argument('--train-meta',default='datasets/raw/Nonspeech7k/metadata of train set .csv')
    ap.add_argument('--test-meta',default='datasets/raw/Nonspeech7k/metadata of test set.csv')
    ap.add_argument('--out-dir',default='datasets/processed/nonspeech7k')
    args=ap.parse_args()
    train=load(REPO/args.train_meta); test=load(REPO/args.test_meta)
    assert train.label.notna().all() and test.label.notna().all(), 'Unknown class labels remain.'
    test_keys=set(zip(test.source,test.file_id))
    train['test_overlap_group']=[(s,i) in test_keys for s,i in zip(train.source,train.file_id)]
    clean=train[(~train.test_overlap_group) & (train.augmentation_id.fillna(0)==0)].copy()
    test=test[test.augmentation_id.fillna(0)==0].copy()
    root=(REPO/args.root); out=(REPO/args.out_dir); out.mkdir(parents=True,exist_ok=True)
    # Relative paths are deterministic and portable across machines.
    def rels(df, split):
        paths=[]
        splitroot=root/split
        for fn in df.filename:
            matches=list(splitroot.rglob(fn))
            paths.append(str(matches[0].relative_to(REPO)) if matches else '')
        return paths
    train['audio_path']=rels(train,'train')
    clean['audio_path']=rels(clean,'train'); test['audio_path']=rels(test,'test')
    assert (clean.audio_path!='').all(), 'Missing train WAV(s) after preparation.'
    assert (test.audio_path!='').all(), 'Missing test WAV(s) after preparation.'
    cols=['filename','file_id','label','label_raw','duration_ms','augmentation_id','source','audio_path','test_overlap_group']
    clean[cols].to_csv(out/'train_clean_manifest.csv',index=False)
    test.assign(test_overlap_group=False)[cols].to_csv(out/'test_official_manifest.csv',index=False)
    # Keep a quarantine manifest so exclusions are transparent and reproducible.
    train[train.test_overlap_group][cols].to_csv(out/'train_test_overlap_quarantine.csv',index=False)
    print(f'[PASS] Clean train rows: {len(clean)}')
    print(f'[PASS] Official test rows: {len(test)}')
    print(f'[INFO] Quarantined train rows: {int(train.test_overlap_group.sum())}')
    print('[INFO] Label normalization: yawm -> yawn')
    print(f'[PASS] Output: {out}')
if __name__=='__main__':main()
