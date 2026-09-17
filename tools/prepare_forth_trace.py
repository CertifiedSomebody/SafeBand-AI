"""Fast, leakage-safe FORTH-TRACE left-wrist preparation for SafeBand.

Reads only dev1 (left wrist), segments at 51.2 Hz, computes 9-DoF ACC/GYRO/MAG
features, and writes compact processed CSVs. Raw files are never changed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd
FS=51.2
RAW_COLS=['device_id','ax','ay','az','gx','gy','gz','mx','my','mz','timestamp','label_id']
CHANNELS=['ax','ay','az','gx','gy','gz','mx','my','mz']
CORE_MAP={1:'STANDING',2:'SITTING',3:'SITTING',4:'WALKING',5:'WALKING',6:'STAIRS',7:'STAIRS'}
FULL_MAP={**CORE_MAP,**{i:'TRANSITION' for i in range(8,17)}}

def discover(root):
    fs=sorted(root.glob('part*/part*dev1.csv'),key=lambda p:int(p.parent.name[4:]))
    if not fs: raise FileNotFoundError(f'No part*/part*dev1.csv under {root}')
    return fs

def _features_for_windows(a, fs):
    """a: [n_windows, n_samples, 9], returns DataFrame of vectorized features."""
    out={}; mags=[]
    for j,c in enumerate(CHANNELS):
        x=a[:,:,j]; d=np.diff(x,axis=1)
        q=np.percentile(x,[5,10,25,50,75,90,95],axis=1)
        mu=x.mean(1); sd=x.std(1); centered=x-mu[:,None]
        z=np.divide(centered,sd[:,None],out=np.zeros_like(centered),where=sd[:,None]>1e-12)
        out.update({f'{c}_mean':mu,f'{c}_std':sd,f'{c}_min':x.min(1),f'{c}_max':x.max(1),f'{c}_median':q[3],f'{c}_p05':q[0],f'{c}_p10':q[1],f'{c}_p25':q[2],f'{c}_p75':q[4],f'{c}_p90':q[5],f'{c}_p95':q[6],f'{c}_range':np.ptp(x,axis=1),f'{c}_iqr':q[4]-q[2],f'{c}_rms':np.sqrt(np.mean(x*x,1)),f'{c}_skew':np.mean(z**3,1),f'{c}_kurtosis':np.mean(z**4,1)-3,f'{c}_mad':np.mean(np.abs(centered),1),f'{c}_absdiff_mean':np.mean(np.abs(d),1),f'{c}_diff_std':d.std(1),f'{c}_diff_energy':np.mean(d*d,1)})
        out[f'{c}_zero_crossings']=np.count_nonzero((centered[:,:-1]*centered[:,1:])<0,axis=1)
    acc=np.sqrt(np.sum(a[:,:,:3]**2,axis=2)); gyro=np.sqrt(np.sum(a[:,:,3:6]**2,axis=2)); mag=np.sqrt(np.sum(a[:,:,6:9]**2,axis=2))
    for name,x in [('acc_mag',acc),('gyro_mag',gyro),('mag_mag',mag)]:
        d=np.diff(x,axis=1); mu=x.mean(1); sd=x.std(1); q=np.percentile(x,[5,10,25,50,75,90,95],axis=1); centered=x-mu[:,None]; z=np.divide(centered,sd[:,None],out=np.zeros_like(centered),where=sd[:,None]>1e-12)
        out.update({f'{name}_mean':mu,f'{name}_std':sd,f'{name}_min':x.min(1),f'{name}_max':x.max(1),f'{name}_median':q[3],f'{name}_p05':q[0],f'{name}_p10':q[1],f'{name}_p25':q[2],f'{name}_p75':q[4],f'{name}_p90':q[5],f'{name}_p95':q[6],f'{name}_range':np.ptp(x,axis=1),f'{name}_iqr':q[4]-q[2],f'{name}_rms':np.sqrt(np.mean(x*x,1)),f'{name}_skew':np.mean(z**3,1),f'{name}_kurtosis':np.mean(z**4,1)-3,f'{name}_mad':np.mean(np.abs(centered),1),f'{name}_absdiff_mean':np.mean(np.abs(d),1),f'{name}_diff_std':d.std(1),f'{name}_diff_energy':np.mean(d*d,1),f'{name}_zero_crossings':np.count_nonzero((centered[:,:-1]*centered[:,1:])<0,axis=1)})
        p=np.abs(np.fft.rfft(centered,axis=1))**2; f=np.fft.rfftfreq(x.shape[1],1/fs); p[:,0]=0; total=p.sum(1); safe=np.where(total>1e-12,total,1); pn=p/safe[:,None]
        ent=-(np.where(pn>0,pn*np.log2(np.where(pn>0,pn,1)),0).sum(1))/np.log2(len(f)); dom=f[np.argmax(p,axis=1)]
        out[f'{name}_fft_low']=(p[:,(f>=.3)&(f<3)].sum(1)/safe); out[f'{name}_fft_mid']=(p[:,(f>=3)&(f<7)].sum(1)/safe); out[f'{name}_fft_high']=(p[:,(f>=7)&(f<15)].sum(1)/safe); out[f'{name}_spectral_entropy']=ent; out[f'{name}_dominant_hz']=dom
    # Correlations inside each modality and between magnitudes.
    pairs=[('acc',0,1),('acc',0,2),('acc',1,2),('gyro',3,4),('gyro',3,5),('gyro',4,5),('mag',6,7),('mag',6,8),('mag',7,8)]
    for name,i,j in pairs:
        xi=a[:,:,i]-a[:,:,i].mean(1)[:,None]; xj=a[:,:,j]-a[:,:,j].mean(1)[:,None]; den=np.sqrt(np.sum(xi*xi,1)*np.sum(xj*xj,1)); out[f'{name}_{i%3}_{j%3}_corr']=np.divide(np.sum(xi*xj,1),den,out=np.zeros(len(a)),where=den>1e-12)
    for name1,x,name2,y in [('acc_mag',acc,'gyro_mag',gyro),('acc_mag',acc,'mag_mag',mag),('gyro_mag',gyro,'mag_mag',mag)]:
        xx=x-x.mean(1)[:,None]; yy=y-y.mean(1)[:,None]; den=np.sqrt(np.sum(xx*xx,1)*np.sum(yy*yy,1)); out[f'{name1}_{name2}_corr']=np.divide(np.sum(xx*yy,1),den,out=np.zeros(len(a)),where=den>1e-12)
    return pd.DataFrame(out).replace([np.inf,-np.inf],np.nan).fillna(0.0)

def process_part(path,samples,stride,purity,mapping):
    df=pd.read_csv(path,header=None,names=RAW_COLS)
    df=df.dropna(subset=CHANNELS+['label_id']).reset_index(drop=True); labels=df.label_id.astype(int).to_numpy(); n=len(df); starts=np.arange(0,n-samples+1,stride)
    if len(starts)==0: return pd.DataFrame()
    idx=starts[:,None]+np.arange(samples)[None,:]; y=labels[idx]
    mapped=np.full(y.shape,'EXCLUDE',dtype=object)
    for k,v in mapping.items(): mapped[y==k]=v
    valid=mapped!='EXCLUDE'; counts={lab:(mapped==lab).sum(1) for lab in set(mapping.values())}; best=np.empty(len(starts),dtype=object); best_count=np.zeros(len(starts),int)
    for lab,c in counts.items(): take=c>best_count; best[take]=lab; best_count[take]=c[take]
    purity_arr=best_count/np.maximum(valid.sum(1),1); keep=(valid.mean(1)>=purity)&(purity_arr>=purity)
    if not keep.any(): return pd.DataFrame()
    a=np.stack([df[c].to_numpy(float)[idx[keep]] for c in CHANNELS],axis=2)
    feats=_features_for_windows(a,FS); p=int(path.parent.name[4:]); feats['participant_id']=p; feats['start_row']=starts[keep]; feats['end_row']=starts[keep]+samples-1; feats['window_samples']=samples; feats['sample_rate_hz']=FS; feats['label_purity']=purity_arr[keep]; feats['activity_label']=best[keep]
    return feats

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--out-dir',type=Path,default=ROOT/'datasets/processed/forth_trace'); ap.add_argument('--window-seconds',type=float,default=3.0); ap.add_argument('--overlap',type=float,default=.5); ap.add_argument('--purity',type=float,default=.90); args=ap.parse_args()
    samples=int(round(FS*args.window_seconds)); stride=max(1,int(round(samples*(1-args.overlap)))); args.out_dir.mkdir(parents=True,exist_ok=True); files=discover(args.data_dir)
    core_parts=[]; full_parts=[]; audit={'dataset':'FORTH-TRACE','device':'dev1 / left wrist','sample_rate_hz':FS,'window_seconds':args.window_seconds,'window_samples':samples,'overlap':args.overlap,'purity':args.purity,'files':[]}
    for f in files:
        c=process_part(f,samples,stride,args.purity,CORE_MAP); t=process_part(f,samples,stride,args.purity,FULL_MAP); core_parts.append(c); full_parts.append(t); p=int(f.parent.name[4:]); audit['files'].append({'participant':p,'file':str(f),'core_windows':len(c),'transition_aware_windows':len(t)}); print(f'part{p:02d}: core={len(c):5d} transition_aware={len(t):5d}',flush=True)
    core=pd.concat(core_parts,ignore_index=True); full=pd.concat(full_parts,ignore_index=True); cp=args.out_dir/'core_activity_features.csv'; fp=args.out_dir/'transition_aware_features.csv'; core.to_csv(cp,index=False); full.to_csv(fp,index=False)
    audit.update({'core_windows':len(core),'transition_aware_windows':len(full),'feature_count':len([c for c in core.columns if c not in {'participant_id','start_row','end_row','window_samples','sample_rate_hz','label_purity','activity_label'}]),'core_class_counts':core.activity_label.value_counts().sort_index().to_dict(),'transition_aware_class_counts':full.activity_label.value_counts().sort_index().to_dict()})
    (args.out_dir/'prepare_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8'); print('\nCore:',audit['core_class_counts']); print('Transition:',audit['transition_aware_class_counts']); print('Saved:',cp,fp)
if __name__=='__main__': main()
