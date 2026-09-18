from pathlib import Path
import sys,argparse,json,random
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from ai.nonspeech7k_model_final import SmallLogMelCNN,LABELS

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='datasets/processed/nonspeech7k/logmel_v3_1.npz'); ap.add_argument('--out',default='models/nonspeech7k_audio_final_v1'); ap.add_argument('--epochs',type=int,default=16); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    seed_all(args.seed)
    ip=Path(args.input); ip=ip if ip.is_absolute() else ROOT/ip
    if not ip.is_file(): raise FileNotFoundError(f'Cache not found: {ip}')
    d=np.load(ip,allow_pickle=False); X=d['X'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str)
    if X.shape!=(6283,1,64,128) or len(y)!=6283 or len(groups)!=6283: raise RuntimeError(f'Unexpected cache dimensions: {X.shape}, {len(y)}, {len(groups)}')
    if len(set(groups.tolist()))!=1899: raise RuntimeError('Expected 1899 File-ID groups')
    if not np.isfinite(X).all() or sorted(np.unique(y).tolist())!=list(range(7)): raise RuntimeError('Invalid cache values/labels')
    mean=float(X.mean(dtype=np.float64)); std=max(float(X.std(dtype=np.float64)),1e-6); X=((X-mean)/std).astype(np.float32)
    weights=compute_class_weight(class_weight='balanced',classes=np.arange(7),y=y).astype(np.float32)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=SmallLogMelCNN().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=torch.cuda.is_available())
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb)
            if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at epoch {epoch}')
            loss.backward(); opt.step(); losses.append(float(loss.item()))
        ml=float(np.mean(losses)); history.append({'epoch':epoch,'train_loss':ml}); print(f'[FINAL] epoch={epoch:02d} train_loss={ml:.4f}')
    out=Path(args.out); out=out if out.is_absolute() else ROOT/out; out.mkdir(parents=True,exist_ok=True)
    model_path=out/'nonspeech7k_audio_model.pt'
    torch.save({'model_state_dict':model.state_dict(),'model_name':'SmallLogMelCNN','labels':LABELS,'input_shape':[1,64,128],'sample_rate':16000,'n_mels':64,'n_fft':512,'hop_length':160,'fmin':50,'fmax':7600,'logmel_db_floor':-80.0,'normalization_mean':mean,'normalization_std':std,'epochs':args.epochs,'learning_rate':args.lr,'weight_decay':args.weight_decay,'batch_size':args.batch_size,'seed':args.seed,'training_samples':6283,'training_file_id_groups':1899,'class_weights':weights.tolist()},model_path)
    meta={'model_file':str(model_path.relative_to(ROOT)),'training_samples':6283,'groups':1899,'epochs':args.epochs,'normalization_mean':mean,'normalization_std':std,'history':history,'locked_v3_1_cv':{'mean_accuracy':0.715743246979729,'mean_balanced_accuracy':0.7046669739595138,'mean_macro_f1':0.6496588095637443,'pooled_accuracy':0.7252904663377368,'pooled_balanced_accuracy':0.7175890499177379,'pooled_macro_f1':0.6591197556329655}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(f'[PASS] model -> {model_path}'); print(f'[PASS] metadata -> {out/"metadata.json"}')
if __name__=='__main__': main()
