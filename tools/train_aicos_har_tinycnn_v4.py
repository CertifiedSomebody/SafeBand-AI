"""Compact temporal CNN benchmark with an internal validation split.

Official Fold=test remains untouched. The non-test data are split by
acquisition_id, not by individual windows, so overlapping windows from one
acquisition cannot leak between train and validation.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
class TinyCNN(nn.Module):
    def __init__(self,nc):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(3,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,96,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1));self.fc=nn.Linear(96,nc)
    def forward(self,x):return self.fc(self.net(x).squeeze(-1))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--npz",required=True);ap.add_argument("--manifest",required=True);ap.add_argument("--out",default="models/aicos_har/tinycnn_v4");ap.add_argument("--epochs",type=int,default=30);ap.add_argument("--batch-size",type=int,default=128);ap.add_argument("--patience",type=int,default=5);ap.add_argument("--seed",type=int,default=42);args=ap.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    X=np.load(args.npz)["X"].astype(np.float32);rows=list(csv.DictReader(open(args.manifest,encoding="utf-8-sig")));labels=sorted({r["activity"] for r in rows});mp={v:i for i,v in enumerate(labels)};y=np.array([mp[r["activity"]] for r in rows]);fold=np.char.lower(np.array([r["fold"] for r in rows],dtype=str));groups=np.array([r["acquisition_id"] for r in rows])
    te=fold=="test";non=np.flatnonzero(~te);gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed);tr_rel,va_rel=next(gss.split(non,y[non],groups[non]));tr=non[tr_rel];va=non[va_rel]
    mean=X[tr].mean(axis=(0,2),keepdims=True);std=X[tr].std(axis=(0,2),keepdims=True)+1e-6;Xn=(X-mean)/std
    ds=TensorDataset(torch.from_numpy(Xn[tr]),torch.from_numpy(y[tr]));dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True)
    model=TinyCNN(len(labels));counts=np.bincount(y[tr],minlength=len(labels));weights=counts.sum()/np.maximum(counts,1);weights=weights/weights.mean();lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32));opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-1;best_state=None;bad=0
    for ep in range(args.epochs):
        model.train();total=0
        for xb,yb in dl:opt.zero_grad();loss=lossfn(model(xb),yb);loss.backward();opt.step();total+=loss.item()*len(yb)
        model.eval()
        with torch.no_grad():vp=model(torch.from_numpy(Xn[va])).argmax(1).numpy()
        score=f1_score(y[va],vp,average="macro")
        print(f"epoch {ep+1}/{args.epochs} train_loss={total/len(ds):.5f} val_macro_f1={score:.5f}")
        if score>best:best=score;best_state={k:v.cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=args.patience:print("early stopping");break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():pred=model(torch.from_numpy(Xn[te])).argmax(1).numpy()
    result={"accuracy":accuracy_score(y[te],pred),"balanced_accuracy":balanced_accuracy_score(y[te],pred),"macro_f1":f1_score(y[te],pred,average="macro"),"classes":labels,"confusion_matrix":confusion_matrix(y[te],pred,labels=np.arange(len(labels))).tolist(),"train_acquisitions":int(len(set(groups[tr]))),"validation_acquisitions":int(len(set(groups[va]))),"official_test_samples":int(te.sum())}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":model.state_dict(),"labels":labels,"mean":mean.reshape(-1).tolist(),"std":std.reshape(-1).tolist()},out/"model.pt");(out/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(result)
if __name__=="__main__":main()
