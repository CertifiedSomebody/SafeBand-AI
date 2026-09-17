from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from activity_sequence_models_v2 import run_cv, DEFAULT
import argparse

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=DEFAULT)
    ap.add_argument("--out",type=Path,default=ROOT/"models/activity_accgyro_v2")
    ap.add_argument("--epochs",type=int,default=70); ap.add_argument("--patience",type=int,default=12)
    ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    run_cv("transformer",a.input,a.out,a.epochs,a.patience,a.batch_size,a.lr,a.weight_decay,a.seed)
