from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np,torch
from ai.nonspeech7k_model_final import SmallLogMelCNN
from tools.common_nonspeech7k_audio_final import load_audio,make_logmel

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('--model',default='models/nonspeech7k_audio_final_v1/nonspeech7k_audio_model.pt'); a=ap.parse_args()
    mp=Path(a.model); mp=mp if mp.is_absolute() else ROOT/mp; wav=Path(a.audio); wav=wav if wav.is_absolute() else ROOT/wav
    if not mp.is_file(): raise FileNotFoundError(mp)
    if not wav.is_file(): raise FileNotFoundError(wav)
    c=torch.load(mp,map_location='cpu'); labels=c['labels']; model=SmallLogMelCNN(len(labels)); model.load_state_dict(c['model_state_dict']); model.eval()
    x,sr=load_audio(wav,c['sample_rate']); z=make_logmel(x,sr); t=torch.from_numpy(z[None,None]).float(); t=(t-c['normalization_mean'])/c['normalization_std']
    with torch.no_grad(): p=torch.softmax(model(t),dim=1)[0].numpy()
    order=np.argsort(p)[::-1]; print(json.dumps({'audio':str(wav),'predicted_label':labels[int(order[0])],'confidence':float(p[order[0]]),'probabilities':{labels[int(i)]:float(p[i]) for i in order}},indent=2))
if __name__=='__main__': main()
