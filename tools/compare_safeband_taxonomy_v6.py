from pathlib import Path
import json, argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--report",required=True); args=ap.parse_args()
    r=json.loads(Path(args.report).read_text())
    print("\nSafeBand taxonomy ACC-only V6\n")
    print(f"Train windows: {r['train_windows']}")
    print(f"Test windows : {r['test_windows']}")
    print("\nModel                         Accuracy   Balanced Acc   Macro-F1")
    print("-"*68)
    for n,x in r["models"].items():
        print(f"{n:28s} {x['accuracy']*100:8.2f}% {x['balanced_accuracy']*100:13.2f}% {x['macro_f1']*100:10.2f}%")
if __name__=="__main__": main()
