"""Validate AICOS-HAR window NPZ + manifest before ML training."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def resolve_npz(path):
    p=Path(path)
    if p.exists(): return p
    # Protect against the old builder's np.savez_compressed('name.csv') behavior.
    if p.suffix.lower()==".csv" and Path(str(p)+".npz").exists(): return Path(str(p)+".npz")
    raise FileNotFoundError(f"Window file not found: {p}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--windows",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--metadata",default=None); ap.add_argument("--expected-windows",type=int,default=None); ap.add_argument("--out",default="reports/aicos_har_window_validation_v5.json"); args=ap.parse_args()
    wp=resolve_npz(args.windows); mp=Path(args.manifest)
    if not mp.exists(): raise SystemExit(f"Manifest not found: {mp}")
    z=np.load(wp,allow_pickle=False)
    if "X" not in z.files: raise SystemExit(f"NPZ has no X array: {z.files}")
    X=z["X"]
    rows=list(csv.DictReader(mp.open(encoding="utf-8-sig",newline="")))
    required={"acquisition_id","participant_id","activity","fold","position"}
    missing=required-set(rows[0]) if rows else required
    if missing: raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    checks={"windows_file":str(wp),"manifest_file":str(mp),"shape":list(X.shape),"manifest_rows":len(rows),"finite":bool(np.isfinite(X).all()),"dtype":str(X.dtype)}
    errors=[]
    if X.ndim!=3 or X.shape[1:]!=(3,100): errors.append(f"expected shape (N,3,100), got {X.shape}")
    if len(rows)!=len(X): errors.append(f"manifest rows {len(rows)} != windows {len(X)}")
    if not np.isfinite(X).all(): errors.append("windows contain NaN/Inf")
    if len(rows):
        aids=[r["acquisition_id"] for r in rows]; parts=[r["participant_id"] for r in rows]; acts=[r["activity"] for r in rows]; folds=[r["fold"].strip().lower() for r in rows]
        checks.update({"unique_acquisitions":len(set(aids)),"unique_participants":len(set(parts)),"activity_counts":{a:acts.count(a) for a in sorted(set(acts))},"fold_counts":{f:folds.count(f) for f in sorted(set(folds))},"participants_by_fold":{f:len({p for p,ff in zip(parts,folds) if ff==f}) for f in sorted(set(folds))}})
        byp={}
        for p,f in zip(parts,folds): byp.setdefault(p,set()).add(f)
        overlap={p:sorted(v) for p,v in byp.items() if len(v)>1}
        checks["participants_in_multiple_folds"]=overlap
        if overlap: errors.append(f"{len(overlap)} participants occur in multiple folds")
        if "test" not in set(folds): errors.append("no official test fold found")
    if args.expected_windows is not None and len(X)!=args.expected_windows: errors.append(f"expected {args.expected_windows} windows, got {len(X)}")
    if args.metadata:
        with open(args.metadata,encoding="utf-8-sig",newline="") as f: meta_ids={r["AcquisitionID"].strip() for r in csv.DictReader(f)}
        window_ids={r["acquisition_id"] for r in rows}; missing_ids=sorted(meta_ids-window_ids); extra_ids=sorted(window_ids-meta_ids)
        checks["metadata_acquisitions"]=len(meta_ids); checks["metadata_ids_without_windows"]=len(missing_ids); checks["window_ids_not_in_metadata"]=len(extra_ids); checks["missing_id_examples"]=missing_ids[:10]; checks["extra_id_examples"]=extra_ids[:10]
        if missing_ids: errors.append(f"{len(missing_ids)} metadata acquisitions have no windows")
        if extra_ids: errors.append(f"{len(extra_ids)} window acquisition IDs are absent from metadata")
    checks["status"]="PASS" if not errors else "FAIL"; checks["errors"]=errors
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print("AICOS-HAR WINDOW VALIDATION V5"); print(f"Resolved NPZ: {wp}"); print(f"Windows: {len(X)}"); print(f"Manifest rows: {len(rows)}"); print(f"Unique acquisitions: {checks.get('unique_acquisitions',0)}"); print(f"Unique participants: {checks.get('unique_participants',0)}"); print(f"Status: {checks['status']}"); print(f"Report: {out}")
    if errors:
        for e in errors: print("ERROR:",e)
        raise SystemExit(1)

if __name__=="__main__": main()
