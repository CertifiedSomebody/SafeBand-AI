
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,json,joblib,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def met(y,p):
 e=p-y
 return {"n":len(y),"mae_bpm":float(mean_absolute_error(y,p)),"rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),
 "r2":float(r2_score(y,p)),"within_3_bpm_pct":float(np.mean(np.abs(e)<=3)*100),
 "within_5_bpm_pct":float(np.mean(np.abs(e)<=5)*100),"within_10_bpm_pct":float(np.mean(np.abs(e)<=10)*100),
 "bias_bpm":float(np.mean(e)),"median_abs_error_bpm":float(np.median(np.abs(e))),
 "p90_abs_error_bpm":float(np.percentile(np.abs(e),90)),"p95_abs_error_bpm":float(np.percentile(np.abs(e),95)),
 "max_abs_error_bpm":float(np.max(np.abs(e)))}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--input",required=True);p.add_argument("--model-out",required=True);p.add_argument("--report-out",required=True);p.add_argument("--predictions-out",required=True)
 p.add_argument("--test-subject",default="s9");p.add_argument("--alpha",type=float,default=.35);p.add_argument("--max-jump-bpm",type=float,default=25)
 a=p.parse_args(); df=pd.read_csv(a.input); test=df[df.subject==a.test_subject].copy(); dev=df[df.subject!=a.test_subject].copy()
 y=dev.target_hr.values; groups=dev.subject.values
 feat_cols=[c for c in df.columns if c not in ["subject","record","start_sample","target_hr"]]
 configs={"ppg_signal":["spectral_hr","peak_hr","autocorr_hr","candidate_hr","candidate_spread","ppg_std","ppg_rms"],
 "ppg_motion":["spectral_hr","peak_hr","autocorr_hr","candidate_hr","candidate_spread","ppg_std","ppg_rms","acc_mag_std","acc_mag_rms","gyro_mag_std","gyro_mag_rms"],
 "all":["spectral_hr","peak_hr","autocorr_hr","candidate_hr","candidate_spread","ppg_std","ppg_rms","acc_mag_std","acc_mag_rms","gyro_mag_std","gyro_mag_rms"]}
 models={"ridge":make_pipeline(StandardScaler(),Ridge(alpha=10)),"rf":RandomForestRegressor(n_estimators=700,min_samples_leaf=2,random_state=42,n_jobs=-1),
 "extra_trees":ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,random_state=42,n_jobs=-1),
 "hgb":HistGradientBoostingRegressor(max_iter=450,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1.5,random_state=42)}
 gkf=GroupKFold(5); results={}; best=None
 for cn,cols in configs.items():
  for mn,m in models.items():
   pred=np.full(len(dev),np.nan)
   for tr,va in gkf.split(dev,y,groups):
    Xtr=dev.iloc[tr][cols].fillna(dev.iloc[tr][cols].median()).fillna(0); Xva=dev.iloc[va][cols].fillna(Xtr.median()).fillna(0)
    m.fit(Xtr,y[tr]);pred[va]=m.predict(Xva)
   mm=met(y,pred);results[f"{cn}_{mn}"]=mm
   if best is None or mm["mae_bpm"]<best[0]: best=(mm["mae_bpm"],cn,mn)
 # Signal baseline and temporal-safe model evaluation
 signal=np.nanmedian(dev[["spectral_hr","peak_hr","autocorr_hr"]].values,axis=1)
 results["signal_candidate_baseline"]=met(y,signal)
 cn,mn=best[1],best[2]; cols=configs[cn]; X=dev[cols].fillna(dev[cols].median()).fillna(0)
 model=models[mn]; model.fit(X,y)
 Xt=test[cols].fillna(X.median()).fillna(0); raw=model.predict(Xt)
 # bounded temporal smoothing only; no learned affine calibration.
 smooth=[]
 prev=None
 for x in raw:
  z=float(x)
  if prev is not None: z=np.clip(z,prev-a.max_jump_bpm,prev+a.max_jump_bpm); z=a.alpha*z+(1-a.alpha)*prev
  smooth.append(z);prev=z
 smooth=np.asarray(smooth)
 testm=met(test.target_hr.values,smooth)
 out=pd.DataFrame({"subject":test.subject,"record":test.record,"start_sample":test.start_sample,"target_hr":test.target_hr,"prediction_raw":raw,"prediction_smoothed":smooth})
 Path(a.predictions_out).parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.predictions_out,index=False)
 artifact={"model":model,"feature_columns":cols,"version":"PPG WALKING V5.3 FINAL","alpha":a.alpha,"max_jump_bpm":a.max_jump_bpm}
 Path(a.model_out).parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,a.model_out)
 report={"version":"PPG WALKING V5.3 FINAL","objective":"Final walking HR estimator; signal-derived candidates plus bounded temporal stabilization.","test_subject":a.test_subject,
 "development_subjects":sorted(dev.subject.unique()),"parameters":{"window_sec":8,"shift_sec":2,"fs_hz":256,"ppg_band_hz":[.5,5],"hr_range_bpm":[40,220],"cv":"5-fold GroupKFold","selection_metric":"development OOF MAE","temporal_alpha":a.alpha,"max_jump_bpm":a.max_jump_bpm,"affine_calibration":False},
 "results":results,"selected":{"variant":cn,"model":mn,"oof_mae_bpm":best[0]},"held_out_test":testm}
 Path(a.report_out).parent.mkdir(parents=True,exist_ok=True);Path(a.report_out).write_text(json.dumps(report,indent=2))
 print(json.dumps({"selected":report["selected"],"held_out_test":testm},indent=2))
if __name__=="__main__": main()
