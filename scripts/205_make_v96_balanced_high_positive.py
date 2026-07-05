"""Calibrate high-positive tail and refill only consensus candidates removed by v95 veto."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
ROOT=Path('.');TRAIN=ROOT/'data/processed/v82_v34_features.parquet';HOLD=ROOT/'models/v76_trendyol_contrastive/holdout_terms.txt';FEATURE_REPORT=ROOT/'reports/experiments/v83_independent_pairlocal_classifier.json';SCORE=ROOT/'data/processed/v83_independent_test_scores.parquet';RANK=ROOT/'data/processed/v72_anchorless_trendlex_features.parquet';BASE=ROOT/'submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv';OUT=ROOT/'submissions/final_candidates_v96';REPORT=ROOT/'reports/experiments/v96_balanced_high_positive.json'
def main():
 features=json.loads(FEATURE_REPORT.read_text())['features'];d=pd.read_parquet(TRAIN);d.term_id=d.term_id.astype(str);hold=set(HOLD.read_text().splitlines());tr=d[~d.term_id.isin(hold)];va=d[d.term_id.isin(hold)]
 m=CatBoostClassifier(loss_function='Logloss',iterations=760,learning_rate=.04,depth=8,l2_leaf_reg=10,random_strength=.7,bootstrap_type='Bernoulli',subsample=.85,random_seed=20260707,verbose=250,allow_writing_files=False,task_type='GPU',devices='0');m.fit(tr[features].fillna(-1).astype('float32'),tr.label.astype('int8'));sv=m.predict_proba(va[features].fillna(-1).astype('float32'))[:,1];y=va.label.to_numpy(np.int8);cal=[]
 for q in [.50,.60,.70,.80,.85,.90,.95,.97,.98,.99]:
  t=float(np.quantile(sv,q));mask=sv>=t;cal.append({'top_fraction':1-q,'threshold':t,'rows':int(mask.sum()),'positive_precision':float(y[mask].mean()),'negative_errors':int((1-y[mask]).sum())})
 s=pd.read_parquet(SCORE);r=pd.read_parquet(RANK,columns=['id','trendyol_qpct','lex_qpct']);b=pd.read_csv(BASE);s.id=s.id.astype(str);r.id=r.id.astype(str);b.id=b.id.astype(str)
 if not s.id.equals(b.id) or not r.id.equals(b.id):raise RuntimeError('alignment')
 base=b.prediction.to_numpy(np.int8);cons=(r.trendyol_qpct.ge(.75)&r.lex_qpct.ge(.75));pool=np.flatnonzero((base==0)&cons.to_numpy());blend=s.v82_score.to_numpy()+.15*r.trendyol_qpct.to_numpy()+.15*r.lex_qpct.to_numpy();pool=pool[np.argsort(-blend[pool])];OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for cap in [2500,5000,7500,10000,15000,17500]:
  pred=base.copy();pred[pool[:cap]]=1;p=OUT/f'FINAL_CANDIDATE_v96_consensus_add{cap}.csv';pd.DataFrame({'id':b.id,'prediction':pred}).to_csv(p,index=False);rows.append({'cap':cap,'used':int(min(cap,len(pool))),'pool':int(len(pool)),'positives':int(pred.sum()),'ratio':float(pred.mean()),'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 rep={'holdout_high_tail':cal,'test_consensus_pool':int(len(pool)),'candidates':rows};REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(rep,indent=2),encoding='utf8');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
