"""Use an honest pair-local model only as a high-precision negative veto on v89."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from catboost import CatBoostClassifier

ROOT=Path('.')
TRAIN=ROOT/'data/processed/v82_v34_features.parquet'
TEST_SCORE=ROOT/'data/processed/v83_independent_test_scores.parquet'
TEST_FEAT=ROOT/'data/processed/v21_test_features.parquet'
BASE=ROOT/'submissions/final_candidates_v89/FINAL_CANDIDATE_v89_v71_dual_veto_below_m2.csv'
HOLD=ROOT/'models/v76_trendyol_contrastive/holdout_terms.txt'
FEATURE_REPORT=ROOT/'reports/experiments/v83_independent_pairlocal_classifier.json'
OUT=ROOT/'submissions/final_candidates_v94';REPORT=ROOT/'reports/experiments/v94_high_precision_negative_guard.json'

def metrics(y,s,pct):
 rows=[]
 for q in [.02,.05,.08,.10,.15,.20,.25,.30,.40,.50]:
  m=pct<=q;rows.append({'bottom_pct':q,'rows':int(m.sum()),'negative_precision':float((1-y[m]).mean()),'positive_errors':int(y[m].sum())})
 for t in np.quantile(s,[.01,.02,.05,.10,.15,.20,.25,.30,.40,.50]):
  m=s<=t;rows.append({'raw_threshold':float(t),'rows':int(m.sum()),'negative_precision':float((1-y[m]).mean()),'positive_errors':int(y[m].sum())})
 return rows

def save(ids,pred,name,base,rows):
 p=OUT/f'FINAL_CANDIDATE_v94_{name}.csv';pd.DataFrame({'id':ids,'prediction':pred}).to_csv(p,index=False)
 rows.append({'variant':name,'file':str(p),'positives':int(pred.sum()),'changed_vs_base':int((pred!=base).sum()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

def main():
 rep=json.loads(FEATURE_REPORT.read_text());features=rep['features'];d=pd.read_parquet(TRAIN);d.term_id=d.term_id.astype(str)
 hold=set(HOLD.read_text().splitlines());tr=d[~d.term_id.isin(hold)];va=d[d.term_id.isin(hold)].copy()
 model=CatBoostClassifier(loss_function='Logloss',iterations=760,learning_rate=.04,depth=8,l2_leaf_reg=10,
  random_strength=.7,bootstrap_type='Bernoulli',subsample=.85,random_seed=20260706,verbose=200,allow_writing_files=False,task_type='GPU',devices='0')
 model.fit(tr[features].fillna(-1).astype('float32'),tr.label.astype('int8'))
 s=model.predict_proba(va[features].fillna(-1).astype('float32'))[:,1]
 va['s']=s;va['pct']=va.groupby('term_id').s.rank(method='average',pct=True)
 calibration=metrics(va.label.to_numpy(np.int8),s,va.pct.to_numpy())

 ts=pd.read_parquet(TEST_SCORE);tf=pd.read_parquet(TEST_FEAT,columns=['id','term_id','query_has_digit','digit_mismatch','query_has_size','size_mismatch'])
 b=pd.read_csv(BASE);ts.id=ts.id.astype(str);tf.id=tf.id.astype(str);b.id=b.id.astype(str)
 if not ts.id.equals(b.id) or not tf.id.equals(b.id):raise RuntimeError('alignment')
 ts['pct']=ts.groupby('term_id').v82_score.rank(method='average',pct=True)
 base=b.prediction.to_numpy(np.int8);hard=(tf.query_has_digit.eq(1)&tf.digit_mismatch.eq(1))|(tf.query_has_size.eq(1)&tf.size_mismatch.eq(1))
 guarded=base.copy();guarded[hard.to_numpy()&(base==1)]=0
 OUT.mkdir(parents=True,exist_ok=True);rows=[];save(b.id,guarded,'constraint_digit_size',base,rows)
 for pct in [.05,.08,.10,.15,.20]:
  pool=np.flatnonzero((guarded==1)&(ts.pct.to_numpy()<=pct));pool=pool[np.argsort(ts.v82_score.to_numpy()[pool])]
  for cap in [2500,5000,10000,20000]:
   pred=guarded.copy();pred[pool[:cap]]=0;save(b.id,pred,f'constraint_lexneg_pct{str(pct).replace(".","p")}_cap{cap}',base,rows)
 report={'holdout_calibration':calibration,'constraint_removed':int((hard.to_numpy()&(base==1)).sum()),'candidates':rows}
 REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
