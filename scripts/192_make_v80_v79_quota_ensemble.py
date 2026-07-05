"""Final quota-preserving candidates from v79 and the independently scored v77."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path('.');F=ROOT/'data/processed/v72_anchorless_trendlex_features.parquet';V77=ROOT/'data/processed/v78_v77_rank_features.parquet';V79=ROOT/'data/processed/v79_trendyol_hardnegative/pair_scores.npy';BASE=ROOT/'submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv';V71=ROOT/'submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv';OUT=ROOT/'submissions/final_candidates_v80';REPORT=ROOT/'reports/experiments/v80_v79_quota_ensemble.json'
def h(x):return hashlib.md5(x.encode()).hexdigest()[:8]
def top(term,s,b):
 w=pd.DataFrame({'t':term,'s':s});r=w.groupby('t',sort=False).s.rank(method='first',ascending=False);return (r.to_numpy()<=w.t.map(b).to_numpy()).astype(np.int8)
def main():
 OUT.mkdir(parents=True,exist_ok=True);f=pd.read_parquet(F);r77=pd.read_parquet(V77,columns=['id','v77_qpct']);r77.id=r77.id.astype(str);f.id=f.id.astype(str);f.term_id=f.term_id.astype(str);f=f.merge(r77,on='id',validate='one_to_one');s=np.load(V79,mmap_mode='r');f['v79_score']=np.asarray(s,np.float32);f['v79_qpct']=f.groupby('term_id').v79_score.rank(method='average',pct=True).astype(np.float32)
 b=pd.read_csv(BASE);v=pd.read_csv(V71);b.id=b.id.astype(str);v.id=v.id.astype(str);base=b.prediction.to_numpy(np.int8);v71=v.prediction.to_numpy(np.int8);bud=pd.Series(base,index=f.term_id).groupby(level=0).sum();a=f.trendyol_qpct.to_numpy();x=f.lex_qpct.to_numpy();q77=f.v77_qpct.to_numpy();q79=f.v79_qpct.to_numpy();avg=.5*(q77+q79)
 specs={'v79_only':q79,'v79_95_lex05':.95*q79+.05*x,'v79_90_lex10':.9*q79+.1*x,'v77v79_avg':avg,'v77v79_avg95_lex05':.95*avg+.05*x,'base25_v79_25_lex50':.25*a+.25*q79+.5*x,'base25_ftavg25_lex50':.25*a+.25*avg+.5*x}
 rows=[]
 for n,s0 in specs.items():
  p=top(f.term_id,s0,bud);path=OUT/f'FINAL_CANDIDATE_v80_{n}_{h(n)}.csv';pd.DataFrame({'id':f.id,'prediction':p}).to_csv(path,index=False);rows.append({'variant':n,'file':str(path),'changed_vs_v22':int((p!=base).sum()),'changed_vs_v71':int((p!=v71).sum()),'positives':int(p.sum()),'positive_ratio':float(p.mean()),'jaccard_v71':float(np.logical_and(p==1,v71==1).sum()/np.logical_or(p==1,v71==1).sum())})
 corr=f[['trendyol_qpct','lex_qpct','v77_qpct','v79_qpct']].sample(250000,random_state=80).corr(method='spearman').to_dict();REPORT.write_text(json.dumps({'v79_holdout':json.loads((ROOT/'reports/experiments/v79_holdout_validation.json').read_text()),'candidates':rows,'correlation':corr,'invariant':'v22 per-query budgets preserved'},indent=2),encoding='utf8');print(pd.DataFrame(rows).sort_values('changed_vs_v71').to_string(index=False));print(json.dumps(corr,indent=2))
if __name__=='__main__':main()
