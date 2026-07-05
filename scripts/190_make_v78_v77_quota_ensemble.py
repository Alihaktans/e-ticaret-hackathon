"""Use validated v77 scores inside the proven v22 per-query budgets."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path('.');FEATURES=ROOT/'data/processed/v72_anchorless_trendlex_features.parquet';SCORES=ROOT/'data/processed/v77_trendyol_hardnegative/v77_pair_scores.npy';BASE=ROOT/'submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv';V71=ROOT/'submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv';OUT=ROOT/'submissions/final_candidates_v78';REPORT=ROOT/'reports/experiments/v78_v77_quota_ensemble.json';SUMMARY=ROOT/'reports/manual_review/v78_v77_quota_ensemble.csv';CACHE=ROOT/'data/processed/v78_v77_rank_features.parquet'
def short(x):return hashlib.md5(x.encode()).hexdigest()[:8]
def top(term,score,budgets):
 w=pd.DataFrame({'term_id':term,'score':score});r=w.groupby('term_id',sort=False).score.rank(method='first',ascending=False);return (r.to_numpy()<=w.term_id.map(budgets).to_numpy()).astype(np.int8)
def main():
 OUT.mkdir(parents=True,exist_ok=True);f=pd.read_parquet(FEATURES);f.id=f.id.astype(str);f.term_id=f.term_id.astype(str);s=np.load(SCORES,mmap_mode='r')
 if len(s)!=len(f) or not np.isfinite(s).all():raise RuntimeError('v77 score alignment/finite failure')
 f['v77_score']=np.asarray(s,np.float32);f['v77_qpct']=f.groupby('term_id',sort=False).v77_score.rank(method='average',pct=True).astype(np.float32);f[['id','term_id','item_id','v77_score','v77_qpct']].to_parquet(CACHE,index=False)
 b=pd.read_csv(BASE);v=pd.read_csv(V71);b.id=b.id.astype(str);v.id=v.id.astype(str)
 if not f.id.equals(b.id) or not f.id.equals(v.id):raise RuntimeError('alignment')
 base=b.prediction.to_numpy(np.int8);v71=v.prediction.to_numpy(np.int8);bud=pd.Series(base,index=f.term_id).groupby(level=0).sum();t=f.trendyol_qpct.to_numpy(np.float32);x=f.lex_qpct.to_numpy(np.float32);z=f.v77_qpct.to_numpy(np.float32)
 specs={'v77_only':z,'v77_95_lex05':.95*z+.05*x,'v77_90_lex10':.9*z+.1*x,'v77_85_lex15':.85*z+.15*x,'v77_80_lex20':.8*z+.2*x,'v77_50_lex50':.5*z+.5*x,'base25_v77_25_lex50':.25*t+.25*z+.5*x,'base15_v77_35_lex50':.15*t+.35*z+.5*x,'base10_v77_40_lex50':.1*t+.4*z+.5*x,'v77_35_lex65':.35*z+.65*x,'base15_v77_20_lex65':.15*t+.2*z+.65*x}
 rows=[]
 for name,score in specs.items():
  pred=top(f.term_id,score,bud);got=pd.Series(pred,index=f.term_id).groupby(level=0).sum()
  if not got.equals(bud):raise RuntimeError(name)
  p=OUT/f'FINAL_CANDIDATE_v78_{name}_{short(name)}.csv';pd.DataFrame({'id':f.id,'prediction':pred}).to_csv(p,index=False);cb=pred!=base;cv=pred!=v71
  rows.append({'variant':name,'file':str(p),'changed_vs_v22':int(cb.sum()),'changed_vs_v71':int(cv.sum()),'changed_terms_vs_v71':int(f.loc[cv,'term_id'].nunique()),'jaccard_v71':float(np.logical_and(pred==1,v71==1).sum()/np.logical_or(pred==1,v71==1).sum()),'positives':int(pred.sum()),'positive_ratio':float(pred.mean())})
 summary=pd.DataFrame(rows).sort_values('changed_vs_v71');summary.to_csv(SUMMARY,index=False);report={'v77_holdout_delta':json.loads((ROOT/'reports/experiments/v77_holdout_validation.json').read_text())['delta'],'spearman_sample':f[['trendyol_qpct','lex_qpct','v69_qpct','v77_qpct']].sample(250000,random_state=78).corr(method='spearman').to_dict(),'candidates':rows,'invariant':'every candidate preserves every v22 query budget'};REPORT.write_text(json.dumps(report,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(report['spearman_sample'],indent=2))
if __name__=='__main__':main()
