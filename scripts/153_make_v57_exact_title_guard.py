from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('.')
POOL=ROOT/'data/processed/v56_root_category_guard_pairs.parquet'; V38=ROOT/'data/processed/v38_lexical_pair_scores.parquet'
SAMPLE=ROOT/'data/raw/sample_submission.csv'; TERMS=ROOT/'data/raw/terms.csv'; ITEMS=ROOT/'data/raw/items.csv'
ANCHORS={'raw_v35':ROOT/'submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv','v40_history2200':ROOT/'submissions/final_candidates_v40/A_v40_raw35_history_impact2200.csv'}
OUT_POOL=ROOT/'data/processed/v57_exact_title_guard_pairs.parquet';OUT_REV=ROOT/'reports/manual_review/review_v57_exact_title_guard_swaps.csv';OUT_SUM=ROOT/'reports/manual_review/v57_exact_title_guard_summary.csv';SUB=ROOT/'submissions/final_candidates_v57'

def main():
 for p in [OUT_POOL.parent,OUT_REV.parent,SUB]:p.mkdir(parents=True,exist_ok=True)
 pool=pd.read_parquet(POOL);ids=set(pool.id_add.astype(str))|set(pool.id_drop.astype(str))
 cols=['v38_exact_phrase_title','v38_word_overlap_title','v38_title_missing_ratio','v38_number_match','v38_model_match','v38_brand_match']
 d=pd.read_parquet(V38,columns=['id']+cols);d.id=d.id.astype(str);d=d[d.id.isin(ids)].drop_duplicates('id')
 for side in ['add','drop']:
  ren={'id':'id_'+side,**{c:c+'_'+side for c in cols}};pool=pool.merge(d.rename(columns=ren),on='id_'+side,how='left')
 for c in cols:pool['diff_'+c]=pool[c+'_add']-pool[c+'_drop']
 terms=pd.read_csv(TERMS);terms.term_id=terms.term_id.astype(str);terms['query_token_count']=terms['query'].fillna('').astype(str).str.findall(r'\w+').str.len()
 pool=pool.merge(terms[['term_id','query','query_token_count']],on='term_id',how='left')
 pool.to_parquet(OUT_POOL,index=False)
 root_top2=pool.root_add.eq(pool.root_top1)|pool.root_add.eq(pool.root_top2)
 root_safe=pool.same_root|((pool.root_prob_add>=pool.root_prob_drop+.04)&(pool.root_prob_add>=.08)&root_top2)
 lexical=(pool.diff_v38_lex_score>0)&(pool.diff_v33_ft_score>0)&(pool.diff_v38_word_overlap_full>=0)&(pool.diff_v38_must_token_coverage_full>=0)&(pool.diff_v38_must_missing_ratio<=0)&(pool.diff_v38_special_token_coverage_full>=0)&(pool.diff_v43_query_pop_support>=0)&(pool.robust_gain>=.15)
 # Require every meaningful query token in the title; an exact phrase is even safer.
 exact=((pool.v38_title_missing_ratio_add<=1e-6)|(pool.v38_exact_phrase_title_add>=1))&(pool.v38_word_overlap_title_add>=pool.v38_word_overlap_title_drop)&(pool.query_token_count>=2)
 # Numeric/model/brand constraints may never become worse after a swap.
 entities=(pool.diff_v38_number_match>=0)&(pool.diff_v38_model_match>=0)&(pool.diff_v38_brand_match>=0)
 sample=pd.read_csv(SAMPLE,usecols=['id']);sample.id=sample.id.astype(str);idx=pd.Series(np.arange(len(sample)),index=sample.id);summ=[];reviews=[]
 for an,path in ANCHORS.items():
  a=pd.read_csv(path);a.id=a.id.astype(str);anchor=a.prediction.astype(np.int8).to_numpy();b=pool[pool.anchor_name.eq(an)].copy();base=(lexical&root_safe&exact&entities).loc[b.index]
  modes={'ultra':base&(b.v55_prob>=.97)&(b.diff_v47_forward_base>0)&(b.diff_v47_reverse_base>0),'safe':base&(b.v55_prob>=.95)&((b.diff_v47_forward_base>0)|(b.diff_v47_reverse_base>0))}
  for mode,mask in modes.items():
   sw=b[mask].sort_values(['v55_prob','robust_gain'],ascending=False).drop_duplicates('term_id');print(an,mode,len(sw));x=sw.head(200).copy();x['mode']=mode;reviews.append(x)
   for cap in [100,250,500,1000]:
    take=sw.head(cap)
    if take.empty:continue
    pred=anchor.copy();pred[idx.loc[take.id_add].to_numpy()]=1;pred[idx.loc[take.id_drop].to_numpy()]=0;name=f'v57_{an}_{mode}_cap{cap}';out=SUB/(name+'.csv');pd.DataFrame({'id':sample.id,'prediction':pred}).to_csv(out,index=False)
    summ.append({'variant':name,'anchor':an,'mode':mode,'accepted_pool':len(sw),'used_swaps':len(take),'v55_prob_mean':float(take.v55_prob.mean()),'exact_phrase_rate':float((take.v38_exact_phrase_title_add>=1).mean()),'diff_vs_anchor':int((pred!=anchor).sum()),'ones':int(pred.sum()),'pos_ratio':float(pred.mean()),'file':str(out)})
 items=pd.read_csv(ITEMS,usecols=['item_id','title','category']);items.item_id=items.item_id.astype(str);meta=items.drop_duplicates('item_id');rev=pd.concat(reviews,ignore_index=True).drop_duplicates(['anchor_name','id_add','id_drop'])
 for side in ['add','drop']:rev=rev.merge(meta.rename(columns={'item_id':'item_id_'+side,'title':'title_'+side,'category':'category_'+side}),on='item_id_'+side,how='left')
 front=['anchor_name','mode','query','id_add','title_add','category_add','id_drop','title_drop','category_drop','v55_prob','robust_gain','v38_exact_phrase_title_add','v38_title_missing_ratio_add'];rev[[c for c in front if c in rev]+[c for c in rev if c not in front]].to_csv(OUT_REV,index=False);pd.DataFrame(summ).to_csv(OUT_SUM,index=False);print(pd.DataFrame(summ).to_string(index=False));print('outputs',OUT_REV,OUT_SUM)
if __name__=='__main__':main()
