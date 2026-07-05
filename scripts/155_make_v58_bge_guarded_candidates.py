from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('.');POOL=ROOT/'data/processed/v57_exact_title_guard_pairs.parquet';BGE=ROOT/'data/processed/v58_bge_exact_pool_scores.parquet';SAMPLE=ROOT/'data/raw/sample_submission.csv';TERMS=ROOT/'data/raw/terms.csv';ITEMS=ROOT/'data/raw/items.csv'
ANCHORS={'raw_v35':ROOT/'submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv','v40_history2200':ROOT/'submissions/final_candidates_v40/A_v40_raw35_history_impact2200.csv'}
OUT_REV=ROOT/'reports/manual_review/review_v58_bge_guarded_swaps.csv';OUT_SUM=ROOT/'reports/manual_review/v58_bge_guarded_summary.csv';OUT_POOL=ROOT/'data/processed/v58_bge_guarded_pairs.parquet';SUB=ROOT/'submissions/final_candidates_v58'
def main():
 for p in [OUT_REV.parent,OUT_POOL.parent,SUB]:p.mkdir(parents=True,exist_ok=True)
 p=pd.read_parquet(POOL);s=pd.read_parquet(BGE,columns=['id','v58_bge_raw']);s.id=s.id.astype(str)
 p=p.merge(s.rename(columns={'id':'id_add','v58_bge_raw':'bge_add'}),on='id_add',how='left').merge(s.rename(columns={'id':'id_drop','v58_bge_raw':'bge_drop'}),on='id_drop',how='left');p['bge_gain']=p.bge_add-p.bge_drop;p.to_parquet(OUT_POOL,index=False)
 root_top2=p.root_add.eq(p.root_top1)|p.root_add.eq(p.root_top2);root_safe=p.same_root|((p.root_prob_add>=p.root_prob_drop+.04)&(p.root_prob_add>=.08)&root_top2)
 lexical=(p.diff_v38_lex_score>0)&(p.diff_v33_ft_score>0)&(p.diff_v38_word_overlap_full>=0)&(p.diff_v38_must_token_coverage_full>=0)&(p.diff_v38_must_missing_ratio<=0)&(p.diff_v38_special_token_coverage_full>=0)&(p.diff_v43_query_pop_support>=0)&(p.robust_gain>=.15)
 exact=((p.v38_title_missing_ratio_add<=1e-6)|(p.v38_exact_phrase_title_add>=1))&(p.v38_word_overlap_title_add>=p.v38_word_overlap_title_drop)&(p.query_token_count>=2)
 entities=(p.diff_v38_number_match>=0)&(p.diff_v38_model_match>=0)&(p.diff_v38_brand_match>=0);base=lexical&root_safe&exact&entities
 sample=pd.read_csv(SAMPLE,usecols=['id']);sample.id=sample.id.astype(str);idx=pd.Series(np.arange(len(sample)),index=sample.id);summ=[];reviews=[]
 for an,path in ANCHORS.items():
  a=pd.read_csv(path);a.id=a.id.astype(str);anchor=a.prediction.astype(np.int8).to_numpy();b=p[p.anchor_name.eq(an)].copy();g=base.loc[b.index]
  modes={'bge_ultra':g&(b.v55_prob>=.95)&(b.bge_add>=2.0)&(b.bge_gain>=2.0),'bge_strict':g&(b.v55_prob>=.95)&(b.bge_add>=1.0)&(b.bge_gain>=2.0),'bge_safe':g&(b.v55_prob>=.92)&(b.bge_add>=0.0)&(b.bge_gain>=1.5)}
  for mode,mask in modes.items():
   sw=b[mask].sort_values(['bge_add','bge_gain','v55_prob'],ascending=False).drop_duplicates('term_id');print(an,mode,len(sw));x=sw.head(250).copy();x['mode']=mode;reviews.append(x)
   for cap in [50,100,250,500]:
    take=sw.head(cap)
    if take.empty:continue
    pred=anchor.copy();pred[idx.loc[take.id_add].to_numpy()]=1;pred[idx.loc[take.id_drop].to_numpy()]=0;name=f'v58_{an}_{mode}_cap{cap}';out=SUB/(name+'.csv');pd.DataFrame({'id':sample.id,'prediction':pred}).to_csv(out,index=False)
    summ.append({'variant':name,'anchor':an,'mode':mode,'accepted_pool':len(sw),'used_swaps':len(take),'bge_add_mean':float(take.bge_add.mean()),'bge_gain_mean':float(take.bge_gain.mean()),'v55_prob_mean':float(take.v55_prob.mean()),'diff_vs_anchor':int((pred!=anchor).sum()),'ones':int(pred.sum()),'pos_ratio':float(pred.mean()),'file':str(out)})
 terms=pd.read_csv(TERMS,usecols=['term_id','query']);terms.term_id=terms.term_id.astype(str);items=pd.read_csv(ITEMS,usecols=['item_id','title','category']);items.item_id=items.item_id.astype(str);rev=pd.concat(reviews,ignore_index=True).drop_duplicates(['anchor_name','id_add','id_drop']).merge(terms,on='term_id',how='left')
 for side in ['add','drop']:rev=rev.merge(items.drop_duplicates('item_id').rename(columns={'item_id':'item_id_'+side,'title':'title_'+side,'category':'category_'+side}),on='item_id_'+side,how='left')
 front=['anchor_name','mode','query','id_add','title_add','category_add','bge_add','id_drop','title_drop','category_drop','bge_drop','bge_gain','v55_prob'];rev[[c for c in front if c in rev]+[c for c in rev if c not in front]].to_csv(OUT_REV,index=False);pd.DataFrame(summ).to_csv(OUT_SUM,index=False);print(pd.DataFrame(summ).to_string(index=False));print('outputs',OUT_REV,OUT_SUM)
if __name__=='__main__':main()
