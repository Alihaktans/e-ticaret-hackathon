from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


ROOT=Path('.')
TRAIN=ROOT/'data/raw/training_pairs.csv'; TEST=ROOT/'data/raw/submission_pairs.csv'; ITEMS=ROOT/'data/raw/items.csv'; TERMS=ROOT/'data/raw/terms.csv'; SAMPLE=ROOT/'data/raw/sample_submission.csv'
V55=ROOT/'data/processed/v55_lexical_intent_guard_pairs.parquet'
ANCHORS={
 'raw_v35':ROOT/'submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv',
 'v40_history2200':ROOT/'submissions/final_candidates_v40/A_v40_raw35_history_impact2200.csv'}
OUT_QUERY=ROOT/'data/processed/v56_query_root_category_prob.parquet'; OUT_POOL=ROOT/'data/processed/v56_root_category_guard_pairs.parquet'
OUT_REVIEW=ROOT/'reports/manual_review/review_v56_root_category_guard_swaps.csv'; OUT_SUMMARY=ROOT/'reports/manual_review/v56_root_category_guard_summary.csv'; OUT_AUDIT=ROOT/'reports/manual_review/v56_root_category_guard_audit.json'; SUB=ROOT/'submissions/final_candidates_v56'

TR=str.maketrans({'ı':'i','İ':'i','ş':'s','Ş':'s','ğ':'g','Ğ':'g','ü':'u','Ü':'u','ö':'o','Ö':'o','ç':'c','Ç':'c'})
def norm(x):
 x=str(x)
 try:
  x=x.encode('latin1').decode('utf-8')
 except (UnicodeEncodeError,UnicodeDecodeError):
  pass
 x=x.translate(TR).lower();x=unicodedata.normalize('NFKD',x);x=re.sub(r'[^a-z0-9]+',' ',x);return re.sub(r'\s+',' ',x).strip()

def train_category_model():
 terms=pd.read_csv(TERMS);terms.term_id=terms.term_id.astype(str);terms['query_norm']=terms['query'].map(norm)
 items=pd.read_csv(ITEMS,usecols=['item_id','category']);items.item_id=items.item_id.astype(str);items['root']=items.category.fillna('unknown').astype(str).str.split('/').str[0].map(norm)
 tr=pd.read_csv(TRAIN,usecols=['term_id','item_id']);tr.term_id=tr.term_id.astype(str);tr.item_id=tr.item_id.astype(str)
 tr=tr.merge(items[['item_id','root']],on='item_id',how='left').merge(terms[['term_id','query_norm']],on='term_id',how='left')
 counts=tr.groupby(['term_id','query_norm','root']).size().rename('n').reset_index();best=counts.sort_values(['term_id','n'],ascending=[True,False]).drop_duplicates('term_id')
 word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=120000,sublinear_tf=True);char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=180000,sublinear_tf=True)
 idx_train,idx_val=train_test_split(np.arange(len(best)),test_size=.22,random_state=42,stratify=best.root)
 Xw=word.fit_transform(best.query_norm);Xc=char.fit_transform(best.query_norm);X=hstack([Xw,Xc],format='csr')
 model=OneVsRestClassifier(LogisticRegression(C=3.0,max_iter=1000,class_weight='balanced',solver='liblinear'),n_jobs=1)
 model.fit(X[idx_train],best.root.iloc[idx_train]);pv=model.predict_proba(X[idx_val]);yv=best.root.iloc[idx_val]
 audit={'n_train_queries':int(len(best)),'n_roots':int(best.root.nunique()),'val_accuracy':float(accuracy_score(yv,model.classes_[pv.argmax(1)])),'val_top2_accuracy':float(top_k_accuracy_score(yv,pv,k=2,labels=model.classes_)),'val_logloss':float(log_loss(yv,pv,labels=model.classes_))}
 # Refit vectorizers remain fixed, model uses every training query.
 model.fit(X,best.root)
 test_terms=set(pd.read_csv(TEST,usecols=['term_id']).term_id.astype(str));tt=terms[terms.term_id.isin(test_terms)].drop_duplicates('term_id').copy();Xt=hstack([word.transform(tt.query_norm),char.transform(tt.query_norm)],format='csr');p=model.predict_proba(Xt)
 order=np.argsort(-p,axis=1);tt['root_top1']=model.classes_[order[:,0]];tt['root_top2']=model.classes_[order[:,1]];tt['root_p1']=p[np.arange(len(p)),order[:,0]].astype(np.float32);tt['root_p2']=p[np.arange(len(p)),order[:,1]].astype(np.float32)
 # Store compact dictionaries as JSON so any item's root probability can be recovered.
 tt['root_prob_json']=[json.dumps({str(model.classes_[j]):round(float(row[j]),7) for j in np.flatnonzero(row>=.01)},ensure_ascii=False) for row in p]
 tt[['term_id','root_top1','root_top2','root_p1','root_p2','root_prob_json']].to_parquet(OUT_QUERY,index=False)
 return audit,tt[['term_id','root_top1','root_top2','root_p1','root_p2','root_prob_json']],items

def enrich_review(d,terms,items):
 d=d.merge(terms[['term_id','query']],on='term_id',how='left');meta=items.drop_duplicates('item_id')
 for side in ['add','drop']:
  ren={'item_id':'item_id_'+side,'title':'title_'+side,'category':'category_'+side,'root':'root_'+side}
  d=d.merge(meta[['item_id','title','category','root']].rename(columns=ren),on='item_id_'+side,how='left')
 front=['anchor_name','mode','query','id_add','title_add','category_add','root_prob_add','id_drop','title_drop','category_drop','root_prob_drop','v55_prob','root_gain','robust_gain']
 return d[[c for c in front if c in d]+[c for c in d if c not in front]]

def main():
 for p in [OUT_QUERY.parent,OUT_REVIEW.parent,SUB]:p.mkdir(parents=True,exist_ok=True)
 audit,qprob,items_small=train_category_model();print('category audit',audit)
 # Reload titles only once for review; category model used the compact item frame above.
 titles=pd.read_csv(ITEMS,usecols=['item_id','title','category']);titles.item_id=titles.item_id.astype(str);titles['root']=titles.category.fillna('unknown').astype(str).str.split('/').str[0].map(norm)
 pool=pd.read_parquet(V55);pool=pool.merge(qprob,on='term_id',how='left')
 roots=titles[['item_id','root']].drop_duplicates('item_id')
 pool=pool.merge(roots.rename(columns={'item_id':'item_id_add','root':'root_add'}),on='item_id_add',how='left').merge(roots.rename(columns={'item_id':'item_id_drop','root':'root_drop'}),on='item_id_drop',how='left')
 maps=pool.root_prob_json.fillna('{}').map(json.loads)
 pool['root_prob_add']=np.array([m.get(r,0.0) for m,r in zip(maps,pool.root_add)],dtype=np.float32);pool['root_prob_drop']=np.array([m.get(r,0.0) for m,r in zip(maps,pool.root_drop)],dtype=np.float32)
 pool['root_gain']=pool.root_prob_add-pool.root_prob_drop;pool['same_root']=pool.root_add.eq(pool.root_drop)
 pool.to_parquet(OUT_POOL,index=False)
 sample=pd.read_csv(SAMPLE,usecols=['id']);sample.id=sample.id.astype(str);idx=pd.Series(np.arange(len(sample)),index=sample.id);summ=[];reviews=[]
 # Keep the V55 lexical-intent gates, then add independently learned root-category consistency.
 basegate=(pool.diff_v38_lex_score>0)&(pool.diff_v33_ft_score>0)&(pool.diff_v38_word_overlap_full>=0)&(pool.diff_v38_must_token_coverage_full>=0)&(pool.diff_v38_must_missing_ratio<=0)&(pool.diff_v38_special_token_coverage_full>=0)&(pool.diff_v43_query_pop_support>=0)&(pool.robust_gain>=.15)
 root_is_top2=pool.root_add.eq(pool.root_top1)|pool.root_add.eq(pool.root_top2)
 root_safe=pool.same_root|((pool.root_prob_add>=pool.root_prob_drop+.04)&(pool.root_prob_add>=.08)&root_is_top2)
 for an,path in ANCHORS.items():
  a=pd.read_csv(path);a.id=a.id.astype(str);anchor=a.prediction.astype(np.int8).to_numpy();b=pool[pool.anchor_name.eq(an)].copy();g=basegate.loc[b.index]&root_safe.loc[b.index]
  modes={'ultra':g&(b.v55_prob>=.97)&(b.diff_v47_forward_base>0)&(b.diff_v47_reverse_base>0),'safe':g&(b.v55_prob>=.95)&((b.diff_v47_forward_base>0)|(b.diff_v47_reverse_base>0)),'balanced':g&(b.v55_prob>=.92)}
  for mode,mask in modes.items():
   sw=b[mask].sort_values(['v55_prob','root_gain','robust_gain'],ascending=False).drop_duplicates('term_id');print(an,mode,len(sw));x=sw.head(150).copy();x['mode']=mode;reviews.append(x)
   for cap in [100,250,500,1000]:
    take=sw.head(cap)
    if take.empty:continue
    pred=anchor.copy();pred[idx.loc[take.id_add].to_numpy()]=1;pred[idx.loc[take.id_drop].to_numpy()]=0;name=f'v56_{an}_{mode}_cap{cap}';out=SUB/(name+'.csv');pd.DataFrame({'id':sample.id,'prediction':pred}).to_csv(out,index=False)
    summ.append({'variant':name,'anchor':an,'mode':mode,'accepted_pool':len(sw),'used_swaps':len(take),'v55_prob_mean':float(take.v55_prob.mean()),'root_gain_mean':float(take.root_gain.mean()),'different_root_rate':float((~take.same_root).mean()),'diff_vs_anchor':int((pred!=anchor).sum()),'ones':int(pred.sum()),'pos_ratio':float(pred.mean()),'file':str(out)})
 terms=pd.read_csv(TERMS);terms.term_id=terms.term_id.astype(str);rev=pd.concat(reviews,ignore_index=True).drop_duplicates(['anchor_name','id_add','id_drop']);enrich_review(rev,terms,titles).to_csv(OUT_REVIEW,index=False);pd.DataFrame(summ).to_csv(OUT_SUMMARY,index=False);OUT_AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(pd.DataFrame(summ).to_string(index=False));print('outputs',OUT_REVIEW,OUT_SUMMARY,OUT_AUDIT)

if __name__=='__main__':main()
