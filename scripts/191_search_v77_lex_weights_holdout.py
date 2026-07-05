"""Search base/v77/v38 rank weights on the untouched candidate-shaped holdout."""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
ROOT=Path('.');DATA=ROOT/'data/processed/v34/v34_hard_negative_pairs.parquet';SCORES=ROOT/'data/processed/v77_holdout_scores.parquet';HOLD=ROOT/'models/v76_trendyol_contrastive/holdout_terms.txt';OUT=ROOT/'reports/experiments/v77_lex_weight_search.csv';REPORT=ROOT/'reports/experiments/v77_lex_weight_search.json'
def load_lex():
 spec=importlib.util.spec_from_file_location('v38lex',ROOT/'scripts/132_score_v38_lexical_signal_fast.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 m=load_lex();hold=set(HOLD.read_text().splitlines());d=pd.read_parquet(DATA,columns=['term_id','item_id','label']).drop_duplicates(['term_id','item_id']);d.term_id=d.term_id.astype(str);d.item_id=d.item_id.astype(str);d=d[d.term_id.isin(hold)].reset_index(drop=True);d['id']=np.arange(len(d)).astype(str)
 terms=pd.read_csv(ROOT/'data/raw/terms.csv');terms.term_id=terms.term_id.astype(str);tf=m.build_term_features(terms[terms.term_id.isin(hold)])
 needed=set(d.item_id);parts=[]
 for c in pd.read_csv(ROOT/'data/raw/items.csv',chunksize=100000,low_memory=False):
  c.item_id=c.item_id.astype(str);m0=c.item_id.isin(needed)
  if m0.any():parts.append(c.loc[m0].copy())
 it=pd.concat(parts).drop_duplicates('item_id');it['title_norm']=it.title.map(m.norm_text);it['category_norm']=it.category.map(m.norm_text);it['brand_norm']=it.brand.map(m.norm_text);a=it.attributes.fillna('').astype(str).str.slice(0,400).map(m.norm_text);it['full_norm']=(it.title_norm+' '+it.category_norm+' '+it.brand_norm+' '+a).str.strip()
 frame=d.merge(tf,on='term_id').merge(it[['item_id','title_norm','category_norm','brand_norm','full_norm']],on='item_id');lex=m.score_rows(frame)[['term_id','item_id','v38_lex_score']]
 scores=pd.read_parquet(SCORES,columns=['term_id','item_id','base_score','v77_score']);scores.term_id=scores.term_id.astype(str);scores.item_id=scores.item_id.astype(str);x=d.merge(scores,on=['term_id','item_id']).merge(lex,on=['term_id','item_id']);
 for c in ['base_score','v77_score','v38_lex_score']:x[c+'_q']=x.groupby('term_id')[c].rank(method='average',pct=True)
 groups=list(x.groupby('term_id',sort=False).indices.values());y=x.label.to_numpy(np.int8);b=x.base_score_q.to_numpy();v=x.v77_score_q.to_numpy();l=x.v38_lex_score_q.to_numpy();rows=[]
 for wb in np.arange(0,1.001,.05):
  for wv in np.arange(0,1.001-wb,.05):
   wl=1-wb-wv;s=wb*b+wv*v+wl*l;correct=0;rec=[];aps=[]
   from sklearn.metrics import average_precision_score
   for idx in groups:
    yy=y[idx];ss=s[idx];k=int(yy.sum());o=np.argsort(-ss);p=np.zeros(len(idx),np.int8);p[o[:k]]=1;correct+=int((p==yy).sum());rec.append(float(yy[o[:k]].sum()/max(1,k)));aps.append(average_precision_score(yy,ss))
   rows.append({'w_base':round(float(wb),2),'w_v77':round(float(wv),2),'w_lex':round(float(wl),2),'accuracy':correct/len(y),'recall_at_true_k':np.mean(rec),'query_map':np.mean(aps)})
 out=pd.DataFrame(rows).sort_values(['query_map','recall_at_true_k'],ascending=False);out.to_csv(OUT,index=False);REPORT.write_text(json.dumps({'top':out.head(25).to_dict('records'),'queries':len(groups),'pairs':len(x),'holdout_disjoint':True},indent=2),encoding='utf8');print(out.head(30).to_string(index=False))
if __name__=='__main__':main()
