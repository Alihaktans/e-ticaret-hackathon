from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path('.');POOL=ROOT/'data/processed/v57_exact_title_guard_pairs.parquet';ITEMS=ROOT/'data/raw/items.csv';TERMS=ROOT/'data/raw/terms.csv';OUT=ROOT/'data/processed/v59_bge_expanded_pool_scores.parquet';MODEL='BAAI/bge-reranker-v2-m3';BATCH=24;MAXLEN=256
@torch.no_grad()
def main():
 p=pd.read_parquet(POOL);rt=p.root_add.eq(p.root_top1)|p.root_add.eq(p.root_top2);rs=p.same_root|((p.root_prob_add>=p.root_prob_drop+.04)&(p.root_prob_add>=.08)&rt);lex=(p.diff_v38_lex_score>0)&(p.diff_v33_ft_score>0)&(p.diff_v38_word_overlap_full>=0)&(p.diff_v38_must_token_coverage_full>=0)&(p.diff_v38_must_missing_ratio<=0)&(p.diff_v38_special_token_coverage_full>=0)&(p.diff_v43_query_pop_support>=0)&(p.robust_gain>=.12)
 base=rs&lex&(p.v55_prob>=.90)&(p.v38_title_missing_ratio_add<=.5)&(p.v38_word_overlap_title_add>=p.v38_word_overlap_title_drop)&(p.query_token_count>=2);p=p[base]
 ids=pd.concat([p[['term_id','id_add','item_id_add']].rename(columns={'id_add':'id','item_id_add':'item_id'}),p[['term_id','id_drop','item_id_drop']].rename(columns={'id_drop':'id','item_id_drop':'item_id'})]).drop_duplicates('id');terms=pd.read_csv(TERMS,usecols=['term_id','query']);terms.term_id=terms.term_id.astype(str);items=pd.read_csv(ITEMS,usecols=['item_id','title','category','brand','attributes']);items.item_id=items.item_id.astype(str);d=ids.merge(terms,on='term_id').merge(items,on='item_id');d['text']='Başlık: '+d.title.fillna('')+' Kategori: '+d.category.fillna('')+' Marka: '+d.brand.fillna('')+' Özellikler: '+d.attributes.fillna('').str.slice(0,500)
 dev='cuda' if torch.cuda.is_available() else 'cpu';print('device',dev,'rows',len(d));tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval();out=[];t=time.time()
 for i in range(0,len(d),BATCH):
  z=tok(d['query'].iloc[i:i+BATCH].fillna('').tolist(),d.text.iloc[i:i+BATCH].tolist(),padding=True,truncation=True,max_length=MAXLEN,return_tensors='pt').to(dev);out.extend(m(**z).logits.view(-1).float().cpu().numpy().tolist())
  if i%480==0:print(i,len(d),'min',round((time.time()-t)/60,2))
 d['v59_bge_raw']=np.asarray(out,dtype=np.float32);d[['id','term_id','item_id','v59_bge_raw']].to_parquet(OUT,index=False);print('saved',OUT,'min',(time.time()-t)/60)
if __name__=='__main__':main()
