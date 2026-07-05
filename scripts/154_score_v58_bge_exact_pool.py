from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

ROOT=Path('.');POOL=ROOT/'data/processed/v57_exact_title_guard_pairs.parquet';ITEMS=ROOT/'data/raw/items.csv';TERMS=ROOT/'data/raw/terms.csv';OUT=ROOT/'data/processed/v58_bge_exact_pool_scores.parquet'
MODEL='BAAI/bge-reranker-v2-m3';BATCH=16;MAXLEN=256

@torch.no_grad()
def main():
 pool=pd.read_parquet(POOL);base=(pool.diff_v38_lex_score>0)&(pool.diff_v33_ft_score>0)&(pool.v55_prob>=.90)&(pool.robust_gain>=.15)&(pool.query_token_count>=2)&((pool.v38_title_missing_ratio_add<=1e-6)|(pool.v38_exact_phrase_title_add>=1))
 p=pool[base].copy();ids=pd.concat([p[['term_id','id_add','item_id_add']].rename(columns={'id_add':'id','item_id_add':'item_id'}),p[['term_id','id_drop','item_id_drop']].rename(columns={'id_drop':'id','item_id_drop':'item_id'})]).drop_duplicates('id')
 terms=pd.read_csv(TERMS,usecols=['term_id','query']);terms.term_id=terms.term_id.astype(str);items=pd.read_csv(ITEMS,usecols=['item_id','title','category','brand','attributes']);items.item_id=items.item_id.astype(str)
 d=ids.merge(terms,on='term_id',how='left').merge(items,on='item_id',how='left');d['product_text']='Başlık: '+d.title.fillna('')+' Kategori: '+d.category.fillna('')+' Marka: '+d.brand.fillna('')+' Özellikler: '+d.attributes.fillna('').str.slice(0,500)
 device='cuda' if torch.cuda.is_available() else 'cpu';print('device',device,'rows',len(d));tok=AutoTokenizer.from_pretrained(MODEL);model=AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval();scores=[];t=time.time()
 for i in range(0,len(d),BATCH):
  x=tok(d['query'].iloc[i:i+BATCH].fillna('').tolist(),d.product_text.iloc[i:i+BATCH].tolist(),padding=True,truncation=True,max_length=MAXLEN,return_tensors='pt').to(device);scores.extend(model(**x).logits.view(-1).float().cpu().numpy().tolist())
  if i%160==0:print(i,len(d),'elapsed',round((time.time()-t)/60,2))
 d['v58_bge_raw']=np.asarray(scores,dtype=np.float32);d[['id','term_id','item_id','v58_bge_raw']].to_parquet(OUT,index=False);print('saved',OUT,'elapsed_min',(time.time()-t)/60)
if __name__=='__main__':main()
