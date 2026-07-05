from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer

ROOT=Path('.');POOL=ROOT/'data/processed/v59_expanded_bge_pairs.parquet';ITEMS=ROOT/'data/raw/items.csv';TERMS=ROOT/'data/raw/terms.csv';OUT=ROOT/'data/processed/v61_pairwise_berturk_item_scores.parquet'
MODELS={'v29train':'models/v61_berturk_pairwise_audit/best','v33train':'models/v61_berturk_pairwise_holdout_v29/best'}
def item_text(r):
 parts=[]
 for c,label,lim in [('title','başlık',260),('category','kategori',180),('brand','marka',80),('gender','cinsiyet',50),('age_group','yaş',50),('attributes','özellik',350)]:
  v=str(r.get(c,'') or '').replace('\n',' ')[:lim].strip()
  if v and v.lower()!='nan':parts.append(f'{label}: {v}')
 return ' | '.join(parts)
@torch.no_grad()
def score(model_path,d,device):
 tok=AutoTokenizer.from_pretrained(model_path);m=AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval();out=[]
 for i in range(0,len(d),48):
  e=tok(d.query_text.iloc[i:i+48].tolist(),d.item_text.iloc[i:i+48].tolist(),padding=True,truncation=True,max_length=160,return_tensors='pt').to(device)
  with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):z=m(**e).logits.view(-1).float()
  out.extend(z.cpu().numpy().tolist())
  if i%2400==0:print(model_path,i,len(d),flush=True)
 return np.asarray(out,dtype=np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-v29train',default=MODELS['v29train']);ap.add_argument('--model-v33train',default=MODELS['v33train']);ap.add_argument('--output',default=str(OUT));args=ap.parse_args();models={'v29train':args.model_v29train,'v33train':args.model_v33train};output=Path(args.output)
 p=pd.read_parquet(POOL);p=p[p.bge_add.notna()&p.bge_drop.notna()];ids=pd.concat([p[['term_id','id_add','item_id_add']].rename(columns={'id_add':'id','item_id_add':'item_id'}),p[['term_id','id_drop','item_id_drop']].rename(columns={'id_drop':'id','item_id_drop':'item_id'})]).drop_duplicates('id')
 terms=pd.read_csv(TERMS,usecols=['term_id','query']);terms.term_id=terms.term_id.astype(str);items=pd.read_csv(ITEMS);items.item_id=items.item_id.astype(str);d=ids.merge(terms,on='term_id').merge(items,on='item_id');d['query_text']='sorgu: '+d['query'].fillna('').astype(str).str.slice(0,180);d['item_text']=d.apply(item_text,axis=1);dev='cuda' if torch.cuda.is_available() else 'cpu';print('rows',len(d),'device',dev)
 for name,path in models.items():d['v61_'+name+'_score']=score(path,d,dev)
 d[['id','term_id','item_id','v61_v29train_score','v61_v33train_score']].to_parquet(output,index=False);print('saved',output)
if __name__=='__main__':main()
