"""Replace v95-vetoed rows only when two intent models prefer a same-query alternative."""
from pathlib import Path
import gc,hashlib,json
import numpy as np,pandas as pd,torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
ROOT=Path('.');V89=ROOT/'submissions/final_candidates_v89/FINAL_CANDIDATE_v89_v71_dual_veto_below_m2.csv';V95=ROOT/'submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv';PAIR=ROOT/'data/raw/submission_pairs.csv';TERMS=ROOT/'data/raw/terms.csv';ITEMS=ROOT/'data/raw/items.csv';RANK=ROOT/'data/processed/v72_anchorless_trendlex_features.parquet';LEX=ROOT/'data/processed/v83_independent_test_scores.parquet';CACHE=ROOT/'data/processed/v97_targeted_replacement_scores.parquet';OUT=ROOT/'submissions/final_candidates_v97';REPORT=ROOT/'reports/experiments/v97_targeted_replacements.json';MODELS={'v88':ROOT/'models/v88_intent_berturk_from_v62v29/best','v90':ROOT/'models/v90_intent_berturk_from_v62v33/best'}
def repair(x):
 x='' if pd.isna(x) else str(x)
 try:return x.encode('latin1').decode('utf8')
 except:return x
def item_text(r):
 return ' | '.join(f'{l}: {repair(r.get(c,""))[:n]}' for c,l,n in [('title','başlık',260),('category','kategori',180),('brand','marka',80),('gender','cinsiyet',50),('age_group','yaş',50),('attributes','özellik',350)] if pd.notna(r.get(c,'')) and str(r.get(c,'')) not in ['', 'nan'])
@torch.inference_mode()
def score(d,path):
 dev='cuda';tok=AutoTokenizer.from_pretrained(path);m=AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval().half();z=np.empty(len(d),np.float32)
 for i in range(0,len(d),64):
  j=min(len(d),i+64);e=tok(d.query_text.iloc[i:j].tolist(),d.item_text.iloc[i:j].tolist(),padding=True,truncation=True,max_length=176,return_tensors='pt').to(dev)
  with torch.autocast(device_type='cuda',dtype=torch.float16):z[i:j]=m(**e).logits.view(-1).float().cpu().numpy()
  if j%5000<64:print(path,j,len(d),flush=True)
 del m,tok;gc.collect();torch.cuda.empty_cache();return z
def main():
 a=pd.read_csv(V89);b=pd.read_csv(V95);p=pd.read_csv(PAIR);r=pd.read_parquet(RANK,columns=['id','term_id','trendyol_qpct','lex_qpct']);x=pd.read_parquet(LEX);a.id=a.id.astype(str);b.id=b.id.astype(str);p.id=p.id.astype(str);r.id=r.id.astype(str);x.id=x.id.astype(str)
 if not a.id.equals(b.id) or not a.id.equals(p.id) or not a.id.equals(r.id) or not a.id.equals(x.id):raise RuntimeError('alignment')
 y89=a.prediction.to_numpy(np.int8);y95=b.prediction.to_numpy(np.int8);removed=(y89==1)&(y95==0);comp=.50*x.v82_score.to_numpy()+.25*r.trendyol_qpct.to_numpy()+.25*r.lex_qpct.to_numpy();pairs=[]
 for term,idx in p.groupby('term_id',sort=False).indices.items():
  idx=np.asarray(idx);drop=idx[removed[idx]];add=idx[y89[idx]==0]
  if not len(drop) or not len(add):continue
  drop=drop[np.argsort(comp[drop])];add=add[np.argsort(-comp[add])]
  for ai,di in zip(add[:len(drop)],drop):pairs.append((term,int(ai),int(di)))
 pairs=pd.DataFrame(pairs,columns=['term_id','add_row','drop_row']);need=np.unique(np.r_[pairs.add_row,pairs.drop_row]);d=p.iloc[need][['id','term_id','item_id']].copy();terms=pd.read_csv(TERMS,usecols=['term_id','query']);items=pd.read_csv(ITEMS,low_memory=False);items.item_id=items.item_id.astype(str);items['item_text']=items.apply(item_text,axis=1);d=d.merge(terms,on='term_id').merge(items[['item_id','item_text']],on='item_id');d['query_text']='sorgu: '+d['query'].map(repair).str.slice(0,180);d['row']=d.id.map(pd.Series(np.arange(len(p)),index=p.id));d=d.sort_values('row').reset_index(drop=True)
 if CACHE.exists():
  old=pd.read_parquet(CACHE);d=d.merge(old[['id']+[c for c in old if c.endswith('_score')]],on='id',how='left')
 for name,path in MODELS.items():
  col=name+'_score'
  if col not in d or d[col].isna().any():d[col]=score(d,path)
 d.to_parquet(CACHE,index=False);mp=d.set_index('row');rows=[]
 for name in MODELS:pairs['margin_'+name]=mp.loc[pairs.add_row,name+'_score'].to_numpy()-mp.loc[pairs.drop_row,name+'_score'].to_numpy()
 OUT.mkdir(parents=True,exist_ok=True);outs=[]
 for th in [0,.5,1,2,3]:
  ok=(pairs.margin_v88>th)&(pairs.margin_v90>th);pred=y95.copy();pred[pairs.loc[ok,'add_row'].to_numpy()]=1;path=OUT/f'FINAL_CANDIDATE_v97_dual_replacement_margin{str(th).replace(".","p")}.csv';pd.DataFrame({'id':a.id,'prediction':pred}).to_csv(path,index=False);outs.append({'threshold':th,'accepted':int(ok.sum()),'remaining_removed':int(removed.sum()-ok.sum()),'positives':int(pred.sum()),'ratio':float(pred.mean()),'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 rep={'initial_removed':int(removed.sum()),'proposed_pairs':len(pairs),'margin_correlation':float(pairs[['margin_v88','margin_v90']].corr().iloc[0,1]),'candidates':outs};REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(rep,indent=2),encoding='utf8');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
