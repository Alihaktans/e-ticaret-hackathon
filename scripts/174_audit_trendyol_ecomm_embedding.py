from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score

ROOT=Path('.');ITEMS=ROOT/'data/raw/items.csv';OUT=ROOT/'reports/experiments/v70_trendyol_embedding_source_audit.csv';INFO=ROOT/'reports/experiments/v70_trendyol_embedding_source_audit.json'
MODEL='Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0'
LABELS=[('v29',ROOT/'reports/manual_review/review_v29_qswap_targets_assistant_labeled.csv'),('v33',ROOT/'reports/manual_review/review_v33_final_qswap_targets_assistant_labeled.csv')]

def clean(x,n):
 if pd.isna(x):return ''
 return ' '.join(str(x).replace('\n',' ').replace('\r',' ').split())[:n]
def product_text(r,mode):
 title=clean(r.get('title',''),300)
 if mode=='title':return title
 base=f"Başlık: {title} | Kategori: {clean(r.get('category',''),220)} | Marka: {clean(r.get('brand',''),100)}"
 if mode=='title_category':return base
 return base+f" | Cinsiyet: {clean(r.get('gender',''),50)} | Yaş: {clean(r.get('age_group',''),50)} | Özellikler: {clean(r.get('attributes',''),450)}"

def truncate_normalize(x,dim):
 x=x[:,:dim].astype(np.float32,copy=False)
 return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def main():
 parts=[]
 for src,p in LABELS:
  d=pd.read_csv(p);lab=pd.to_numeric(d.assistant_swap_label,errors='coerce');recheck=pd.to_numeric(d.get('needs_recheck',0),errors='coerce').fillna(1);conf=d.get('assistant_confidence','').astype(str).str.lower();keep=lab.isin([0,1])&recheck.eq(0)&conf.isin(['high','medium']);d=d.loc[keep].copy();d['label']=lab.loc[keep].astype(np.int8);d['source']=src;parts.append(d[['source','term_id','query','item_id_add','item_id_drop','label']])
 labels=pd.concat(parts,ignore_index=True);needed=set(labels.item_id_add.astype(str))|set(labels.item_id_drop.astype(str));items=pd.read_csv(ITEMS,low_memory=False);items.item_id=items.item_id.astype(str);items=items[items.item_id.isin(needed)].drop_duplicates('item_id');model=SentenceTransformer(MODEL,trust_remote_code=True,truncate_dim=768,device='cuda')
 q_raw=labels['query'].fillna('').astype(str).tolist();q_pref=['query: '+q for q in q_raw];qemb={'raw':model.encode(q_raw,batch_size=128,normalize_embeddings=False,convert_to_numpy=True,show_progress_bar=True),'query_prefix':model.encode(q_pref,batch_size=128,normalize_embeddings=False,convert_to_numpy=True,show_progress_bar=True)}
 rows=[]
 for mode in ['title','title_category','full']:
  text=[product_text(r,mode) for _,r in items.iterrows()];emb=model.encode(text,batch_size=96,normalize_embeddings=False,convert_to_numpy=True,show_progress_bar=True);mp={iid:emb[i] for i,iid in enumerate(items.item_id)};ea0=np.stack([mp[str(x)] for x in labels.item_id_add]);ed0=np.stack([mp[str(x)] for x in labels.item_id_drop])
  for qmode,qe in qemb.items():
   for dim in [128,512,768]:
    q=truncate_normalize(qe,dim);ea=truncate_normalize(ea0,dim);ed=truncate_normalize(ed0,dim);margin=(q*ea).sum(1)-(q*ed).sum(1)
    for src in ['v29','v33','all']:
     mask=np.ones(len(labels),dtype=bool) if src=='all' else labels.source.eq(src).to_numpy();y=labels.label.to_numpy()[mask];m=margin[mask];pred=(m>0).astype(np.int8);rows.append({'text_mode':mode,'query_mode':qmode,'dim':dim,'source':src,'n':int(mask.sum()),'auc':float(roc_auc_score(y,m)),'ap':float(average_precision_score(y,m)),'macro_f1_zero':float(f1_score(y,pred,average='macro')),'accuracy_zero':float((pred==y).mean()),'margin_mean':float(m.mean()),'margin_std':float(m.std())})
 result=pd.DataFrame(rows).sort_values(['source','auc'],ascending=[True,False]);OUT.parent.mkdir(parents=True,exist_ok=True);result.to_csv(OUT,index=False);INFO.write_text(json.dumps({'model':MODEL,'rows':len(labels),'results':rows,'warning':'Assistant preference labels are not hidden Kaggle truth; source-wise stability is required.'},ensure_ascii=False,indent=2),encoding='utf8');print(result.to_string(index=False));print('saved',OUT,INFO)
if __name__=='__main__':main()
