from pathlib import Path
import json,re,unicodedata
import numpy as np
import pandas as pd
ROOT=Path('.');V34=ROOT/'data/processed/v34/v34_hard_negative_pairs.parquet';V61=ROOT/'data/processed/v61_berturk_pairwise_triplets.parquet';ITEMS=ROOT/'data/raw/items.csv';OUT=ROOT/'data/processed/v62_berturk_pairwise_triplets.parquet';REPORT=ROOT/'reports/experiments/v62_berturk_pairwise_dataset_report.json'
TR=str.maketrans({'ı':'i','İ':'i','ş':'s','Ş':'s','ğ':'g','Ğ':'g','ü':'u','Ü':'u','ö':'o','Ö':'o','ç':'c','Ç':'c'});STOP={'ve','ile','icin','bir','bu','set','takim','adet','model','urun','renk'}
def norm(x):
 x=str(x or '')
 try:x=x.encode('latin1').decode('utf8')
 except (UnicodeError):pass
 x=unicodedata.normalize('NFKD',x.translate(TR).lower());return re.sub(r'\s+',' ',re.sub('[^a-z0-9]+',' ',x)).strip()
def toks(x):return {z for z in norm(x).split() if len(z)>1 and z not in STOP}
def ov(q,t):a=toks(q);return len(a&toks(t))/max(1,len(a))
def item_text(r):
 parts=[]
 for c,l,n in [('title','başlık',260),('category','kategori',180),('brand','marka',80),('gender','cinsiyet',50),('age_group','yaş',50),('attributes','özellik',350)]:
  v=str(r.get(c,'') or '').replace('\n',' ')[:n].strip()
  if v and v.lower()!='nan':parts.append(f'{l}: {v}')
 return ' | '.join(parts)
def main():
 cols=['term_id','item_id','label','v33_retrieval_score','query','title','category'];d=pd.read_parquet(V34,columns=cols);d.term_id=d.term_id.astype(str);d.item_id=d.item_id.astype(str);d['root']=d.category.fillna('unknown').astype(str).str.split('/').str[0].map(norm);d['leaf']=d.category.fillna('unknown').astype(str).map(norm);d['overlap']=[ov(q,t) for q,t in zip(d['query'],d.title)]
 pos=d[d.label.eq(1)].copy();neg=d[d.label.eq(0)].copy();roots=pos.groupby('term_id').root.apply(set).to_dict();leaves=pos.groupby('term_id').leaf.apply(set).to_dict();best=pos.sort_values(['term_id','overlap'],ascending=[True,False]).drop_duplicates('term_id')[['term_id','item_id','overlap']].rename(columns={'item_id':'item_id_pos','overlap':'pos_title_overlap'})
 neg['same_root']=[r in roots.get(t,set()) for t,r in zip(neg.term_id,neg.root)];neg['new_leaf']=[c not in leaves.get(t,set()) for t,c in zip(neg.term_id,neg.leaf)];neg=pd.merge(neg,best,on='term_id',how='inner');neg=neg[neg.same_root&neg.new_leaf&(neg.pos_title_overlap>=neg.overlap+.15)].copy();neg['v33_retrieval_score']=pd.to_numeric(neg.v33_retrieval_score,errors='coerce').fillna(-99);neg=neg.sort_values(['term_id','v33_retrieval_score'],ascending=[True,False]).groupby('term_id',sort=False).head(6)
 x=neg.rename(columns={'item_id':'item_id_neg','overlap':'neg_title_overlap'});x['source_type']='same_root_leaf_mismatch';x['confidence_weight']=np.float32(.7);x['query_text']='sorgu: '+x['query'].fillna('').astype(str).str.slice(0,180);x['fold']=x.term_id.map(lambda z:int(__import__('hashlib').md5(z.encode()).hexdigest()[:8],16)%10).astype(np.int8);x['is_manual']=np.int8(0)
 needed=set(x.item_id_pos)|set(x.item_id_neg);items=pd.read_csv(ITEMS,low_memory=False);items.item_id=items.item_id.astype(str);it=items[items.item_id.isin(needed)].drop_duplicates('item_id').copy();it['txt']=it.apply(item_text,axis=1);mp=it.set_index('item_id').txt;x['pos_text']=x.item_id_pos.map(mp);x['neg_text']=x.item_id_neg.map(mp)
 keep=['term_id','query','item_id_pos','item_id_neg','source_type','confidence_weight','pos_title_overlap','neg_title_overlap','pos_text','neg_text','query_text','fold','is_manual'];x=x[keep].dropna(subset=['pos_text','neg_text']);base=pd.read_parquet(V61);out=pd.concat([base,x],ignore_index=True).drop_duplicates(['term_id','item_id_pos','item_id_neg']);out.to_parquet(OUT,index=False);rep={'rows':len(out),'added_same_root':len(x),'unique_terms':out.term_id.nunique(),'source_counts':out.source_type.value_counts().to_dict(),'warning':'Same-root leaf mismatch rows are weak preferences with weight 0.7; source-holdout validation decides whether they help.'};REPORT.write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(rep,ensure_ascii=False,indent=2));print('saved',OUT)
if __name__=='__main__':main()
