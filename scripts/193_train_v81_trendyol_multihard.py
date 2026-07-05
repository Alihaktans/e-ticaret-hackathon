"""Train Trendyol encoder with three explicit retrieval negatives per query."""
from pathlib import Path
import argparse,hashlib,json,os,random
os.environ.setdefault('HF_MODULES_CACHE',str(Path('.cache/hf_modules').resolve()))
import numpy as np,pandas as pd,torch
from sentence_transformers import InputExample,SentenceTransformer,losses
from torch.utils.data import Dataset,DataLoader
ROOT=Path('.');SNAP=Path.home()/'.cache/huggingface/hub/models--Trendyol--TY-ecomm-embed-multilingual-base-v1.2.0/snapshots/00c030c9a56bff9403f95c1b45f4b82e669e243c';DATA=ROOT/'data/processed/v34/v34_hard_negative_pairs.parquet'
def clean(x,n):
 if pd.isna(x):return ''
 return ' '.join(str(x).replace('\n',' ').replace('\r',' ').split())[:n]
def text(r):return f'Başlık: {clean(r.title,300)} | Kategori: {clean(r.category,220)} | Marka: {clean(r.brand,100)}'
def fold(t):return int(hashlib.md5(str(t).encode()).hexdigest()[:8],16)%10
class D(Dataset):
 def __init__(self,x):self.rows=x[['q','p','n1','n2','n3']].values.tolist()
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return InputExample(texts=self.rows[i])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='models/v81_trendyol_multihard');ap.add_argument('--batch-size',type=int,default=6);ap.add_argument('--lr',type=float,default=5e-6);ap.add_argument('--seed',type=int,default=20260704);args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 cols=['term_id','label','v33_retrieval_score','query','title','category','brand'];d=pd.read_parquet(DATA,columns=cols);d.term_id=d.term_id.astype(str);d=d[d.term_id.map(fold).ne(0)]
 p=d[d.label.eq(1)].sample(frac=1,random_state=args.seed).drop_duplicates('term_id').copy();p['p']=[text(r) for r in p.itertuples(index=False)];n=d[d.label.eq(0)].sort_values(['term_id','v33_retrieval_score'],ascending=[True,False]).copy();n['rank']=n.groupby('term_id').cumcount()+1;n=n[n['rank'].le(3)];n['txt']=[text(r) for r in n.itertuples(index=False)];wide=n.pivot(index='term_id',columns='rank',values='txt').rename(columns={1:'n1',2:'n2',3:'n3'}).dropna();x=p[['term_id','query','p']].rename(columns={'query':'q'}).merge(wide,on='term_id');x.q=x.q.fillna('').astype(str)
 cfg=vars(args)|{'base_revision':'00c030c9','queries':len(x),'holdout_fold':0,'explicit_negatives_per_query':3,'product_format':'Başlık | Kategori | Marka'};(out/'training_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(cfg,ensure_ascii=False,indent=2),flush=True)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);m=SentenceTransformer(str(SNAP),trust_remote_code=True,local_files_only=True,device='cuda');m.max_seq_length=192;loader=DataLoader(D(x),batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0);loss=losses.MultipleNegativesRankingLoss(m,scale=20.);steps=len(loader);m.fit(train_objectives=[(loader,loss)],epochs=1,warmup_steps=int(.08*steps),optimizer_params={'lr':args.lr},weight_decay=.01,use_amp=True,show_progress_bar=True,output_path=str(out/'final'),checkpoint_path=str(out/'checkpoints'),checkpoint_save_steps=700,checkpoint_save_total_limit=1);print('saved',out/'final')
if __name__=='__main__':main()
