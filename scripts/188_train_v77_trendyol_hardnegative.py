"""Hard-negative contrastive adaptation of the Trendyol encoder."""
from pathlib import Path
import argparse, hashlib, json, os, random

os.environ.setdefault("HF_MODULES_CACHE", str(Path(".cache/hf_modules").resolve()))
import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset

ROOT=Path('.')
SNAP=Path.home()/'.cache/huggingface/hub/models--Trendyol--TY-ecomm-embed-multilingual-base-v1.2.0/snapshots/00c030c9a56bff9403f95c1b45f4b82e669e243c'
DATA=ROOT/'data/processed/v34/v34_hard_negative_pairs.parquet'

def clean(x,n):
 if pd.isna(x): return ''
 return ' '.join(str(x).replace('\n',' ').replace('\r',' ').split())[:n]
def item_text(r): return f'Başlık: {clean(r.title,300)} | Kategori: {clean(r.category,220)} | Marka: {clean(r.brand,100)}'
def fold(t): return int(hashlib.md5(str(t).encode()).hexdigest()[:8],16)%10

class Triples(Dataset):
 def __init__(self,d): self.q=d.q.tolist();self.p=d.p.tolist();self.n=d.n.tolist()
 def __len__(self): return len(self.q)
 def __getitem__(self,i): return InputExample(texts=[self.q[i],self.p[i],self.n[i]])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='models/v77_trendyol_hardnegative');ap.add_argument('--base',default=str(SNAP));ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--lr',type=float,default=5e-6);ap.add_argument('--temperature',type=float,default=.05);ap.add_argument('--seed',type=int,default=20260703);args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 cols=['term_id','label','v33_retrieval_score','query','title','category','brand']
 d=pd.read_parquet(DATA,columns=cols);d.term_id=d.term_id.astype(str);d['fold']=d.term_id.map(fold)
 tr=d[~d['fold'].eq(0)].copy()
 pos=tr[tr.label.eq(1)].sample(frac=1,random_state=args.seed).drop_duplicates('term_id')
 neg=tr[tr.label.eq(0)].sort_values(['term_id','v33_retrieval_score'],ascending=[True,False]).drop_duplicates('term_id')
 p=pos[['term_id','query','title','category','brand']].copy();p['p']=[item_text(r) for r in p.itertuples(index=False)]
 n=neg[['term_id','title','category','brand','v33_retrieval_score']].copy();n['n']=[item_text(r) for r in n.itertuples(index=False)]
 triples=p[['term_id','query','p']].merge(n[['term_id','n','v33_retrieval_score']],on='term_id',validate='one_to_one').rename(columns={'query':'q'})
 triples.q=triples.q.fillna('').astype(str)
 cfg=vars(args)|{'base_revision':'00c030c9','train_queries':len(triples),'holdout_fold':0,'negative':'highest-v33-retrieval non-positive per query','product_format':'Başlık | Kategori | Marka'}
 (out/'training_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(cfg,ensure_ascii=False,indent=2),flush=True)
 torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);random.seed(args.seed);np.random.seed(args.seed)
 model=SentenceTransformer(str(args.base),trust_remote_code=True,local_files_only=True,device='cuda');model.max_seq_length=192
 loader=DataLoader(Triples(triples),batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0)
 loss=losses.MultipleNegativesRankingLoss(model,scale=1/args.temperature);steps=len(loader)*args.epochs
 model.fit(train_objectives=[(loader,loss)],epochs=args.epochs,warmup_steps=max(10,int(.08*steps)),optimizer_params={'lr':args.lr},weight_decay=.01,use_amp=True,show_progress_bar=True,output_path=str(out/'final'),checkpoint_path=str(out/'checkpoints'),checkpoint_save_steps=500,checkpoint_save_total_limit=2)
 print('saved',out/'final')
if __name__=='__main__':main()
