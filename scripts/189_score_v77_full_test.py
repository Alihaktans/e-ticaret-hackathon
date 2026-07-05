"""Encode all test candidates with the validated v77 hard-negative model."""
from pathlib import Path
import argparse
import os, json, time
os.environ.setdefault("HF_MODULES_CACHE", str(Path(".cache/hf_modules").resolve()))
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT=Path('.');RAW=ROOT/'data/raw';IDX=ROOT/'data/processed/v70_trendyol_embedding';OUT=ROOT/'data/processed/v77_trendyol_hardnegative';MODEL=ROOT/'models/v77_trendyol_hardnegative/final';DIM=768
def clean(s,n): return s.fillna('').astype(str).str.replace(r'[\r\n]+',' ',regex=True).str.replace(r'\s+',' ',regex=True).str.strip().str.slice(0,n)
def product_text(d): return ('Başlık: '+clean(d.title,300)+' | Kategori: '+clean(d.category,220)+' | Marka: '+clean(d.brand,100)).tolist()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default=str(MODEL));ap.add_argument('--out',default=str(OUT));ap.add_argument('--holdout-delta-map',type=float,default=.04890458237463885);args=ap.parse_args();model_path=Path(args.model);out_dir=Path(args.out)
 out_dir.mkdir(parents=True,exist_ok=True);items=np.load(IDX/'item_ids.npy');terms=np.load(IDX/'term_ids.npy');n=json.loads((IDX/'index.json').read_text())['pairs']
 model=SentenceTransformer(str(model_path),trust_remote_code=True,local_files_only=True,device='cuda');model.max_seq_length=192
 qpath=out_dir/'query_embeddings.f16.dat';ipath=out_dir/'item_embeddings.f16.dat';donepath=out_dir/'item_done.npy'
 if not qpath.exists():
  t=pd.read_csv(RAW/'terms.csv',usecols=['term_id','query']);t.term_id=t.term_id.astype(str);txt=t.drop_duplicates('term_id').set_index('term_id').reindex(terms)['query'].fillna('').astype(str).tolist();e=model.encode(txt,batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True);m=np.memmap(qpath,dtype=np.float16,mode='w+',shape=(len(terms),DIM));m[:]=e;m.flush()
 mm=np.memmap(ipath,dtype=np.float16,mode='r+' if ipath.exists() else 'w+',shape=(len(items),DIM));done=np.load(donepath) if donepath.exists() else np.zeros(len(items),bool);index=pd.Index(items);seen=0;start=time.time()
 for chunk in pd.read_csv(RAW/'items.csv',usecols=['item_id','title','category','brand'],chunksize=10000,low_memory=False):
  pos=index.get_indexer(chunk.item_id.astype(str));keep=pos>=0
  if not keep.any():continue
  pos=pos[keep];frame=chunk.loc[keep];todo=~done[pos]
  if todo.any():
   pp=pos[todo];ff=frame.loc[todo];e=model.encode(product_text(ff),batch_size=192,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False);mm[pp]=e.astype(np.float16);done[pp]=True
  seen+=int(keep.sum())
  if seen%100000<10000: mm.flush();np.save(donepath,done);print('items',int(done.sum()),'/',len(done),'elapsed_min',round((time.time()-start)/60,1),flush=True)
 mm.flush();np.save(donepath,done)
 if not done.all():raise RuntimeError(f'missing items {(~done).sum()}')
 scores=np.lib.format.open_memmap(out_dir/'pair_scores.npy',dtype=np.float32,mode='w+',shape=(n,));ie=np.memmap(ipath,dtype=np.float16,mode='r',shape=(len(items),DIM));qe=np.memmap(qpath,dtype=np.float16,mode='r',shape=(len(terms),DIM));ii=pd.Index(items);ti=pd.Index(terms);off=0
 for chunk in pd.read_csv(RAW/'submission_pairs.csv',usecols=['term_id','item_id'],chunksize=25000):
  a=np.asarray(qe[ti.get_indexer(chunk.term_id.astype(str))],np.float32);b=np.asarray(ie[ii.get_indexer(chunk.item_id.astype(str))],np.float32);scores[off:off+len(chunk)]=np.einsum('ij,ij->i',a,b,optimize=True);off+=len(chunk)
  if off%250000<25000:scores.flush();print('pairs',off,'/',n,flush=True)
 scores.flush();(out_dir/'metadata.json').write_text(json.dumps({'model':str(model_path),'base_revision':'00c030c9','pairs':n,'items':len(items),'terms':len(terms),'holdout_delta_query_map':args.holdout_delta_map,'product_format':'Başlık | Kategori | Marka'},indent=2),encoding='utf8');print('done',out_dir)
if __name__=='__main__':main()
