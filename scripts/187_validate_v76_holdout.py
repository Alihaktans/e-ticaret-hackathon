"""Compare base and v76 on untouched, candidate-shaped train-query holdout."""
from pathlib import Path
import os
import json
import argparse

os.environ.setdefault("HF_MODULES_CACHE", str(Path(".cache/hf_modules").resolve()))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(".")
BASE = Path.home() / ".cache/huggingface/hub/models--Trendyol--TY-ecomm-embed-multilingual-base-v1.2.0/snapshots/00c030c9a56bff9403f95c1b45f4b82e669e243c"
FT = ROOT / "models/v76_trendyol_contrastive/final"
HOLDOUT = ROOT / "models/v76_trendyol_contrastive/holdout_terms.txt"
DATA = ROOT / "data/processed/v34/v34_hard_negative_pairs.parquet"
OUT = ROOT / "reports/experiments/v76_holdout_validation.json"
SCORES = ROOT / "data/processed/v76_holdout_scores.parquet"

def clean(x,n):
    if pd.isna(x): return ""
    return " ".join(str(x).replace("\n"," ").replace("\r"," ").split())[:n]

def item_text(r):
    return f"Başlık: {clean(r.title,300)} | Kategori: {clean(r.category,220)} | Marka: {clean(r.brand,100)}"

def score(model_path, queries, items):
    model=SentenceTransformer(str(model_path),trust_remote_code=True,local_files_only=True,device="cuda")
    model.max_seq_length=192
    q=model.encode(queries,batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
    p=model.encode(items,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
    return np.einsum("ij,ij->i",q,p).astype(np.float32)

def metrics(d,col):
    y=d.label.to_numpy(np.int8); s=d[col].to_numpy()
    rows=[]
    for _,g in d.groupby("term_id",sort=False):
        yy=g.label.to_numpy(np.int8); ss=g[col].to_numpy(); k=int(yy.sum())
        order=np.argsort(-ss); pred=np.zeros(len(g),dtype=np.int8); pred[order[:k]]=1
        rows.append({"ap":average_precision_score(yy,ss),"budget_accuracy":float((pred==yy).mean()),"recall_at_true_k":float(yy[order[:k]].sum()/max(1,k))})
    q=pd.DataFrame(rows)
    return {"pair_auc":float(roc_auc_score(y,s)),"pair_ap":float(average_precision_score(y,s)),
            "query_map":float(q.ap.mean()),"query_budget_accuracy":float(q.budget_accuracy.mean()),
            "query_recall_at_true_k":float(q.recall_at_true_k.mean()),"queries":len(q),"pairs":len(d)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ft',default=str(FT));ap.add_argument('--name',default='v76');args=ap.parse_args()
    ft_path=Path(args.ft)
    hold=set(HOLDOUT.read_text(encoding="utf8").splitlines())
    cols=["term_id","item_id","label","query","title","category","brand"]
    d=pd.read_parquet(DATA,columns=cols); d.term_id=d.term_id.astype(str); d=d[d.term_id.isin(hold)].copy().reset_index(drop=True)
    d["query_deploy"]=d["query"].fillna("").astype(str); d["item_deploy"]=[item_text(r) for r in d.itertuples(index=False)]
    d["base_score"]=score(BASE,d.query_deploy.tolist(),d.item_deploy.tolist())
    score_col=args.name+"_score";d[score_col]=score(ft_path,d.query_deploy.tolist(),d.item_deploy.tolist())
    scores_path=ROOT/f"data/processed/{args.name}_holdout_scores.parquet";out_path=ROOT/f"reports/experiments/{args.name}_holdout_validation.json"
    d[["term_id","item_id","label","base_score",score_col]].to_parquet(scores_path,index=False)
    report={"holdout_is_training_disjoint":True,"base":metrics(d,"base_score"),args.name:metrics(d,score_col)}
    report["delta"]={k:report[args.name][k]-report["base"][k] for k in ["pair_auc","pair_ap","query_map","query_budget_accuracy","query_recall_at_true_k"]}
    out_path.parent.mkdir(parents=True,exist_ok=True); out_path.write_text(json.dumps(report,indent=2),encoding="utf8"); print(json.dumps(report,indent=2))

if __name__=="__main__":main()
