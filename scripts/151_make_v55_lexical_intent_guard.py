from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
SAMPLE = ROOT / "data/raw/sample_submission.csv"
LABELS = ROOT / "data/processed/v53_swap_preference_features.parquet"
V54_POOL = ROOT / "data/processed/v54_source_robust_candidate_pairs.parquet"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
ANCHORS = {
    "raw_v35": ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    "v40_history2200": ROOT / "submissions/final_candidates_v40/A_v40_raw35_history_impact2200.csv",
}

FEATURES = [
    "diff_v38_lex_score", "diff_v47_reverse_base", "diff_v47_forward_base",
    "diff_v38_word_overlap_full", "diff_v38_must_missing_ratio",
    "diff_v38_must_token_coverage_full", "diff_v47_mutual_score",
    "diff_v33_ft_score", "diff_v38_special_token_coverage_full",
    "diff_v38_char_ngram_sim", "diff_v43_query_pop_support",
]

SCORE_SPECS = {
    ROOT / "data/processed/v38_lexical_pair_scores.parquet": [
        "v38_word_overlap_full", "v38_must_missing_ratio", "v38_must_token_coverage_full",
        "v38_special_token_coverage_full", "v38_char_ngram_sim",
    ],
    ROOT / "data/processed/v47_reciprocal_shadow_scores.parquet": ["v47_reverse_base", "v47_forward_base"],
    ROOT / "data/processed/v43_pop_brand_category_pair_scores.parquet": ["v43_query_pop_support"],
}

OUT_POOL = ROOT / "data/processed/v55_lexical_intent_guard_pairs.parquet"
OUT_REVIEW = ROOT / "reports/manual_review/review_v55_lexical_intent_guard_swaps.csv"
OUT_SUMMARY = ROOT / "reports/manual_review/v55_lexical_intent_guard_summary.csv"
OUT_AUDIT = ROOT / "reports/manual_review/v55_lexical_intent_guard_model_audit.json"
SUB_DIR = ROOT / "submissions/final_candidates_v55"


def fit_model(d: pd.DataFrame):
    x = d[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = d.label.astype(np.int8).to_numpy()
    xa = pd.concat([x, -x], ignore_index=True)
    ya = np.r_[y, 1-y]
    source = np.r_[d.source.to_numpy(), d.source.to_numpy()]
    counts = pd.Series(source).value_counts()
    weights = np.array([1/counts[s] for s in source])
    weights *= len(weights)/weights.sum()
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.08, max_iter=2000, class_weight="balanced", random_state=42))
    m.fit(xa, ya, logisticregression__sample_weight=weights)
    return m


def audit(d: pd.DataFrame):
    out=[]
    for tr,te in [("v29","v33"),("v33","v29")]:
        a=d[d.source.eq(tr)].copy(); b=d[d.source.eq(te)].copy()
        ids=set(b.id_add.astype(str))|set(b.id_drop.astype(str))
        a=a[~a.id_add.astype(str).isin(ids)&~a.id_drop.astype(str).isin(ids)]
        m=fit_model(a); p=m.predict_proba(b[FEATURES])[:,1]
        row={"train":tr,"test":te,"auc":float(roc_auc_score(b.label,p)),"ap":float(average_precision_score(b.label,p))}
        for t in [.9,.95]:
            z=p>=t; row[f"n_{t}"]=int(z.sum()); row[f"precision_{t}"]=float(b.loc[z,"label"].mean()) if z.any() else None
        out.append(row)
    return out


def attach_pair_features(pool: pd.DataFrame) -> pd.DataFrame:
    ids=set(pool.id_add.astype(str))|set(pool.id_drop.astype(str))
    row=pd.DataFrame({"id":sorted(ids)})
    for path,cols in SCORE_SPECS.items():
        d=pd.read_parquet(path,columns=["id"]+cols)
        d["id"]=d.id.astype(str)
        d=d[d.id.isin(ids)].drop_duplicates("id")
        row=row.merge(d,on="id",how="left",validate="one_to_one")
        print(path.name,"matched",len(d))
    add=row.rename(columns={"id":"id_add",**{c:c+"_add" for c in row if c!="id"}})
    drop=row.rename(columns={"id":"id_drop",**{c:c+"_drop" for c in row if c!="id"}})
    pool=pool.merge(add,on="id_add",how="left").merge(drop,on="id_drop",how="left")
    for cols in SCORE_SPECS.values():
        for c in cols: pool["diff_"+c]=pd.to_numeric(pool[c+"_add"],errors="coerce")-pd.to_numeric(pool[c+"_drop"],errors="coerce")
    return pool


def enrich(d: pd.DataFrame) -> pd.DataFrame:
    terms=pd.read_csv(TERMS)
    if "query" not in terms: terms=terms.rename(columns={[c for c in terms if c!="term_id"][0]:"query"})
    terms.term_id=terms.term_id.astype(str)
    d=d.merge(terms[["term_id","query"]],on="term_id",how="left")
    items=pd.read_csv(ITEMS)
    item_col=next((c for c in items if c.lower() in ["item_id","product_id"]),items.columns[0]); items[item_col]=items[item_col].astype(str)
    title=next((c for c in items if c.lower() in ["title","name","product_name"]),None)
    brand=next((c for c in items if "brand" in c.lower() or "marka" in c.lower()),None)
    cat=next((c for c in items if "category" in c.lower() or "kategori" in c.lower()),None)
    cols=[item_col]+[c for c in [title,brand,cat] if c]
    meta=items[cols].drop_duplicates(item_col)
    for side in ["add","drop"]:
        ren={item_col:"item_id_"+side}
        if title:ren[title]="title_"+side
        if brand:ren[brand]="brand_"+side
        if cat:ren[cat]="category_"+side
        d=d.merge(meta.rename(columns=ren),on="item_id_"+side,how="left")
    front=["anchor_name","mode","query","id_add","title_add","category_add","id_drop","title_drop","category_drop","v55_prob","robust_gain"]
    return d[[c for c in front if c in d]+[c for c in d if c not in front]]


def main():
    OUT_REVIEW.parent.mkdir(parents=True,exist_ok=True); OUT_POOL.parent.mkdir(parents=True,exist_ok=True); SUB_DIR.mkdir(parents=True,exist_ok=True)
    labels=pd.read_parquet(LABELS)
    report=audit(labels); print(json.dumps(report,indent=2))
    model=fit_model(labels)
    pool=pd.read_parquet(V54_POOL)
    pool=attach_pair_features(pool)
    pool["v55_prob"]=model.predict_proba(pool[FEATURES])[:,1].astype(np.float32)
    pool.to_parquet(OUT_POOL,index=False)
    sample=pd.read_csv(SAMPLE,usecols=["id"]); sample.id=sample.id.astype(str); idx=pd.Series(np.arange(len(sample)),index=sample.id)
    summaries=[]; reviews=[]
    for anchor_name,path in ANCHORS.items():
        a=pd.read_csv(path); a.id=a.id.astype(str)
        if not a.id.reset_index(drop=True).equals(sample.id.reset_index(drop=True)): raise RuntimeError("anchor mismatch")
        anchor=a.prediction.astype(np.int8).to_numpy()
        base=pool[pool.anchor_name.eq(anchor_name)].copy()
        common=(base.diff_v38_lex_score>0)&(base.diff_v33_ft_score>0)&(base.diff_v38_word_overlap_full>=0)&(base.diff_v38_must_token_coverage_full>=0)&(base.diff_v38_must_missing_ratio<=0)&(base.diff_v38_special_token_coverage_full>=0)&(base.diff_v43_query_pop_support>=0)&(base.robust_gain>=.15)
        modes={
            "ultra":common&(base.v55_prob>=.95)&(base.diff_v47_forward_base>0)&(base.diff_v47_reverse_base>0),
            "safe":common&(base.v55_prob>=.92)&((base.diff_v47_forward_base>0)|(base.diff_v47_reverse_base>0)),
            "balanced":common&(base.v55_prob>=.88),
        }
        for mode,mask in modes.items():
            swaps=base[mask].sort_values(["v55_prob","robust_gain"],ascending=False).drop_duplicates("term_id")
            print(anchor_name,mode,len(swaps))
            q=swaps.head(120).copy();q["mode"]=mode;reviews.append(q)
            for cap in [100,250,500,1000]:
                take=swaps.head(cap)
                if take.empty:continue
                pred=anchor.copy();pred[idx.loc[take.id_add].to_numpy()]=1;pred[idx.loc[take.id_drop].to_numpy()]=0
                variant=f"v55_{anchor_name}_{mode}_cap{cap}"; out=SUB_DIR/(variant+".csv")
                pd.DataFrame({"id":sample.id,"prediction":pred}).to_csv(out,index=False)
                summaries.append({"variant":variant,"anchor":anchor_name,"mode":mode,"accepted_pool":len(swaps),"used_swaps":len(take),"v55_prob_mean":float(take.v55_prob.mean()),"v55_prob_min":float(take.v55_prob.min()),"robust_gain_mean":float(take.robust_gain.mean()),"diff_vs_anchor":int((pred!=anchor).sum()),"ones":int(pred.sum()),"pos_ratio":float(pred.mean()),"file":str(out)})
    review=pd.concat(reviews,ignore_index=True).drop_duplicates(["anchor_name","id_add","id_drop"])
    enrich(review).to_csv(OUT_REVIEW,index=False)
    pd.DataFrame(summaries).to_csv(OUT_SUMMARY,index=False)
    OUT_AUDIT.write_text(json.dumps({"features":FEATURES,"holdout":report,"warning":"High offline pair precision is not a leaderboard estimate; manual semantic audit remains mandatory."},ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False));print("outputs",OUT_REVIEW,OUT_SUMMARY,OUT_AUDIT)


if __name__=="__main__":main()
