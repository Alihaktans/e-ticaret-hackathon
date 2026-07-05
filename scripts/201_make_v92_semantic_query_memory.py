"""Transfer true train positives to semantically near test queries, with honest train-query CV."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import faiss
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio as fuzz_ratio


ROOT = Path(".")
TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
SUB_PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
V34 = ROOT / "data/processed/v34/v34_hard_negative_pairs.parquet"
BASE = ROOT / "submissions/final_candidates_v89/FINAL_CANDIDATE_v89_v71_dual_veto_below_m2.csv"
OUT_DIR = ROOT / "submissions/final_candidates_v92"
REPORT = ROOT / "reports/experiments/v92_semantic_query_memory.json"
REVIEW = ROOT / "reports/manual_review/v92_semantic_query_memory_adds.csv"


def repair(x):
    x = "" if pd.isna(x) else str(x)
    try: x = x.encode("latin1").decode("utf8")
    except (UnicodeError, UnicodeEncodeError): pass
    return x


def norm(x):
    x = unicodedata.normalize("NFKD", repair(x).lower().replace("ı", "i"))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", x)).strip()


def load_embedding(prefix):
    ids = pd.read_parquet(ROOT / f"data/processed/{prefix}_all_term_ids.parquet").iloc[:, 0].astype(str).to_numpy()
    emb = np.load(ROOT / f"data/processed/{prefix}_all_terms_emb.npy", mmap_mode="r")
    return ids, emb


def normalized_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def neighbor_pairs(target_ids, train_ids, all_ids, e5, mini, k=8):
    pos = {x: i for i, x in enumerate(all_ids)}
    tr_idx = np.array([pos[x] for x in train_ids], dtype=np.int64)
    tg_idx = np.array([pos[x] for x in target_ids], dtype=np.int64)
    e5_train = normalized_rows(e5[tr_idx]); mini_train = normalized_rows(mini[tr_idx])
    e5_target = normalized_rows(e5[tg_idx]); mini_target = normalized_rows(mini[tg_idx])
    ie = faiss.IndexFlatIP(e5_train.shape[1]); ie.add(e5_train)
    im = faiss.IndexFlatIP(mini_train.shape[1]); im.add(mini_train)
    _, ne = ie.search(e5_target, k); _, nm = im.search(mini_target, k)
    rows = []
    for i, target in enumerate(target_ids):
        for j in set(ne[i].tolist() + nm[i].tolist()):
            source = train_ids[int(j)]
            if source == target: continue
            rows.append((target, source, i, int(j)))
    out = pd.DataFrame(rows, columns=["target_term_id", "source_term_id", "ti", "sj"])
    ti = out.ti.to_numpy(); sj = out.sj.to_numpy()
    out["e5_sim"] = np.einsum("ij,ij->i", e5_target[ti], e5_train[sj]).astype(np.float32)
    out["mini_sim"] = np.einsum("ij,ij->i", mini_target[ti], mini_train[sj]).astype(np.float32)
    return out.drop(columns=["ti", "sj"])


def enrich_query_relation(maps, qmap):
    out = maps.copy()
    tq = out.target_term_id.map(qmap).fillna("").map(norm)
    sq = out.source_term_id.map(qmap).fillna("").map(norm)
    out["target_query"] = tq; out["source_query"] = sq
    out["char_sim"] = np.array([fuzz_ratio(a, b) / 100 for a, b in zip(tq, sq)], dtype=np.float32)
    target_tokens = [set(x.split()) for x in tq]; source_tokens = [set(x.split()) for x in sq]
    out["target_subset_source"] = np.array([bool(a) and a.issubset(b) for a, b in zip(target_tokens, source_tokens)], dtype=np.int8)
    out["source_subset_target"] = np.array([bool(b) and b.issubset(a) for a, b in zip(target_tokens, source_tokens)], dtype=np.int8)
    out["relation_score"] = (0.42*out.e5_sim + 0.33*out.mini_sim + 0.25*out.char_sim +
                             0.06*out.target_subset_source - 0.03*out.source_subset_target).astype(np.float32)
    return out


def transfer_rows(maps, train_pos, candidates, target_id_col="target_term_id"):
    x = maps.merge(train_pos.rename(columns={"term_id": "source_term_id"}), on="source_term_id", how="inner")
    x = x.merge(candidates.rename(columns={"term_id": target_id_col}), on=[target_id_col, "item_id"], how="inner")
    return x


def config_masks(d):
    return {
        "lexical_safe": (d.char_sim >= .88) & (d.target_subset_source.eq(1)) & (d.e5_sim >= .82) & (d.mini_sim >= .72),
        "dual_strict": (d.e5_sim >= .92) & (d.mini_sim >= .84) & (d.char_sim >= .72) & (d.target_subset_source.eq(1)),
        "typo_strict": (d.char_sim >= .94) & (d.e5_sim >= .84) & (d.mini_sim >= .76),
        "triple_ultra": (d.e5_sim >= .94) & (d.mini_sim >= .88) & (d.char_sim >= .86),
    }


def main():
    terms = pd.read_csv(TERMS, usecols=["term_id", "query"]); terms.term_id = terms.term_id.astype(str)
    qmap = terms.set_index("term_id")["query"].to_dict()
    train = pd.read_csv(TRAIN_PAIRS, usecols=["term_id", "item_id"]); train.term_id=train.term_id.astype(str);train.item_id=train.item_id.astype(str)
    sub = pd.read_csv(SUB_PAIRS, usecols=["id", "term_id", "item_id"]); sub.term_id=sub.term_id.astype(str);sub.item_id=sub.item_id.astype(str);sub.id=sub.id.astype(str)
    train_ids = train.term_id.drop_duplicates().to_numpy(); test_ids = sub.term_id.drop_duplicates().to_numpy()

    e5_ids, e5 = load_embedding("v5_e5base")
    mini_ids, mini = load_embedding("minilm")
    if not np.array_equal(e5_ids, mini_ids):
        mini_pos = {x:i for i,x in enumerate(mini_ids)}
        mini = mini[np.array([mini_pos[x] for x in e5_ids])]
    all_ids = e5_ids

    # Honest train-query transfer validation on candidate-shaped v34 rows.
    cv_maps = enrich_query_relation(neighbor_pairs(train_ids, train_ids, all_ids, e5, mini), qmap)
    v34 = pd.read_parquet(V34, columns=["term_id", "item_id", "label"]);v34.term_id=v34.term_id.astype(str);v34.item_id=v34.item_id.astype(str)
    cv = transfer_rows(cv_maps, train.drop_duplicates(), v34)
    cv = cv.sort_values("relation_score", ascending=False).drop_duplicates(["target_term_id", "item_id"])
    cv_rows = []
    for name, mask in config_masks(cv).items():
        y = cv.loc[mask, "label"]
        cv_rows.append({"config": name, "rows": int(len(y)), "known_positive_rate": float(y.mean()) if len(y) else None,
                        "known_positives": int(y.sum()) if len(y) else 0, "target_queries": int(cv.loc[mask, "target_term_id"].nunique())})

    maps = enrich_query_relation(neighbor_pairs(test_ids, train_ids, all_ids, e5, mini), qmap)
    transferred = transfer_rows(maps, train.drop_duplicates(), sub)
    transferred = transferred.sort_values("relation_score", ascending=False).drop_duplicates("id")
    base = pd.read_csv(BASE); base.id=base.id.astype(str)
    pred0 = base.prediction.to_numpy(np.int8); id_to_row = pd.Series(np.arange(len(base)), index=base.id)
    OUT_DIR.mkdir(parents=True, exist_ok=True); REPORT.parent.mkdir(parents=True, exist_ok=True); REVIEW.parent.mkdir(parents=True, exist_ok=True)
    rows=[]; reviews=[]
    for name, mask in config_masks(transferred).items():
        cand = transferred.loc[mask].copy(); cand["row"] = cand.id.map(id_to_row); cand=cand[cand.row.notna()]
        cand = cand[pred0[cand.row.astype(int).to_numpy()] == 0].sort_values("relation_score", ascending=False)
        for cap in [250, 500, 1000, 2000, 5000]:
            take = cand.head(cap); pred=pred0.copy(); pred[take.row.astype(int).to_numpy()] = 1
            path=OUT_DIR/f"FINAL_CANDIDATE_v92_{name}_add{cap}.csv"
            pd.DataFrame({"id":base.id,"prediction":pred}).to_csv(path,index=False)
            rows.append({"config":name,"cap":cap,"used_adds":int(len(take)),"pool":int(len(cand)),
                         "positives":int(pred.sum()),"ratio":float(pred.mean()),"file":str(path),
                         "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        if len(cand): reviews.append(cand.head(200).assign(config=name))
    if reviews: pd.concat(reviews,ignore_index=True).to_csv(REVIEW,index=False)
    report={"method":"dual-embedding semantic query memory with direction guard","cv":cv_rows,
            "map_similarity":maps[["e5_sim","mini_sim","char_sim","relation_score"]].describe().to_dict(),
            "unique_transferred_test_rows":int(len(transferred)),"candidates":rows}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
