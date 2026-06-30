from __future__ import annotations

"""Build domain-matched random negatives using test queries and random catalog items."""

import gc
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = PROC / "test_query_random_feature_frame.parquet"
N = 300_000
SEED = 2026
CHUNK = 50_000

cv_cfg = runpy.run_path(str(ROOT / "scripts" / "35_train_pu_random_negative_cv.py"))
FEATURES: list[str] = cv_cfg["FEATURES"]
feature_module = runpy.run_path(str(ROOT / "scripts" / "30_v6_dual_embedding_cv_threshold.py"))
add_text_features = feature_module["add_text_features"]


def score_pairs(term_ids: np.ndarray, item_ids: np.ndarray, prefix: str) -> np.ndarray:
    term_table = pd.read_parquet(PROC / f"{prefix}_all_term_ids.parquet")["term_id"].astype(str)
    item_table = pd.read_parquet(PROC / f"{prefix}_all_item_ids.parquet")["item_id"].astype(str)
    term_map = dict(zip(term_table, range(len(term_table))))
    item_map = dict(zip(item_table, range(len(item_table))))
    ti = np.fromiter((term_map[x] for x in term_ids), dtype=np.int64, count=len(term_ids))
    ii = np.fromiter((item_map[x] for x in item_ids), dtype=np.int64, count=len(item_ids))
    term_emb = np.load(PROC / f"{prefix}_all_terms_emb.npy", mmap_mode="r")
    item_emb = np.load(PROC / f"{prefix}_all_items_emb.npy", mmap_mode="r")
    scores = np.empty(len(term_ids), dtype=np.float32)
    for start in range(0, len(term_ids), CHUNK):
        end = min(start + CHUNK, len(term_ids))
        a = np.asarray(term_emb[ti[start:end]], dtype=np.float32)
        b = np.asarray(item_emb[ii[start:end]], dtype=np.float32)
        scores[start:end] = np.sum(a * b, axis=1)
    return scores


def main() -> None:
    if OUT.exists():
        print(f"Cache exists: {OUT}")
        return
    rng = np.random.default_rng(SEED)
    test_terms = pd.read_csv(RAW / "submission_pairs.csv", usecols=["term_id"])["term_id"]
    sampled_terms = test_terms.sample(n=N, replace=True, random_state=SEED).astype(str).to_numpy()
    items = pd.read_csv(
        RAW / "items.csv",
        usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    )
    items["item_id"] = items["item_id"].astype(str)
    sampled_item_idx = rng.integers(0, len(items), N)
    sampled_items = items.iloc[sampled_item_idx]["item_id"].to_numpy()

    frame = pd.DataFrame({"term_id": sampled_terms, "item_id": sampled_items})
    frame["mini_score"] = score_pairs(sampled_terms, sampled_items, "minilm")
    frame["e5_score"] = score_pairs(sampled_terms, sampled_items, "v5_e5base")
    terms = pd.read_csv(RAW / "terms.csv")
    terms["term_id"] = terms["term_id"].astype(str)
    frame = frame.merge(terms, on="term_id", how="left", validate="many_to_one")
    frame = frame.merge(items, on="item_id", how="left", validate="many_to_one")
    frame = add_text_features(frame)
    frame["score_diff_e5_minus_mini"] = frame["e5_score"] - frame["mini_score"]
    frame["score_product"] = frame["e5_score"] * frame["mini_score"]
    frame["score_mean_dual"] = (frame["e5_score"] + frame["mini_score"]) / 2.0
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame[["term_id", "item_id"] + FEATURES].to_parquet(OUT, index=False)
    print(f"Saved {len(frame):,} rows: {OUT}")
    del frame
    gc.collect()


if __name__ == "__main__":
    main()
