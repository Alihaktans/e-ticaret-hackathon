from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import RAW_DIR, PROCESSED_DIR, EXPERIMENT_REPORTS_DIR
from trendyol.text_preprocess import normalize_text, build_item_text
from trendyol.embedding import load_sentence_model, encode_texts, cosine_by_index


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_TAG = "minilm"
BATCH_SIZE = 256

TERM_CACHE = PROCESSED_DIR / f"{MODEL_TAG}_all_terms_emb.npy"
ITEM_CACHE = PROCESSED_DIR / f"{MODEL_TAG}_all_items_emb.npy"
TERM_IDS_PATH = PROCESSED_DIR / f"{MODEL_TAG}_all_term_ids.parquet"
ITEM_IDS_PATH = PROCESSED_DIR / f"{MODEL_TAG}_all_item_ids.parquet"

TRAIN_SCORE_PATH = PROCESSED_DIR / "embedding_v1_train_scores.parquet"
REPORT_PATH = EXPERIMENT_REPORTS_DIR / "embedding_v1_train_score_report.json"


def build_all_entity_embeddings():
    print("=" * 100)
    print("BUILD ALL ENTITY EMBEDDINGS")
    print("=" * 100)

    terms = pl.read_csv(RAW_DIR / "terms.csv").sort("term_id").to_pandas()
    items = pl.read_csv(RAW_DIR / "items.csv").sort("item_id").to_pandas()

    print(f"Terms: {len(terms):,}")
    print(f"Items: {len(items):,}")

    terms[["term_id"]].to_parquet(TERM_IDS_PATH, index=False)
    items[["item_id"]].to_parquet(ITEM_IDS_PATH, index=False)

    model = load_sentence_model(MODEL_NAME)

    query_texts = terms["query"].fillna("").map(normalize_text).tolist()

    print("Building item texts...")
    item_texts = [
        build_item_text(row)
        for _, row in tqdm(items.iterrows(), total=len(items), desc="item text")
    ]

    term_emb = encode_texts(
        model=model,
        texts=query_texts,
        batch_size=BATCH_SIZE,
        cache_path=TERM_CACHE,
    )

    item_emb = encode_texts(
        model=model,
        texts=item_texts,
        batch_size=BATCH_SIZE,
        cache_path=ITEM_CACHE,
    )

    print(f"term_emb: {term_emb.shape} {term_emb.dtype}")
    print(f"item_emb: {item_emb.shape} {item_emb.dtype}")

    return terms, items, term_emb, item_emb


def score_train_pairs(terms, items, term_emb, item_emb):
    print("=" * 100)
    print("SCORE TRAIN PAIRS")
    print("=" * 100)

    train_path = PROCESSED_DIR / "train_enriched_v1.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")

    train = pd.read_parquet(
        train_path,
        columns=["id", "term_id", "item_id", "label", "negative_type"],
    )

    print(f"Train rows: {len(train):,}")

    term_to_idx = dict(zip(terms["term_id"], range(len(terms))))
    item_to_idx = dict(zip(items["item_id"], range(len(items))))

    scores = np.empty(len(train), dtype=np.float32)
    chunk_size = 500_000

    for start in tqdm(range(0, len(train), chunk_size), desc="train score chunks"):
        end = min(start + chunk_size, len(train))
        chunk = train.iloc[start:end]

        term_idx = chunk["term_id"].map(term_to_idx).to_numpy(dtype=np.int64)
        item_idx = chunk["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)

        scores[start:end] = cosine_by_index(term_emb, item_emb, term_idx, item_idx)

    train_scores = train.copy()
    train_scores["embedding_score"] = scores

    train_scores.to_parquet(TRAIN_SCORE_PATH, index=False)
    print(f"Saved: {TRAIN_SCORE_PATH}")

    return train_scores


def make_report(train_scores: pd.DataFrame):
    print("=" * 100)
    print("TRAIN SCORE ANALYSIS")
    print("=" * 100)

    label_summary = (
        train_scores
        .groupby("label")["embedding_score"]
        .describe()
        .reset_index()
    )

    type_summary = (
        train_scores
        .groupby("negative_type")["embedding_score"]
        .describe()
        .reset_index()
    )

    print("\nBy label:")
    print(label_summary)

    print("\nBy negative_type:")
    print(type_summary)

    pos = train_scores.loc[train_scores["label"] == 1, "embedding_score"]
    neg = train_scores.loc[train_scores["label"] == 0, "embedding_score"]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "train_score_path": str(TRAIN_SCORE_PATH),
        "rows": int(len(train_scores)),
        "positive_score_mean": float(pos.mean()),
        "negative_score_mean": float(neg.mean()),
        "positive_score_median": float(pos.median()),
        "negative_score_median": float(neg.median()),
        "mean_gap_pos_minus_neg": float(pos.mean() - neg.mean()),
        "label_summary": label_summary.to_dict(orient="records"),
        "negative_type_summary": type_summary.to_dict(orient="records"),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {REPORT_PATH}")

    print("\nDONE")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    terms, items, term_emb, item_emb = build_all_entity_embeddings()
    train_scores = score_train_pairs(terms, items, term_emb, item_emb)
    make_report(train_scores)


if __name__ == "__main__":
    main()
