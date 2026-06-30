from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import RAW_DIR, PROCESSED_DIR, SUBMISSIONS_DIR
from trendyol.text_preprocess import normalize_text, build_item_text
from trendyol.embedding import load_sentence_model, encode_texts, cosine_by_index


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256

TOP_RATIO = 0.08
MIN_POS_PER_QUERY = 1
MAX_POS_PER_QUERY = 20

MODEL_TAG = "minilm"
RUN_NAME = "embedding_v1_minilm_topratio_008"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("02 MAKE EMBEDDING SUBMISSION")
    print("=" * 100)

    print("Reading raw tables...")
    pairs = pl.read_csv(RAW_DIR / "submission_pairs.csv")
    terms = pl.read_csv(RAW_DIR / "terms.csv")
    items = pl.read_csv(RAW_DIR / "items.csv")

    test_terms = (
        pairs.select("term_id")
        .unique()
        .join(terms, on="term_id", how="left")
        .sort("term_id")
    )

    test_items = (
        pairs.select("item_id")
        .unique()
        .join(items, on="item_id", how="left")
        .sort("item_id")
    )

    print(f"Unique test terms: {test_terms.height:,}")
    print(f"Unique test items: {test_items.height:,}")

    test_terms_pd = test_terms.to_pandas()
    test_items_pd = test_items.to_pandas()

    model = load_sentence_model(MODEL_NAME)

    query_texts = test_terms_pd["query"].fillna("").map(normalize_text).tolist()

    print("Building item texts...")
    item_texts = [
        build_item_text(row)
        for _, row in tqdm(test_items_pd.iterrows(), total=len(test_items_pd), desc="item text")
    ]

    term_emb = encode_texts(
        model=model,
        texts=query_texts,
        batch_size=BATCH_SIZE,
        cache_path=PROCESSED_DIR / f"{MODEL_TAG}_test_term_emb.npy",
    )

    item_emb = encode_texts(
        model=model,
        texts=item_texts,
        batch_size=BATCH_SIZE,
        cache_path=PROCESSED_DIR / f"{MODEL_TAG}_test_item_emb.npy",
    )

    term_to_idx = dict(zip(test_terms_pd["term_id"], range(len(test_terms_pd))))
    item_to_idx = dict(zip(test_items_pd["item_id"], range(len(test_items_pd))))

    pairs_pd = pairs.to_pandas()

    print("Scoring pairs...")
    scores = np.empty(len(pairs_pd), dtype=np.float32)
    chunk_size = 500_000

    for start in tqdm(range(0, len(pairs_pd), chunk_size), desc="score chunks"):
        end = min(start + chunk_size, len(pairs_pd))
        chunk = pairs_pd.iloc[start:end]

        term_idx = chunk["term_id"].map(term_to_idx).to_numpy(dtype=np.int64)
        item_idx = chunk["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)

        scores[start:end] = cosine_by_index(term_emb, item_emb, term_idx, item_idx)

    pairs_pd["score"] = scores
    pairs_pd["prediction"] = 0

    print("Applying query-level top-ratio rule...")

    for _, idx in tqdm(pairs_pd.groupby("term_id").groups.items(), desc="binarize"):
        n = len(idx)

        k = round(n * TOP_RATIO)
        k = max(MIN_POS_PER_QUERY, k)
        k = min(MAX_POS_PER_QUERY, k)
        k = min(k, n)

        top_idx = pairs_pd.loc[idx, "score"].nlargest(k).index
        pairs_pd.loc[top_idx, "prediction"] = 1

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    submission = sample[["id"]].merge(
        pairs_pd[["id", "prediction"]],
        on="id",
        how="left",
    )

    submission["prediction"] = submission["prediction"].fillna(0).astype(int)

    assert list(submission.columns) == ["id", "prediction"]
    assert len(submission) == len(sample)
    assert submission["id"].equals(sample["id"])
    assert set(submission["prediction"].unique()) <= {0, 1}

    out_path = SUBMISSIONS_DIR / f"{RUN_NAME}.csv"
    submission.to_csv(out_path, index=False)

    score_path = PROCESSED_DIR / f"{RUN_NAME}_scores.parquet"
    pairs_pd[["id", "term_id", "item_id", "score"]].to_parquet(score_path, index=False)

    print("\nDONE")
    print(f"Saved submission: {out_path}")
    print(f"Saved scores    : {score_path}")
    print("\nPrediction counts:")
    print(submission["prediction"].value_counts())
    print("\nPrediction ratio:")
    print(submission["prediction"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
