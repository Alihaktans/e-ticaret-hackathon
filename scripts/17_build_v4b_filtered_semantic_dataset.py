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
from trendyol.text_preprocess import normalize_text, extract_root_category, build_item_text
from trendyol.embedding import cosine_by_index


RUN_NAME = "v4b_filtered_semantic_hard"

RANDOM_STATE = 42

CANDIDATE_PATH = PROCESSED_DIR / "v4_semantic_hard_negative_candidates.parquet"

TERM_EMB_PATH = PROCESSED_DIR / "minilm_all_terms_emb.npy"
ITEM_EMB_PATH = PROCESSED_DIR / "minilm_all_items_emb.npy"
TERM_IDS_PATH = PROCESSED_DIR / "minilm_all_term_ids.parquet"
ITEM_IDS_PATH = PROCESSED_DIR / "minilm_all_item_ids.parquet"

TRAIN_PAIRS_PATH = PROCESSED_DIR / "train_pairs_v4b_filtered_semantic_hard.parquet"
TRAIN_ENRICHED_PATH = PROCESSED_DIR / "train_enriched_v4b_filtered_semantic_hard.parquet"
TRAIN_SCORE_PATH = PROCESSED_DIR / "embedding_v4b_train_scores.parquet"

REPORT_PATH = EXPERIMENT_REPORTS_DIR / "v4b_filtered_semantic_hard_report.json"

# Güvenlik filtreleri
MIN_SEMANTIC_RANK = 10
MAX_SEMANTIC_RANK = 120
MAX_ABOVE_POS_P75 = 0.03

EASY_RANDOM_NEG_PER_POS = 1
FILTERED_SEMANTIC_NEG_PER_POS = 1


def build_positive_maps(train_pos: pd.DataFrame):
    train_pos["term_id"] = train_pos["term_id"].astype(str)
    train_pos["item_id"] = train_pos["item_id"].astype(str)

    pos_by_term = (
        train_pos
        .groupby("term_id")["item_id"]
        .apply(lambda x: set(map(str, x)))
        .to_dict()
    )

    return pos_by_term


def enrich_pairs(pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    print("Enriching V4B dataset...")

    df = pairs.merge(terms, on="term_id", how="left")
    df = df.merge(items, on="item_id", how="left")

    missing_query = int(df["query"].isna().sum())
    missing_title = int(df["title"].isna().sum())

    if missing_query or missing_title:
        raise ValueError(f"Metadata join problem: query={missing_query}, title={missing_title}")

    text_cols = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]

    for col in text_cols:
        df[col] = df[col].fillna("").map(normalize_text)

    df["root_category"] = df["category"].map(extract_root_category)
    df["item_text"] = df.apply(build_item_text, axis=1)
    df["pair_text"] = "arama: " + df["query"] + " [SEP] " + df["item_text"]

    return df


def score_pairs(train_pairs: pd.DataFrame) -> pd.DataFrame:
    print("Scoring V4B train pairs...")

    term_ids = pd.read_parquet(TERM_IDS_PATH)
    item_ids = pd.read_parquet(ITEM_IDS_PATH)

    term_ids["term_id"] = term_ids["term_id"].astype(str)
    item_ids["item_id"] = item_ids["item_id"].astype(str)

    term_to_idx = dict(zip(term_ids["term_id"], range(len(term_ids))))
    item_to_idx = dict(zip(item_ids["item_id"], range(len(item_ids))))

    term_emb = np.load(TERM_EMB_PATH, mmap_mode="r")
    item_emb = np.load(ITEM_EMB_PATH, mmap_mode="r")

    scores = np.empty(len(train_pairs), dtype=np.float32)

    chunk_size = 500_000

    for start in tqdm(range(0, len(train_pairs), chunk_size), desc="score chunks"):
        end = min(start + chunk_size, len(train_pairs))
        chunk = train_pairs.iloc[start:end]

        term_idx = chunk["term_id"].map(term_to_idx).to_numpy(dtype=np.int64)
        item_idx = chunk["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)

        scores[start:end] = cosine_by_index(term_emb, item_emb, term_idx, item_idx)

    out = train_pairs.copy()
    out["embedding_score"] = scores

    return out


def generate_easy_random(train_pos: pd.DataFrame, all_item_ids: np.ndarray, pos_by_term: dict[str, set[str]]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    print("Generating easy_random negatives...")

    for i, row in tqdm(enumerate(train_pos.itertuples(index=False)), total=len(train_pos), desc="easy_random"):
        term_id = str(row.term_id)

        for _ in range(EASY_RANDOM_NEG_PER_POS):
            for _try in range(100):
                item_id = str(all_item_ids[rng.integers(0, len(all_item_ids))])

                if item_id not in pos_by_term.get(term_id, set()):
                    rows.append(
                        {
                            "id": f"V4B_EASY_{i}",
                            "term_id": term_id,
                            "item_id": item_id,
                            "label": 0,
                            "negative_type": "easy_random",
                        }
                    )
                    break

    return pd.DataFrame(rows)


def generate_filtered_semantic(train_pos: pd.DataFrame, candidates: pd.DataFrame, pos_by_term: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Computing positive score stats per term...")

    # Pozitif çiftlerin embedding score'unu üret.
    pos_scores = score_pairs(train_pos[["id", "term_id", "item_id", "label"]].copy())

    pos_stats = (
        pos_scores
        .groupby("term_id")["embedding_score"]
        .quantile(0.75)
        .rename("pos_p75")
        .reset_index()
    )

    candidates = candidates.copy()
    candidates["term_id"] = candidates["term_id"].astype(str)
    candidates["item_id"] = candidates["item_id"].astype(str)

    cand = candidates.merge(pos_stats, on="term_id", how="left")

    before = len(cand)

    cand = cand[
        (cand["semantic_rank"] >= MIN_SEMANTIC_RANK)
        & (cand["semantic_rank"] <= MAX_SEMANTIC_RANK)
        & (cand["embedding_score"] <= cand["pos_p75"] + MAX_ABOVE_POS_P75)
    ].copy()

    after = len(cand)

    print(f"Candidates before filter: {before:,}")
    print(f"Candidates after filter : {after:,}")
    print(f"Kept ratio              : {after / before:.4f}")

    cand = cand.sort_values(["term_id", "semantic_rank"]).reset_index(drop=True)

    cand_by_term = {
        term_id: group["item_id"].tolist()
        for term_id, group in cand.groupby("term_id")
    }

    counters = {}
    rows = []

    print("Generating filtered_semantic_hard negatives...")

    for i, row in tqdm(enumerate(train_pos.itertuples(index=False)), total=len(train_pos), desc="filtered_semantic"):
        term_id = str(row.term_id)
        pool = cand_by_term.get(term_id, [])

        if not pool:
            continue

        start = counters.get(term_id, 0)
        chosen = None

        for offset in range(len(pool)):
            item_id = str(pool[(start + offset) % len(pool)])

            if item_id not in pos_by_term.get(term_id, set()):
                chosen = item_id
                counters[term_id] = start + offset + 1
                break

        if chosen is None:
            continue

        rows.append(
            {
                "id": f"V4B_FILTERED_SEM_{i}",
                "term_id": term_id,
                "item_id": chosen,
                "label": 0,
                "negative_type": "filtered_semantic_hard",
            }
        )

    neg = pd.DataFrame(rows)

    return neg, cand


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("17 BUILD V4B FILTERED SEMANTIC HARD DATASET")
    print("=" * 100)

    for path in [CANDIDATE_PATH, TERM_EMB_PATH, ITEM_EMB_PATH, TERM_IDS_PATH, ITEM_IDS_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_pos = pl.read_csv(RAW_DIR / "training_pairs.csv").to_pandas()
    train_pos["term_id"] = train_pos["term_id"].astype(str)
    train_pos["item_id"] = train_pos["item_id"].astype(str)

    train_pos_out = train_pos[["id", "term_id", "item_id", "label"]].copy()
    train_pos_out["negative_type"] = "positive"

    pos_by_term = build_positive_maps(train_pos)

    candidates = pd.read_parquet(CANDIDATE_PATH)

    items = pl.read_csv(RAW_DIR / "items.csv").to_pandas()
    terms = pl.read_csv(RAW_DIR / "terms.csv").to_pandas()

    items["item_id"] = items["item_id"].astype(str)
    terms["term_id"] = terms["term_id"].astype(str)

    all_item_ids = items["item_id"].to_numpy()

    filtered_sem, filtered_candidates = generate_filtered_semantic(train_pos, candidates, pos_by_term)
    easy = generate_easy_random(train_pos, all_item_ids, pos_by_term)

    negatives = pd.concat([filtered_sem, easy], ignore_index=True)

    before_dedup = len(negatives)
    negatives = negatives.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)
    after_dedup = len(negatives)

    print(f"Negatives before dedup: {before_dedup:,}")
    print(f"Negatives after dedup : {after_dedup:,}")

    train_pairs = pd.concat(
        [
            train_pos_out[["id", "term_id", "item_id", "label", "negative_type"]],
            negatives[["id", "term_id", "item_id", "label", "negative_type"]],
        ],
        ignore_index=True,
    )

    train_pairs = train_pairs.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    train_enriched = enrich_pairs(train_pairs, terms, items)

    train_scores = score_pairs(train_pairs)

    train_pairs.to_parquet(TRAIN_PAIRS_PATH, index=False)
    train_enriched.to_parquet(TRAIN_ENRICHED_PATH, index=False)
    train_scores.to_parquet(TRAIN_SCORE_PATH, index=False)

    label_summary = train_scores.groupby("label")["embedding_score"].describe()
    type_summary = train_scores.groupby("negative_type")["embedding_score"].describe()

    print("\nScore summary by label:")
    print(label_summary)

    print("\nScore summary by negative_type:")
    print(type_summary)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "filter_config": {
            "min_semantic_rank": MIN_SEMANTIC_RANK,
            "max_semantic_rank": MAX_SEMANTIC_RANK,
            "max_above_pos_p75": MAX_ABOVE_POS_P75,
        },
        "candidate_rows_before_filter": int(len(candidates)),
        "candidate_rows_after_filter": int(len(filtered_candidates)),
        "train_rows": int(len(train_pairs)),
        "negative_before_dedup": int(before_dedup),
        "negative_after_dedup": int(after_dedup),
        "label_distribution": {
            str(k): int(v)
            for k, v in train_pairs["label"].value_counts().sort_index().to_dict().items()
        },
        "negative_type_distribution": {
            str(k): int(v)
            for k, v in train_pairs["negative_type"].value_counts().to_dict().items()
        },
        "score_summary_by_label": label_summary.reset_index().to_dict(orient="records"),
        "score_summary_by_negative_type": type_summary.reset_index().to_dict(orient="records"),
        "outputs": {
            "train_pairs": str(TRAIN_PAIRS_PATH),
            "train_enriched": str(TRAIN_ENRICHED_PATH),
            "train_scores": str(TRAIN_SCORE_PATH),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved: {TRAIN_PAIRS_PATH}")
    print(f"Saved: {TRAIN_ENRICHED_PATH}")
    print(f"Saved: {TRAIN_SCORE_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
