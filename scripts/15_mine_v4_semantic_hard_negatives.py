from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import RAW_DIR, PROCESSED_DIR, EXPERIMENT_REPORTS_DIR
from trendyol.text_preprocess import normalize_text, extract_root_category, build_item_text


RUN_NAME = "v4_semantic_hard_negatives"

TERM_EMB_PATH = PROCESSED_DIR / "minilm_all_terms_emb.npy"
ITEM_EMB_PATH = PROCESSED_DIR / "minilm_all_items_emb.npy"
TERM_IDS_PATH = PROCESSED_DIR / "minilm_all_term_ids.parquet"
ITEM_IDS_PATH = PROCESSED_DIR / "minilm_all_item_ids.parquet"

CANDIDATE_PATH = PROCESSED_DIR / "v4_semantic_hard_negative_candidates.parquet"
TRAIN_PAIRS_PATH = PROCESSED_DIR / "train_pairs_v4_semantic_hard.parquet"
TRAIN_ENRICHED_PATH = PROCESSED_DIR / "train_enriched_v4_semantic_hard.parquet"
REPORT_PATH = EXPERIMENT_REPORTS_DIR / "v4_semantic_hard_negative_report.json"

RANDOM_STATE = 42

# GPU mining ayarları
TERM_CHUNK_SIZE = 96
ITEM_CHUNK_SIZE = 65_536
TOPK_RAW = 300
TOPK_KEEP = 120

# Dataset ayarı
SEMANTIC_HARD_NEG_PER_POS = 1
EASY_RANDOM_NEG_PER_POS = 1


def require_files() -> None:
    required = [
        TERM_EMB_PATH,
        ITEM_EMB_PATH,
        TERM_IDS_PATH,
        ITEM_IDS_PATH,
        RAW_DIR / "training_pairs.csv",
        RAW_DIR / "terms.csv",
        RAW_DIR / "items.csv",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")


def get_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadı. Bu script GPU için yazıldı.")
    return "cuda"


def load_ids() -> tuple[pd.DataFrame, pd.DataFrame]:
    term_ids = pd.read_parquet(TERM_IDS_PATH)
    item_ids = pd.read_parquet(ITEM_IDS_PATH)

    term_ids["term_id"] = term_ids["term_id"].astype(str)
    item_ids["item_id"] = item_ids["item_id"].astype(str)

    return term_ids, item_ids


def build_positive_maps(train_pos: pd.DataFrame) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    train_pos["term_id"] = train_pos["term_id"].astype(str)
    train_pos["item_id"] = train_pos["item_id"].astype(str)

    pos_by_term = (
        train_pos
        .groupby("term_id")["item_id"]
        .apply(lambda x: set(map(str, x)))
        .to_dict()
    )

    positive_pair_set = set(zip(train_pos["term_id"], train_pos["item_id"]))

    return pos_by_term, positive_pair_set


def mine_candidates() -> pd.DataFrame:
    print("=" * 100)
    print("MINE SEMANTIC HARD NEGATIVE CANDIDATES")
    print("=" * 100)

    if CANDIDATE_PATH.exists():
        print(f"Candidate cache exists, loading: {CANDIDATE_PATH}")
        return pd.read_parquet(CANDIDATE_PATH)

    device = get_device()
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_pos = pl.read_csv(RAW_DIR / "training_pairs.csv").to_pandas()
    train_pos["term_id"] = train_pos["term_id"].astype(str)
    train_pos["item_id"] = train_pos["item_id"].astype(str)

    pos_by_term, _ = build_positive_maps(train_pos)

    term_ids, item_ids = load_ids()

    train_terms = sorted(train_pos["term_id"].unique().tolist())
    term_to_idx = dict(zip(term_ids["term_id"], range(len(term_ids))))
    item_id_array = item_ids["item_id"].to_numpy()

    train_term_indices = np.array([term_to_idx[t] for t in train_terms], dtype=np.int64)

    print(f"Unique train terms: {len(train_terms):,}")
    print(f"All items         : {len(item_ids):,}")

    print("Loading embeddings...")
    term_emb = np.load(TERM_EMB_PATH, mmap_mode="r")
    item_emb = np.load(ITEM_EMB_PATH, mmap_mode="r")

    print(f"term_emb: {term_emb.shape} {term_emb.dtype}")
    print(f"item_emb: {item_emb.shape} {item_emb.dtype}")

    rows = []

    item_count = item_emb.shape[0]

    for term_start in tqdm(range(0, len(train_term_indices), TERM_CHUNK_SIZE), desc="term chunks"):
        term_end = min(term_start + TERM_CHUNK_SIZE, len(train_term_indices))
        batch_term_indices = train_term_indices[term_start:term_end]
        batch_term_ids = train_terms[term_start:term_end]

        term_batch_np = np.asarray(term_emb[batch_term_indices], dtype=np.float16)
        term_batch = torch.from_numpy(term_batch_np).to(device=device, dtype=torch.float16)

        batch_size = term_batch.shape[0]

        best_scores = torch.empty((batch_size, 0), device=device, dtype=torch.float16)
        best_indices = torch.empty((batch_size, 0), device=device, dtype=torch.long)

        with torch.no_grad():
            for item_start in range(0, item_count, ITEM_CHUNK_SIZE):
                item_end = min(item_start + ITEM_CHUNK_SIZE, item_count)

                item_batch_np = np.asarray(item_emb[item_start:item_end], dtype=np.float16)
                item_batch = torch.from_numpy(item_batch_np).to(device=device, dtype=torch.float16)

                sim = term_batch @ item_batch.T

                k = min(TOPK_RAW, sim.shape[1])
                chunk_scores, chunk_pos = torch.topk(sim, k=k, dim=1)
                chunk_indices = chunk_pos + item_start

                combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
                combined_indices = torch.cat([best_indices, chunk_indices], dim=1)

                keep_k = min(TOPK_RAW, combined_scores.shape[1])
                best_scores, keep_pos = torch.topk(combined_scores, k=keep_k, dim=1)
                best_indices = combined_indices.gather(1, keep_pos)

                del item_batch, sim, chunk_scores, chunk_pos, chunk_indices, combined_scores, combined_indices
                torch.cuda.empty_cache()

        best_scores_cpu = best_scores.float().cpu().numpy()
        best_indices_cpu = best_indices.cpu().numpy()

        for i, term_id in enumerate(batch_term_ids):
            positive_items = pos_by_term.get(term_id, set())
            kept = 0

            for raw_rank, (item_idx, score) in enumerate(zip(best_indices_cpu[i], best_scores_cpu[i]), start=1):
                item_id = str(item_id_array[int(item_idx)])

                if item_id in positive_items:
                    continue

                rows.append(
                    {
                        "term_id": term_id,
                        "item_id": item_id,
                        "semantic_rank": kept + 1,
                        "raw_rank": raw_rank,
                        "embedding_score": float(score),
                    }
                )

                kept += 1
                if kept >= TOPK_KEEP:
                    break

        del term_batch, best_scores, best_indices
        torch.cuda.empty_cache()

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        raise RuntimeError("No candidates mined.")

    candidates.to_parquet(CANDIDATE_PATH, index=False)

    print(f"\nSaved candidates: {CANDIDATE_PATH}")
    print(candidates.head())
    print(candidates.groupby("term_id").size().describe())

    return candidates


def generate_easy_random_negatives(
    train_pos: pd.DataFrame,
    all_item_ids: np.ndarray,
    pos_by_term: dict[str, set[str]],
) -> pd.DataFrame:
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
                            "id": f"V4_EASY_{i}",
                            "term_id": term_id,
                            "item_id": item_id,
                            "label": 0,
                            "negative_type": "easy_random",
                        }
                    )
                    break

    return pd.DataFrame(rows)


def generate_semantic_hard_negatives(
    train_pos: pd.DataFrame,
    candidates: pd.DataFrame,
    pos_by_term: dict[str, set[str]],
) -> pd.DataFrame:
    print("Preparing candidate pools...")

    cand_by_term = {
        term_id: group["item_id"].astype(str).tolist()
        for term_id, group in candidates.sort_values(["term_id", "semantic_rank"]).groupby("term_id")
    }

    counters: dict[str, int] = {}
    rows = []

    print("Generating semantic_hard negatives...")

    for i, row in tqdm(enumerate(train_pos.itertuples(index=False)), total=len(train_pos), desc="semantic_hard"):
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
                "id": f"V4_SEMANTIC_{i}",
                "term_id": term_id,
                "item_id": chosen,
                "label": 0,
                "negative_type": "semantic_hard",
            }
        )

    return pd.DataFrame(rows)


def enrich_pairs(
    pairs: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    print("Enriching V4 train dataset...")

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


def build_v4_dataset(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 100)
    print("BUILD V4 TRAIN DATASET")
    print("=" * 100)

    if TRAIN_PAIRS_PATH.exists() and TRAIN_ENRICHED_PATH.exists():
        print(f"V4 dataset cache exists: {TRAIN_PAIRS_PATH}")
        train_pairs = pd.read_parquet(TRAIN_PAIRS_PATH)
        train_enriched = pd.read_parquet(TRAIN_ENRICHED_PATH)
        return train_pairs, train_enriched

    train_pos = pl.read_csv(RAW_DIR / "training_pairs.csv").to_pandas()
    train_pos["term_id"] = train_pos["term_id"].astype(str)
    train_pos["item_id"] = train_pos["item_id"].astype(str)

    pos_by_term, positive_pair_set = build_positive_maps(train_pos)

    items = pl.read_csv(RAW_DIR / "items.csv").to_pandas()
    terms = pl.read_csv(RAW_DIR / "terms.csv").to_pandas()

    items["item_id"] = items["item_id"].astype(str)
    terms["term_id"] = terms["term_id"].astype(str)

    all_item_ids = items["item_id"].to_numpy()

    train_pos_out = train_pos[["id", "term_id", "item_id", "label"]].copy()
    train_pos_out["negative_type"] = "positive"

    semantic_neg = generate_semantic_hard_negatives(train_pos, candidates, pos_by_term)
    easy_neg = generate_easy_random_negatives(train_pos, all_item_ids, pos_by_term)

    negatives = pd.concat([semantic_neg, easy_neg], ignore_index=True)

    before = len(negatives)
    negatives = negatives.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)
    after = len(negatives)

    print(f"Negatives before dedup: {before:,}")
    print(f"Negatives after dedup : {after:,}")

    train_pairs = pd.concat(
        [
            train_pos_out[["id", "term_id", "item_id", "label", "negative_type"]],
            negatives[["id", "term_id", "item_id", "label", "negative_type"]],
        ],
        ignore_index=True,
    )

    train_pairs = train_pairs.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    train_enriched = enrich_pairs(train_pairs, terms, items)

    train_pairs.to_parquet(TRAIN_PAIRS_PATH, index=False)
    train_enriched.to_parquet(TRAIN_ENRICHED_PATH, index=False)

    print(f"Saved: {TRAIN_PAIRS_PATH}")
    print(f"Saved: {TRAIN_ENRICHED_PATH}")

    return train_pairs, train_enriched


def write_report(candidates: pd.DataFrame, train_pairs: pd.DataFrame) -> None:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "candidate_path": str(CANDIDATE_PATH),
        "train_pairs_path": str(TRAIN_PAIRS_PATH),
        "train_enriched_path": str(TRAIN_ENRICHED_PATH),
        "mining_config": {
            "term_chunk_size": TERM_CHUNK_SIZE,
            "item_chunk_size": ITEM_CHUNK_SIZE,
            "topk_raw": TOPK_RAW,
            "topk_keep": TOPK_KEEP,
        },
        "dataset_config": {
            "semantic_hard_neg_per_pos": SEMANTIC_HARD_NEG_PER_POS,
            "easy_random_neg_per_pos": EASY_RANDOM_NEG_PER_POS,
        },
        "candidate_rows": int(len(candidates)),
        "candidate_terms": int(candidates["term_id"].nunique()),
        "train_rows": int(len(train_pairs)),
        "label_distribution": {
            str(k): int(v)
            for k, v in train_pairs["label"].value_counts().sort_index().to_dict().items()
        },
        "negative_type_distribution": {
            str(k): int(v)
            for k, v in train_pairs["negative_type"].value_counts().to_dict().items()
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Saved report: {REPORT_PATH}")
    print("\nLabel distribution:")
    print(train_pairs["label"].value_counts())
    print("\nNegative type distribution:")
    print(train_pairs["negative_type"].value_counts())


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    require_files()

    candidates = mine_candidates()
    train_pairs, _ = build_v4_dataset(candidates)
    write_report(candidates, train_pairs)


if __name__ == "__main__":
    main()
