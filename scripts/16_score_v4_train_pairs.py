from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import PROCESSED_DIR, EXPERIMENT_REPORTS_DIR
from trendyol.embedding import cosine_by_index


RUN_NAME = "embedding_v4_train_scores"

TRAIN_PAIRS_PATH = PROCESSED_DIR / "train_pairs_v4_semantic_hard.parquet"

TERM_EMB_PATH = PROCESSED_DIR / "minilm_all_terms_emb.npy"
ITEM_EMB_PATH = PROCESSED_DIR / "minilm_all_items_emb.npy"
TERM_IDS_PATH = PROCESSED_DIR / "minilm_all_term_ids.parquet"
ITEM_IDS_PATH = PROCESSED_DIR / "minilm_all_item_ids.parquet"

OUT_PATH = PROCESSED_DIR / "embedding_v4_train_scores.parquet"
REPORT_PATH = EXPERIMENT_REPORTS_DIR / "embedding_v4_train_score_report.json"


def main() -> None:
    print("=" * 100)
    print("16 SCORE V4 TRAIN PAIRS")
    print("=" * 100)

    for path in [TRAIN_PAIRS_PATH, TERM_EMB_PATH, ITEM_EMB_PATH, TERM_IDS_PATH, ITEM_IDS_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Reading train pairs...")
    train = pd.read_parquet(TRAIN_PAIRS_PATH)
    print(f"Train rows: {len(train):,}")
    print(train["label"].value_counts())
    print(train["negative_type"].value_counts())

    print("Reading ids...")
    term_ids = pd.read_parquet(TERM_IDS_PATH)
    item_ids = pd.read_parquet(ITEM_IDS_PATH)

    term_ids["term_id"] = term_ids["term_id"].astype(str)
    item_ids["item_id"] = item_ids["item_id"].astype(str)

    term_to_idx = dict(zip(term_ids["term_id"], range(len(term_ids))))
    item_to_idx = dict(zip(item_ids["item_id"], range(len(item_ids))))

    print("Loading embeddings...")
    term_emb = np.load(TERM_EMB_PATH, mmap_mode="r")
    item_emb = np.load(ITEM_EMB_PATH, mmap_mode="r")

    print(f"term_emb: {term_emb.shape} {term_emb.dtype}")
    print(f"item_emb: {item_emb.shape} {item_emb.dtype}")

    scores = np.empty(len(train), dtype=np.float32)
    chunk_size = 500_000

    for start in tqdm(range(0, len(train), chunk_size), desc="score chunks"):
        end = min(start + chunk_size, len(train))
        chunk = train.iloc[start:end]

        term_idx = chunk["term_id"].map(term_to_idx).to_numpy(dtype=np.int64)
        item_idx = chunk["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)

        scores[start:end] = cosine_by_index(term_emb, item_emb, term_idx, item_idx)

    out = train.copy()
    out["embedding_score"] = scores

    out.to_parquet(OUT_PATH, index=False)

    print("\nScore summary by label:")
    label_summary = out.groupby("label")["embedding_score"].describe()
    print(label_summary)

    print("\nScore summary by negative_type:")
    type_summary = out.groupby("negative_type")["embedding_score"].describe()
    print(type_summary)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "out_path": str(OUT_PATH),
        "rows": int(len(out)),
        "label_distribution": {
            str(k): int(v)
            for k, v in out["label"].value_counts().sort_index().to_dict().items()
        },
        "negative_type_distribution": {
            str(k): int(v)
            for k, v in out["negative_type"].value_counts().to_dict().items()
        },
        "label_score_summary": label_summary.reset_index().to_dict(orient="records"),
        "negative_type_score_summary": type_summary.reset_index().to_dict(orient="records"),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved: {OUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
