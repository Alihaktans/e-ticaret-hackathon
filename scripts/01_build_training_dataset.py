from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import PROCESSED_DIR, EXPERIMENT_REPORTS_DIR
from trendyol.data_io import read_items_pd, read_terms_pd, read_training_pairs_pd
from trendyol.negative_sampling import NegativeSamplingConfig, generate_negative_pairs
from trendyol.text_preprocess import normalize_text, extract_root_category, build_item_text


def enrich_pairs(
    pairs: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    df = pairs.merge(terms, on="term_id", how="left")
    df = df.merge(items, on="item_id", how="left")

    missing_query = int(df["query"].isna().sum())
    missing_title = int(df["title"].isna().sum())

    if missing_query or missing_title:
        raise ValueError(
            f"Metadata join problem: missing_query={missing_query}, missing_title={missing_title}"
        )

    text_cols = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]

    for col in text_cols:
        df[col] = df[col].fillna("").map(normalize_text)

    df["root_category"] = df["category"].map(extract_root_category)

    # Ürün metni model/feature aşamasında tekrar tekrar üretmek yerine cache'liyoruz.
    df["item_text"] = df.apply(build_item_text, axis=1)
    df["pair_text"] = "arama: " + df["query"] + " [SEP] " + df["item_text"]

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = NegativeSamplingConfig(
        random_state=42,
        random_negatives_per_positive=1,
        same_root_negatives_per_positive=1,
    )

    print("=" * 100)
    print("01 BUILD TRAINING DATASET")
    print("=" * 100)

    print("Reading raw tables...")
    items = read_items_pd()
    terms = read_terms_pd()
    train_pos = read_training_pairs_pd()

    print(f"items     : {len(items):,}")
    print(f"terms     : {len(terms):,}")
    print(f"train_pos : {len(train_pos):,}")

    train_pos = train_pos[["id", "term_id", "item_id", "label"]].copy()
    train_pos["negative_type"] = "positive"

    print("\nGenerating negatives...")
    negatives = generate_negative_pairs(train_pos, items, cfg)

    print("\nCombining positives + negatives...")
    train_pairs = pd.concat(
        [
            train_pos[["id", "term_id", "item_id", "label", "negative_type"]],
            negatives[["id", "term_id", "item_id", "label", "negative_type"]],
        ],
        ignore_index=True,
    )

    train_pairs = train_pairs.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)

    print(train_pairs["label"].value_counts())
    print(train_pairs["negative_type"].value_counts())

    print("\nEnriching dataset...")
    train_enriched = enrich_pairs(train_pairs, terms, items)

    pairs_path = PROCESSED_DIR / "train_pairs_v1.parquet"
    enriched_path = PROCESSED_DIR / "train_enriched_v1.parquet"

    print("\nSaving parquet files...")
    train_pairs.to_parquet(pairs_path, index=False)
    train_enriched.to_parquet(enriched_path, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.__dict__,
        "rows": {
            "positive": int((train_pairs["label"] == 1).sum()),
            "negative": int((train_pairs["label"] == 0).sum()),
            "total": int(len(train_pairs)),
        },
        "label_distribution": {
            str(k): int(v)
            for k, v in train_pairs["label"].value_counts().sort_index().to_dict().items()
        },
        "negative_type_distribution": {
            str(k): int(v)
            for k, v in train_pairs["negative_type"].value_counts().to_dict().items()
        },
        "outputs": {
            "train_pairs": str(pairs_path),
            "train_enriched": str(enriched_path),
        },
        "notes": [
            "Training data is positive-only in raw form.",
            "Negatives are synthetic and contain easy_random + same_root_hard examples.",
            "This dataset is suitable for first baseline experiments, not final validation truth.",
        ],
    }

    report_path = EXPERIMENT_REPORTS_DIR / "train_dataset_v1_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved: {pairs_path}")
    print(f"Saved: {enriched_path}")
    print(f"Saved: {report_path}")
    print("\nPreview:")
    print(train_enriched[["id", "term_id", "item_id", "label", "negative_type", "query", "title", "root_category"]].head())


if __name__ == "__main__":
    main()
