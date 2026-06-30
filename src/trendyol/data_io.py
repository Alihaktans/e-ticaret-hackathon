from __future__ import annotations

import pandas as pd
import polars as pl

from trendyol.config import RAW_DIR, REQUIRED_RAW_FILES


def read_items_pd() -> pd.DataFrame:
    return pl.read_csv(RAW_DIR / REQUIRED_RAW_FILES["items"]).to_pandas()


def read_terms_pd() -> pd.DataFrame:
    return pl.read_csv(RAW_DIR / REQUIRED_RAW_FILES["terms"]).to_pandas()


def read_training_pairs_pd() -> pd.DataFrame:
    return pl.read_csv(RAW_DIR / REQUIRED_RAW_FILES["training_pairs"]).to_pandas()


def read_submission_pairs_pd() -> pd.DataFrame:
    return pl.read_csv(RAW_DIR / REQUIRED_RAW_FILES["submission_pairs"]).to_pandas()
