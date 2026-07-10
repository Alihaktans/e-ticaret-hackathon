from __future__ import annotations

import pandas as pd
import polars as pl

from trendyol.config import EXPECTED_COLUMNS, RAW_DIR, REQUIRED_RAW_FILES


def _read_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / REQUIRED_RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")

    frame = pl.read_csv(path).to_pandas()
    missing = sorted(set(EXPECTED_COLUMNS[name]) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def read_items_pd() -> pd.DataFrame:
    return _read_csv("items")


def read_terms_pd() -> pd.DataFrame:
    return _read_csv("terms")


def read_training_pairs_pd() -> pd.DataFrame:
    return _read_csv("training_pairs")


def read_submission_pairs_pd() -> pd.DataFrame:
    return _read_csv("submission_pairs")


def read_sample_submission_pd() -> pd.DataFrame:
    return _read_csv("sample_submission")


def audit_raw_data() -> dict[str, int]:
    """Validate every required raw file and return row counts."""
    return {name: len(_read_csv(name)) for name in REQUIRED_RAW_FILES}
