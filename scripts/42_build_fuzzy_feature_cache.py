from __future__ import annotations

"""Build reproducible RapidFuzz pair features for train and test frames."""

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FUZZY = [
    "qt_fuzz_ratio", "qt_fuzz_partial", "qt_fuzz_token_set",
    "qc_fuzz_ratio", "qc_fuzz_partial", "qc_fuzz_token_set",
]


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    q = frame["query"].fillna("").astype(str).tolist()
    t = frame["title"].fillna("").astype(str).tolist()
    c = frame["category"].fillna("").astype(str).tolist()
    funcs = [fuzz.ratio, fuzz.partial_ratio, fuzz.token_set_ratio]
    out = pd.DataFrame({"id": frame["id"].astype(str)})
    for name, func in zip(FUZZY[:3], funcs):
        out[name] = np.fromiter(
            (func(a, b) / 100.0 for a, b in zip(q, t)), dtype=np.float32, count=len(q)
        )
    for name, func in zip(FUZZY[3:], funcs):
        out[name] = np.fromiter(
            (func(a, b) / 100.0 for a, b in zip(q, c)), dtype=np.float32, count=len(q)
        )
    return out


def main() -> None:
    specs = [
        (
            PROC / "v6_dual_e5_minilm_train_frame.parquet",
            PROC / "pu_train_fuzzy_features.parquet",
            True,
        ),
        (
            PROC / "v6_dual_e5_minilm_test_frame.parquet",
            PROC / "test_fuzzy_features.parquet",
            False,
        ),
    ]
    for source, target, filter_train in specs:
        if target.exists():
            print(f"Cache exists: {target}")
            continue
        columns = ["id", "query", "title", "category"]
        if filter_train:
            columns.append("negative_type")
        frame = pd.read_parquet(source, columns=columns)
        if filter_train:
            frame = frame[frame["negative_type"].isin(["positive", "easy_random"])].copy()
        result = compute(frame)
        result.to_parquet(target, index=False)
        print(f"Saved {len(result):,} rows: {target}")


if __name__ == "__main__":
    main()
