from __future__ import annotations

"""Estimate the test positive prior from frozen embedding score mixtures.

This is a prior-shift diagnostic, not a source of labels. Test-query/random-item
pairs provide a domain-matched negative component. Known train positives provide
the positive component. A semantic-hard component is included defensively.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "experiments" / "test_class_prior_mixture.json"

N_RANDOM = 300_000
RANDOM_SEED = 2026
CHUNK = 50_000
BIN_COUNTS = [20, 30, 40, 50]


def score_random_test_query_item_pairs() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    test_terms = pd.read_csv(RAW / "submission_pairs.csv", usecols=["term_id"])["term_id"]
    sampled_terms = test_terms.sample(
        n=N_RANDOM, replace=True, random_state=RANDOM_SEED
    ).astype(str).to_numpy()
    del test_terms

    all_items = pd.read_parquet(PROC / "v5_e5base_all_item_ids.parquet")["item_id"]
    all_items = all_items.astype(str).to_numpy()
    sampled_items = all_items[rng.integers(0, len(all_items), N_RANDOM)]

    result: dict[str, np.ndarray] = {}
    for prefix, output_name in [("minilm", "mini"), ("v5_e5base", "e5")]:
        term_ids = pd.read_parquet(PROC / f"{prefix}_all_term_ids.parquet")["term_id"].astype(str)
        item_ids = pd.read_parquet(PROC / f"{prefix}_all_item_ids.parquet")["item_id"].astype(str)
        term_map = dict(zip(term_ids, range(len(term_ids))))
        item_map = dict(zip(item_ids, range(len(item_ids))))
        term_idx = np.fromiter(
            (term_map[x] for x in sampled_terms), dtype=np.int64, count=N_RANDOM
        )
        item_idx = np.fromiter(
            (item_map[x] for x in sampled_items), dtype=np.int64, count=N_RANDOM
        )
        term_emb = np.load(PROC / f"{prefix}_all_terms_emb.npy", mmap_mode="r")
        item_emb = np.load(PROC / f"{prefix}_all_items_emb.npy", mmap_mode="r")
        scores = np.empty(N_RANDOM, dtype=np.float32)
        for start in range(0, N_RANDOM, CHUNK):
            end = min(start + CHUNK, N_RANDOM)
            a = np.asarray(term_emb[term_idx[start:end]], dtype=np.float32)
            b = np.asarray(item_emb[item_idx[start:end]], dtype=np.float32)
            scores[start:end] = np.sum(a * b, axis=1)
        result[output_name] = scores
    return pd.DataFrame(result)


def load_components() -> tuple[pd.DataFrame, pd.DataFrame]:
    mini_train = pd.read_parquet(
        PROC / "embedding_v4_train_scores.parquet",
        columns=["id", "negative_type", "embedding_score"],
    ).rename(columns={"embedding_score": "mini"})
    e5_train = pd.read_parquet(
        PROC / "embedding_v5_e5base_train_scores.parquet",
        columns=["id", "embedding_score"],
    ).rename(columns={"embedding_score": "e5"})
    train = mini_train.merge(e5_train, on="id", validate="one_to_one")

    mini_test = pd.read_parquet(
        PROC / "embedding_v1_minilm_topratio_008_scores.parquet",
        columns=["id", "score"],
    ).rename(columns={"score": "mini"})
    e5_test = pd.read_parquet(
        PROC / "embedding_v5_e5base_test_scores.parquet",
        columns=["id", "score"],
    ).rename(columns={"score": "e5"})
    test = mini_test.merge(e5_test, on="id", validate="one_to_one")
    return train, test


def normalized_hist(frame: pd.DataFrame, xbins: np.ndarray, ybins: np.ndarray) -> np.ndarray:
    hist = np.histogram2d(frame["mini"], frame["e5"], bins=[xbins, ybins])[0]
    hist = hist.ravel().astype(np.float64)
    return hist / hist.sum()


def main() -> None:
    random_component = score_random_test_query_item_pairs()
    train, test = load_components()
    estimates: list[dict] = []

    for bins in BIN_COUNTS:
        xbins = np.linspace(-0.2, 0.95, bins + 1)
        ybins = np.linspace(0.65, 0.93, bins + 1)
        component_frames = [
            random_component,
            train[train["negative_type"].eq("semantic_hard")],
            train[train["negative_type"].eq("positive")],
        ]
        matrix = np.stack(
            [normalized_hist(x, xbins, ybins) for x in component_frames], axis=1
        )
        target = normalized_hist(test, xbins, ybins)
        # The appended row strongly enforces weights summing to one.
        weights, _ = nnls(
            np.vstack([matrix, np.ones((1, 3)) * 10.0]),
            np.concatenate([target, [10.0]]),
        )
        weights /= weights.sum()
        estimates.append(
            {
                "bins": bins,
                "test_query_random_weight": float(weights[0]),
                "semantic_hard_weight": float(weights[1]),
                "positive_weight": float(weights[2]),
                "histogram_rmse": float(np.sqrt(np.mean((matrix @ weights - target) ** 2))),
            }
        )

    positive_weights = np.array([x["positive_weight"] for x in estimates])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "2D NNLS mixture over MiniLM and multilingual-E5 scores",
        "warning": (
            "This assumes train-positive embedding score distributions transfer to test. "
            "Use as a class-prior estimate, not as ground-truth labels."
        ),
        "n_random_test_query_item_pairs": N_RANDOM,
        "positive_prior_mean": float(positive_weights.mean()),
        "positive_prior_min": float(positive_weights.min()),
        "positive_prior_max": float(positive_weights.max()),
        "estimates": estimates,
        "component_means": {
            "test_query_random": random_component[["mini", "e5"]].mean().to_dict(),
            "train_positive": train.loc[
                train["negative_type"].eq("positive"), ["mini", "e5"]
            ].mean().to_dict(),
            "train_semantic_hard": train.loc[
                train["negative_type"].eq("semantic_hard"), ["mini", "e5"]
            ].mean().to_dict(),
            "test_candidates": test[["mini", "e5"]].mean().to_dict(),
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
