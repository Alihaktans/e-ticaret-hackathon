from __future__ import annotations

"""Audit positive-unlabeled validation with test-aligned candidate bags.

The competition train data contains positives only. This script does not claim
that mined candidates are true negatives. It measures ranking stability under
several explicit contamination assumptions while keeping each validation bag
at exactly 100 pairs, matching the dominant test layout.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "reports" / "experiments"

TRAIN_SCORES = PROCESSED / "embedding_v4_train_scores.parquet"
CANDIDATES = PROCESSED / "v4_semantic_hard_negative_candidates.parquet"
TEST_SCORES = PROCESSED / "embedding_v1_minilm_topratio_008_scores.parquet"

TARGET_BAG_SIZE = 100
TOP_K_VALUES = list(range(1, 31))
CONTAMINATION_PER_TERM = [0, 1, 2, 3]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1]))


def load_aligned_frame() -> tuple[pd.DataFrame, dict]:
    scored = pd.read_parquet(
        TRAIN_SCORES,
        columns=["term_id", "item_id", "label", "negative_type", "embedding_score"],
    )
    positives = scored.loc[
        scored["label"].eq(1), ["term_id", "item_id", "embedding_score"]
    ].drop_duplicates(["term_id", "item_id"])
    positives["observed_label"] = 1
    positives["source"] = "observed_positive"

    candidates = pd.read_parquet(
        CANDIDATES,
        columns=["term_id", "item_id", "semantic_rank", "embedding_score"],
    ).sort_values(["term_id", "semantic_rank"])

    pos_counts = positives.groupby("term_id").size().rename("n_pos")
    eligible = pos_counts[(pos_counts > 0) & (pos_counts < TARGET_BAG_SIZE)]
    positives = positives[positives["term_id"].isin(eligible.index)].copy()
    candidates = candidates[candidates["term_id"].isin(eligible.index)].copy()
    candidates = candidates.merge(eligible, left_on="term_id", right_index=True, how="inner")
    candidates["keep_n"] = TARGET_BAG_SIZE - candidates["n_pos"]
    candidates["within_term_row"] = candidates.groupby("term_id").cumcount()
    candidates = candidates[candidates["within_term_row"] < candidates["keep_n"]].copy()
    candidates["observed_label"] = 0
    candidates["source"] = "unlabeled_retrieved"

    frame = pd.concat(
        [
            positives[["term_id", "item_id", "embedding_score", "observed_label", "source"]],
            candidates[["term_id", "item_id", "embedding_score", "observed_label", "source"]],
        ],
        ignore_index=True,
    )
    bag_sizes = frame.groupby("term_id").size()
    complete_terms = bag_sizes[bag_sizes.eq(TARGET_BAG_SIZE)].index
    frame = frame[frame["term_id"].isin(complete_terms)].copy()
    frame["rank"] = frame.groupby("term_id")["embedding_score"].rank(
        method="first", ascending=False
    ).astype(np.int16)

    meta = {
        "eligible_terms_before_complete_filter": int(len(eligible)),
        "complete_terms": int(len(complete_terms)),
        "rows": int(len(frame)),
        "bag_size": TARGET_BAG_SIZE,
        "observed_positive_ratio": float(frame["observed_label"].mean()),
    }
    return frame, meta


def apply_contamination(frame: pd.DataFrame, positives_per_term: int) -> np.ndarray:
    labels = frame["observed_label"].to_numpy(dtype=np.int8, copy=True)
    if positives_per_term == 0:
        return labels

    unlabeled = frame[frame["observed_label"].eq(0)].copy()
    assumed_positive_idx = (
        unlabeled.sort_values(["term_id", "embedding_score"], ascending=[True, False])
        .groupby("term_id")
        .head(positives_per_term)
        .index.to_numpy()
    )
    labels[assumed_positive_idx] = 1
    return labels


def evaluate_top_k(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    ranks = frame["rank"].to_numpy()
    observed = frame["observed_label"].to_numpy()

    for contamination in CONTAMINATION_PER_TERM:
        y_true = apply_contamination(frame, contamination)
        for k in TOP_K_VALUES:
            pred = (ranks <= k).astype(np.int8)
            rows.append(
                {
                    "assumed_unlabeled_positives_per_term": contamination,
                    "top_k": k,
                    "macro_f1": macro_f1(y_true, pred),
                    "positive_f1": float(f1_score(y_true, pred, pos_label=1)),
                    "negative_f1": float(f1_score(y_true, pred, pos_label=0)),
                    "pred_positive_ratio": float(pred.mean()),
                    "assumed_true_positive_ratio": float(y_true.mean()),
                    "observed_positive_recall": float(
                        pred[observed == 1].mean() if np.any(observed == 1) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def score_shift_summary(frame: pd.DataFrame) -> dict:
    test = pd.read_parquet(TEST_SCORES, columns=["score"])["score"]
    out: dict[str, dict] = {}
    groups = {
        "aligned_observed_positive": frame.loc[
            frame["observed_label"].eq(1), "embedding_score"
        ],
        "aligned_unlabeled": frame.loc[
            frame["observed_label"].eq(0), "embedding_score"
        ],
        "test_candidates": test,
    }
    for name, values in groups.items():
        q = values.quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        out[name] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "quantiles": {str(k): float(v) for k, v in q.items()},
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, meta = load_aligned_frame()
    results = evaluate_top_k(frame)
    shift = score_shift_summary(frame)

    csv_path = OUT_DIR / "pu_candidate_aligned_topk.csv"
    json_path = OUT_DIR / "pu_candidate_aligned_audit.json"
    results.to_csv(csv_path, index=False)

    best_by_assumption = (
        results.sort_values("macro_f1", ascending=False)
        .groupby("assumed_unlabeled_positives_per_term", as_index=False)
        .first()
        .sort_values("assumed_unlabeled_positives_per_term")
    )
    robust = (
        results.groupby("top_k", as_index=False)["macro_f1"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values(["min", "mean"], ascending=False)
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "warning": (
            "Unlabeled retrieved pairs are not verified negatives; scores are sensitivity "
            "analysis, not unbiased estimates of leaderboard Macro-F1."
        ),
        "meta": meta,
        "best_by_contamination_assumption": best_by_assumption.to_dict(orient="records"),
        "robust_top_k": robust.head(10).to_dict(orient="records"),
        "score_shift": shift,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print("\nBest top-k by contamination assumption:")
    print(best_by_assumption.to_string(index=False))
    print("\nRobust top-k across assumptions:")
    print(robust.head(10).to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
