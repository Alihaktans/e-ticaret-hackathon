from __future__ import annotations

"""Term-grouped PU validation using domain-supported random negatives only.

The previous CV used mined hard negatives whose distribution does not match the
test candidates. This experiment deliberately excludes semantic-hard negatives
and excludes candidate-pool rank features that change with bag construction.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "data" / "processed" / "v6_dual_e5_minilm_train_frame.parquet"
OUT_DIR = ROOT / "reports" / "experiments"

TARGET_POS_PRIOR = 0.4249825984122894
FOLD_SEEDS = [42, 2026, 777]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.005), 3)

NUMERIC_FEATURES = [
    "e5_score",
    "mini_score",
    "score_diff_e5_minus_mini",
    "score_product",
    "score_mean_dual",
    "query_char_len",
    "title_char_len",
    "category_char_len",
    "query_token_count",
    "title_token_count",
    "category_token_count",
    "qt_overlap_count",
    "qt_overlap_query_ratio",
    "qt_overlap_title_ratio",
    "qc_overlap_count",
    "qc_overlap_query_ratio",
    "qc_overlap_category_ratio",
    "query_in_title",
    "title_in_query",
    "brand_in_query",
]
FEATURES = NUMERIC_FEATURES


def target_prior_weights(y: np.ndarray) -> np.ndarray:
    observed_prior = float(y.mean())
    pos_weight = TARGET_POS_PRIOR / observed_prior
    neg_weight = (1.0 - TARGET_POS_PRIOR) / (1.0 - observed_prior)
    return np.where(y == 1, pos_weight, neg_weight)


def evaluate(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    weights = target_prior_weights(y)
    rows = []
    for threshold in THRESHOLDS:
        pred = (proba >= threshold).astype(np.int8)
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(
                    f1_score(y, pred, average="macro", labels=[0, 1], sample_weight=weights)
                ),
                "positive_f1": float(
                    f1_score(y, pred, pos_label=1, sample_weight=weights)
                ),
                "negative_f1": float(
                    f1_score(y, pred, pos_label=0, sample_weight=weights)
                ),
                "pred_positive_ratio_weighted": float(np.average(pred, weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(FRAME)
    frame = frame[frame["negative_type"].isin(["positive", "easy_random"])].copy()
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    terms = np.array(sorted(frame["term_id"].astype(str).unique()))
    fold_rows = []
    fold_meta = []

    for fold, seed in enumerate(FOLD_SEEDS, start=1):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, valid_idx = next(splitter.split(terms, groups=terms))
        train_terms = set(terms[train_idx])
        valid_terms = set(terms[valid_idx])
        train = frame[frame["term_id"].isin(train_terms)].copy()
        valid = frame[frame["term_id"].isin(valid_terms)].copy()

        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=3.0,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
            random_state=seed,
        )
        model.fit(train[FEATURES], train["label"].astype(int))
        proba = model.predict_proba(valid[FEATURES])[:, 1]
        scores = evaluate(valid["label"].to_numpy(dtype=np.int8), proba)
        scores["fold"] = fold
        scores["seed"] = seed
        fold_rows.append(scores)
        best = scores.loc[scores["macro_f1"].idxmax()]
        fold_meta.append(
            {
                "fold": fold,
                "seed": seed,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "best_iteration": int(model.n_iter_),
                "auc": float(roc_auc_score(valid["label"], proba)),
                "best_threshold": float(best["threshold"]),
                "best_macro_f1": float(best["macro_f1"]),
            }
        )
        print(f"Fold {fold}: {fold_meta[-1]}")

    raw = pd.concat(fold_rows, ignore_index=True)
    summary = (
        raw.groupby("threshold", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            min_macro_f1=("macro_f1", "min"),
            mean_positive_f1=("positive_f1", "mean"),
            mean_negative_f1=("negative_f1", "mean"),
            mean_pred_positive_ratio=("pred_positive_ratio_weighted", "mean"),
        )
        .sort_values("mean_macro_f1", ascending=False)
    )
    raw_path = OUT_DIR / "pu_random_negative_cv_raw.csv"
    summary_path = OUT_DIR / "pu_random_negative_cv_summary.csv"
    report_path = OUT_DIR / "pu_random_negative_cv_report.json"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_positive_prior": TARGET_POS_PRIOR,
        "features": FEATURES,
        "excluded_features": "all per-term rank and candidate-distribution statistics",
        "folds": fold_meta,
        "best_shared_thresholds": summary.head(20).to_dict(orient="records"),
        "warning": (
            "The estimate is conditional on random negatives representing the dominant test "
            "negative component and train positives transferring to test terms."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nShared threshold summary:")
    print(summary.head(20).to_string(index=False))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
