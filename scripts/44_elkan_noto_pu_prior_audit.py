from __future__ import annotations

"""Estimate test prevalence by treating real test candidates as unlabeled.

This is a diagnostic only. It trains a labeled-positive vs unlabeled classifier,
estimates the labeling propensity c on held-out train-positive terms, and reports
the Elkan-Noto posterior/prevalence estimate. No submission is created.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "experiments" / "elkan_noto_pu_prior_audit.json"
V11 = [
    "e5_score", "mini_score", "score_diff_e5_minus_mini", "score_product", "score_mean_dual",
    "qc_overlap_count", "qc_overlap_query_ratio", "qc_overlap_category_ratio",
    "qt_overlap_count", "qt_overlap_query_ratio", "qt_overlap_title_ratio",
]
FUZZY = [
    "qt_fuzz_ratio", "qt_fuzz_partial", "qt_fuzz_token_set",
    "qc_fuzz_ratio", "qc_fuzz_partial", "qc_fuzz_token_set",
]
FEATURES = V11 + FUZZY
SEEDS = [42, 2026, 777]
N_UNLABELED_TRAIN = 250_000
CHUNK = 250_000


def predict_chunks(model, x: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(x), np.float32)
    for start in range(0, len(x), CHUNK):
        end = min(start + CHUNK, len(x))
        out[start:end] = model.predict_proba(x.iloc[start:end][FEATURES])[:, 1]
    return out


def main() -> None:
    train = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_train_frame.parquet",
        columns=["id", "term_id", "label"] + V11,
    )
    train = train[train["label"].eq(1)].copy()
    train = train.merge(pd.read_parquet(PROC / "pu_train_fuzzy_features.parquet"), on="id")
    test = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_test_frame.parquet", columns=["id"] + V11
    )
    test = test.merge(pd.read_parquet(PROC / "test_fuzzy_features.parquet"), on="id")
    unlabeled_fit = test.sample(n=N_UNLABELED_TRAIN, random_state=2026)
    terms = np.array(sorted(train["term_id"].astype(str).unique()))
    reports = []

    for family in ["logistic", "hgb"]:
        for fold, seed in enumerate(SEEDS, 1):
            a, b = next(
                GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
                .split(terms, groups=terms)
            )
            fit_pos = train[train["term_id"].isin(set(terms[a]))]
            heldout_pos = train[train["term_id"].isin(set(terms[b]))]
            x_fit = pd.concat([fit_pos[FEATURES], unlabeled_fit[FEATURES]], ignore_index=True)
            y_fit = np.r_[np.ones(len(fit_pos), np.int8), np.zeros(len(unlabeled_fit), np.int8)]
            if family == "logistic":
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=500, random_state=seed),
                )
            else:
                model = HistGradientBoostingClassifier(
                    max_iter=250, learning_rate=.06, max_leaf_nodes=31,
                    min_samples_leaf=50, l2_regularization=5,
                    early_stopping=False, random_state=seed,
                )
            model.fit(x_fit, y_fit)
            heldout_score = model.predict_proba(heldout_pos[FEATURES])[:, 1]
            test_score = predict_chunks(model, test)
            # Robust versions are reported because raw probability calibration is imperfect.
            c_mean = float(np.mean(heldout_score))
            c_median = float(np.median(heldout_score))
            posterior = np.clip(test_score / max(c_mean, 1e-6), 0, 1)
            report = {
                "family": family,
                "fold": fold,
                "seed": seed,
                "fit_positive_rows": int(len(fit_pos)),
                "heldout_positive_rows": int(len(heldout_pos)),
                "unlabeled_fit_rows": int(len(unlabeled_fit)),
                "c_mean": c_mean,
                "c_median": c_median,
                "test_labeled_score_mean": float(np.mean(test_score)),
                "prevalence_mean_posterior": float(np.mean(posterior)),
                "posterior_quantiles": {
                    str(q): float(np.quantile(posterior, q))
                    for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
                },
            }
            reports.append(report)
            print(report)

    by_family = {}
    for family in ["logistic", "hgb"]:
        values = np.array(
            [x["prevalence_mean_posterior"] for x in reports if x["family"] == family]
        )
        by_family[family] = {
            "mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "std": float(values.std()),
        }
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Elkan-Noto labeled-positive vs real-test-unlabeled",
        "features": FEATURES,
        "folds": reports,
        "prevalence_summary": by_family,
        "warning": (
            "SCAR and probability-calibration assumptions may not hold because train and test "
            "terms are disjoint. Treat this as a prevalence diagnostic, not ground truth."
        ),
    }
    REPORT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
