from __future__ import annotations

"""Cross-fit a transfer-screened overlap model and blend it with V10."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports" / "experiments"
BASE = ["e5_score", "mini_score", "score_diff_e5_minus_mini", "score_product", "score_mean_dual"]
SAFE_OVERLAP = [
    "qc_overlap_count",
    "qc_overlap_query_ratio",
    "qc_overlap_category_ratio",
    "qt_overlap_count",
    "qt_overlap_query_ratio",
    "qt_overlap_title_ratio",
]
EXPANDED = BASE + SAFE_OVERLAP
SEEDS = [42, 2026, 777]
TARGET_PRIOR = 0.4249825984122894
THRESHOLDS = np.round(np.arange(0.30, 0.801, 0.005), 3)
EXPANDED_WEIGHTS = [0.50, 0.70, 0.85, 1.00]
CHUNK = 250_000


def model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=.08,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=3,
        early_stopping=False,
        random_state=seed,
    )


def predict_chunks(estimator, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    out = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), CHUNK):
        end = min(start + CHUNK, len(frame))
        out[start:end] = estimator.predict_proba(frame.iloc[start:end][features])[:, 1]
    return out


def target_weights(y: np.ndarray) -> np.ndarray:
    observed = float(y.mean())
    return np.where(y == 1, TARGET_PRIOR / observed, (1 - TARGET_PRIOR) / (1 - observed))


def write_submission(ids: pd.Series, pred: np.ndarray, path: Path) -> None:
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv", usecols=["id"])
    if len(ids) != len(sample) or not ids.astype(str).reset_index(drop=True).equals(sample["id"].astype(str)):
        raise ValueError("Submission ID mismatch")
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(path, index=False)


def main() -> None:
    train = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_train_frame.parquet",
        columns=["term_id", "label", "negative_type"] + EXPANDED,
    )
    train = train[train["negative_type"].isin(["positive", "easy_random"])].copy()
    test = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_test_frame.parquet",
        columns=["id", "term_id"] + EXPANDED,
    )
    terms = np.array(sorted(train["term_id"].astype(str).unique()))
    folds = []
    test_base, test_expanded = [], []

    for fold, seed in enumerate(SEEDS, start=1):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(terms, groups=terms))
        tr_terms, va_terms = set(terms[a]), set(terms[b])
        tr, va = train[train["term_id"].isin(tr_terms)], train[train["term_id"].isin(va_terms)]
        base_model = model(seed).fit(tr[BASE], tr["label"])
        expanded_model = model(seed).fit(tr[EXPANDED], tr["label"])
        folds.append(
            {
                "fold": fold,
                "y": va["label"].to_numpy(dtype=np.int8),
                "base": base_model.predict_proba(va[BASE])[:, 1],
                "expanded": expanded_model.predict_proba(va[EXPANDED])[:, 1],
            }
        )
        test_base.append(predict_chunks(base_model, test, BASE))
        test_expanded.append(predict_chunks(expanded_model, test, EXPANDED))
        print(f"Completed fold {fold}")

    rows = []
    for expanded_weight in EXPANDED_WEIGHTS:
        for threshold in THRESHOLDS:
            fold_scores, fold_ratios = [], []
            for fold in folds:
                proba = expanded_weight * fold["expanded"] + (1 - expanded_weight) * fold["base"]
                pred = proba >= threshold
                weights = target_weights(fold["y"])
                fold_scores.append(
                    f1_score(fold["y"], pred, average="macro", labels=[0, 1], sample_weight=weights)
                )
                fold_ratios.append(float(np.average(pred, weights=weights)))
            rows.append(
                {
                    "expanded_weight": expanded_weight,
                    "threshold": float(threshold),
                    "mean_macro_f1": float(np.mean(fold_scores)),
                    "min_macro_f1": float(np.min(fold_scores)),
                    "std_macro_f1": float(np.std(fold_scores)),
                    "mean_pred_positive_ratio": float(np.mean(fold_ratios)),
                    "fold_scores": fold_scores,
                    "fold_positive_ratios": fold_ratios,
                }
            )
    results = pd.DataFrame(rows).sort_values(
        ["mean_macro_f1", "min_macro_f1"], ascending=False
    )
    best = results.iloc[0].to_dict()
    ew = float(best["expanded_weight"])
    threshold = float(best["threshold"])
    target_ratio = float(best["mean_pred_positive_ratio"])

    base_test = np.mean(np.stack(test_base), axis=0)
    expanded_test = np.mean(np.stack(test_expanded), axis=0)
    blended_test = ew * expanded_test + (1 - ew) * base_test
    threshold_pred = (blended_test >= threshold).astype(np.int8)
    ratio_cutoff = float(np.quantile(blended_test, 1 - target_ratio))
    ratio_pred = (blended_test >= ratio_cutoff).astype(np.int8)

    threshold_path = SUBMISSIONS / "CANDIDATE_v11_safe_overlap_shared_threshold.csv"
    ratio_path = SUBMISSIONS / "FINAL_CANDIDATE_v11_safe_overlap_cv_ratio.csv"
    write_submission(test["id"], threshold_pred, threshold_path)
    write_submission(test["id"], ratio_pred, ratio_path)
    pd.DataFrame({"id": test["id"], "term_id": test["term_id"], "proba": blended_test}).to_parquet(
        PROC / "v11_safe_overlap_crossfit_test_proba.parquet", index=False
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_features": EXPANDED,
        "explicitly_excluded_shifted_features": [
            "query_in_title", "title_in_query", "brand_in_query",
            "query/title/category lengths", "all categorical IDs", "all rank/count features",
        ],
        "best_cv": best,
        "test_shared_threshold_positive_ratio": float(threshold_pred.mean()),
        "test_cv_ratio_calibrated_positive_ratio": float(ratio_pred.mean()),
        "cv_ratio_cutoff_on_test": ratio_cutoff,
        "candidate_agreement": float((threshold_pred == ratio_pred).mean()),
        "recommended_candidate": str(ratio_path),
        "alternative_candidate": str(threshold_path),
        "submission_status": "generated_only_not_submitted",
        "warning": "CV-ratio calibration relies on the independently estimated 42.5% test prior.",
    }
    out = REPORTS / "v11_safe_overlap_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
