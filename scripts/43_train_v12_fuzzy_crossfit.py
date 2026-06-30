from __future__ import annotations

"""Cross-fit transfer-safe fuzzy model and calibrate by shared CV positive rate."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
PROC, SUB, REP = ROOT / "data" / "processed", ROOT / "submissions", ROOT / "reports" / "experiments"
V11 = [
    "e5_score", "mini_score", "score_diff_e5_minus_mini", "score_product", "score_mean_dual",
    "qc_overlap_count", "qc_overlap_query_ratio", "qc_overlap_category_ratio",
    "qt_overlap_count", "qt_overlap_query_ratio", "qt_overlap_title_ratio",
]
FUZZY = [
    "qt_fuzz_ratio", "qt_fuzz_partial", "qt_fuzz_token_set",
    "qc_fuzz_ratio", "qc_fuzz_partial", "qc_fuzz_token_set",
]
V12 = V11 + FUZZY
SEEDS = [42, 2026, 777]
TARGET_PRIOR = 0.4249825984122894
THRESHOLDS = np.round(np.arange(.30, .801, .005), 3)
V12_WEIGHTS = [.50, .70, .85, 1.00]
CHUNK = 250_000


def make_model(seed: int):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=.08, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=3, early_stopping=False, random_state=seed,
    )


def predict(model, frame, features):
    out = np.empty(len(frame), np.float32)
    for start in range(0, len(frame), CHUNK):
        end = min(start + CHUNK, len(frame))
        out[start:end] = model.predict_proba(frame.iloc[start:end][features])[:, 1]
    return out


def weights(y):
    observed = y.mean()
    return np.where(y == 1, TARGET_PRIOR / observed, (1 - TARGET_PRIOR) / (1 - observed))


def write(ids, prediction, path):
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv", usecols=["id"])
    if len(ids) != len(sample) or not ids.astype(str).reset_index(drop=True).equals(sample.id.astype(str)):
        raise ValueError("ID mismatch")
    pd.DataFrame({"id": ids, "prediction": prediction.astype(np.int8)}).to_csv(path, index=False)


def main():
    train = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_train_frame.parquet",
        columns=["id", "term_id", "label", "negative_type"] + V11,
    )
    train = train[train.negative_type.isin(["positive", "easy_random"])].copy()
    train_fuzzy = pd.read_parquet(PROC / "pu_train_fuzzy_features.parquet")
    train = train.merge(train_fuzzy, on="id", validate="one_to_one")
    test = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_test_frame.parquet", columns=["id", "term_id"] + V11
    )
    test = test.merge(pd.read_parquet(PROC / "test_fuzzy_features.parquet"), on="id", validate="one_to_one")
    terms = np.array(sorted(train.term_id.astype(str).unique()))

    folds, test_v11, test_v12 = [], [], []
    for fold, seed in enumerate(SEEDS, 1):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(terms, groups=terms))
        tr, va = train[train.term_id.isin(set(terms[a]))], train[train.term_id.isin(set(terms[b]))]
        m11, m12 = make_model(seed).fit(tr[V11], tr.label), make_model(seed).fit(tr[V12], tr.label)
        folds.append({
            "fold": fold, "y": va.label.to_numpy(np.int8),
            "v11": m11.predict_proba(va[V11])[:, 1],
            "v12": m12.predict_proba(va[V12])[:, 1],
        })
        test_v11.append(predict(m11, test, V11)); test_v12.append(predict(m12, test, V12))
        print(f"Completed fold {fold}")

    rows = []
    for v12_weight in V12_WEIGHTS:
        for threshold in THRESHOLDS:
            scores, ratios = [], []
            for fold in folds:
                p = v12_weight * fold["v12"] + (1 - v12_weight) * fold["v11"]
                pred, w = p >= threshold, weights(fold["y"])
                scores.append(f1_score(fold["y"], pred, average="macro", sample_weight=w))
                ratios.append(float(np.average(pred, weights=w)))
            rows.append({
                "v12_weight": v12_weight, "threshold": float(threshold),
                "mean_macro_f1": float(np.mean(scores)), "min_macro_f1": float(np.min(scores)),
                "std_macro_f1": float(np.std(scores)), "mean_pred_positive_ratio": float(np.mean(ratios)),
                "fold_scores": scores, "fold_positive_ratios": ratios,
            })
    results = pd.DataFrame(rows).sort_values(["mean_macro_f1", "min_macro_f1"], ascending=False)
    best = results.iloc[0].to_dict(); bw, bt = float(best["v12_weight"]), float(best["threshold"])
    test_score = bw * np.mean(np.stack(test_v12), axis=0) + (1 - bw) * np.mean(np.stack(test_v11), axis=0)
    threshold_pred = (test_score >= bt).astype(np.int8)
    target_ratio = float(best["mean_pred_positive_ratio"])
    cutoff = float(np.quantile(test_score, 1 - target_ratio)); calibrated_pred = (test_score >= cutoff).astype(np.int8)
    threshold_path = SUB / "CANDIDATE_v12_fuzzy_shared_threshold.csv"
    final_path = SUB / "FINAL_CANDIDATE_v12_fuzzy_cv_ratio.csv"
    write(test.id, threshold_pred, threshold_path); write(test.id, calibrated_pred, final_path)
    pd.DataFrame({"id": test.id, "term_id": test.term_id, "proba": test_score}).to_parquet(
        PROC / "v12_fuzzy_crossfit_test_proba.parquet", index=False
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "best_cv": best,
        "test_shared_threshold_positive_ratio": float(threshold_pred.mean()),
        "test_cv_ratio_positive_ratio": float(calibrated_pred.mean()), "test_cv_ratio_cutoff": cutoff,
        "candidate_agreement": float((threshold_pred == calibrated_pred).mean()),
        "recommended_candidate": str(final_path), "alternative_candidate": str(threshold_path),
        "submission_status": "generated_only_not_submitted",
        "overfit_guard": "Only embedding, soft token overlap, and transfer-screened fuzzy similarities.",
    }
    (REP / "v12_fuzzy_crossfit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
