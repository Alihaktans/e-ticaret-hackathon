from __future__ import annotations

"""Robust cross-fit final using only transferable frozen-embedding features."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "experiments"
SUBMISSIONS = ROOT / "submissions"
FEATURES = [
    "e5_score",
    "mini_score",
    "score_diff_e5_minus_mini",
    "score_product",
    "score_mean_dual",
]
SEEDS = [42, 2026, 777]
TARGET_PRIOR = 0.4249825984122894
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.005), 3)
HIST_BINS = np.linspace(0.0, 1.0, 501)
CHUNK = 250_000


def weighted_macro(y: np.ndarray, pred: np.ndarray) -> float:
    observed = float(y.mean())
    weights = np.where(y == 1, TARGET_PRIOR / observed, (1 - TARGET_PRIOR) / (1 - observed))
    return float(f1_score(y, pred, average="macro", labels=[0, 1], sample_weight=weights))


def hist(values: np.ndarray) -> np.ndarray:
    h = np.histogram(values, bins=HIST_BINS)[0].astype(np.float64)
    return h / h.sum()


def macro_from_rates(prior: float, tpr: float, fpr: float) -> float:
    tp, fn = prior * tpr, prior * (1 - tpr)
    fp, tn = (1 - prior) * fpr, (1 - prior) * (1 - fpr)
    f1p = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    f1n = 2 * tn / max(2 * tn + fp + fn, 1e-12)
    return float((f1p + f1n) / 2)


def mixture(positive_p: np.ndarray, random_p: np.ndarray, test_p: np.ndarray) -> dict:
    neg_h, pos_h, target_h = hist(random_p), hist(positive_p), hist(test_p)
    matrix = np.stack([neg_h, pos_h], axis=1)
    weights, _ = nnls(
        np.vstack([matrix, np.ones((1, 2)) * 10]), np.r_[target_h, 10.0]
    )
    weights /= weights.sum()
    prior = float(weights[1])
    neg_tail = np.cumsum(neg_h[::-1])[::-1]
    pos_tail = np.cumsum(pos_h[::-1])[::-1]
    candidates = []
    for i in range(len(neg_h)):
        score = macro_from_rates(prior, float(pos_tail[i]), float(neg_tail[i]))
        candidates.append((score, float(HIST_BINS[i]), float(pos_tail[i]), float(neg_tail[i])))
    score, threshold, tpr, fpr = max(candidates)
    return {
        "prior": prior,
        "expected_macro_f1": score,
        "threshold": threshold,
        "tpr": tpr,
        "fpr": fpr,
        "pred_positive_ratio": prior * tpr + (1 - prior) * fpr,
        "rmse": float(np.sqrt(np.mean((matrix @ weights - target_h) ** 2))),
    }


def predict(model: HistGradientBoostingClassifier, frame: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(frame), np.float32)
    for start in range(0, len(frame), CHUNK):
        end = min(start + CHUNK, len(frame))
        out[start:end] = model.predict_proba(frame.iloc[start:end][FEATURES])[:, 1]
    return out


def write_submission(ids: pd.Series, pred: np.ndarray, path: Path) -> None:
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv", usecols=["id"])
    if len(ids) != len(sample) or not ids.astype(str).reset_index(drop=True).equals(sample["id"].astype(str)):
        raise ValueError("ID mismatch")
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(path, index=False)


def main() -> None:
    train = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_train_frame.parquet",
        columns=["term_id", "label", "negative_type"] + FEATURES,
    )
    train = train[train["negative_type"].isin(["positive", "easy_random"])].copy()
    test = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_test_frame.parquet", columns=["id", "term_id"] + FEATURES
    )
    random_test = pd.read_parquet(
        PROC / "test_query_random_feature_frame.parquet", columns=FEATURES
    )
    terms = np.array(sorted(train["term_id"].astype(str).unique()))

    fold_reports, fold_test_p, fold_binary = [], [], []
    shared_rows = []
    for fold, seed in enumerate(SEEDS, start=1):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(terms, groups=terms))
        tr_terms, va_terms = set(terms[a]), set(terms[b])
        tr, va = train[train["term_id"].isin(tr_terms)], train[train["term_id"].isin(va_terms)]
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=.08,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=3,
            early_stopping=False,
            random_state=seed,
        ).fit(tr[FEATURES], tr["label"])
        valid_p = model.predict_proba(va[FEATURES])[:, 1]
        random_p = model.predict_proba(random_test[FEATURES])[:, 1]
        test_p = predict(model, test)
        y = va["label"].to_numpy(dtype=np.int8)
        local = []
        for threshold in THRESHOLDS:
            local.append((weighted_macro(y, valid_p >= threshold), float(threshold)))
        cv_score, cv_threshold = max(local)
        for score, threshold in local:
            shared_rows.append({"fold": fold, "threshold": threshold, "macro_f1": score})
        mix = mixture(valid_p[y == 1], random_p, test_p)
        report = {
            "fold": fold,
            "seed": seed,
            "auc": float(roc_auc_score(y, valid_p)),
            "cv_best_macro_f1": cv_score,
            "cv_best_threshold": cv_threshold,
            "test_mixture": mix,
        }
        fold_reports.append(report)
        fold_test_p.append(test_p)
        fold_binary.append(test_p >= mix["threshold"])
        print(f"Fold {fold}: {report}")

    shared = pd.DataFrame(shared_rows).groupby("threshold").macro_f1.agg(["mean", "min", "std"])
    shared = shared.sort_values(["mean", "min"], ascending=False)
    ensemble = np.mean(np.stack(fold_test_p), axis=0)
    majority = (np.sum(np.stack(fold_binary), axis=0) >= 2).astype(np.int8)
    mean_prior = float(np.mean([x["test_mixture"]["prior"] for x in fold_reports]))
    prior_cut = float(np.quantile(ensemble, 1 - mean_prior))
    prior_pred = (ensemble >= prior_cut).astype(np.int8)

    majority_path = SUBMISSIONS / "FINAL_CANDIDATE_v10_embedding_pu_crossfit_majority.csv"
    prior_path = SUBMISSIONS / "CANDIDATE_v10_embedding_pu_prior.csv"
    write_submission(test["id"], majority, majority_path)
    write_submission(test["id"], prior_pred, prior_path)
    pd.DataFrame({"id": test["id"], "term_id": test["term_id"], "proba": ensemble}).to_parquet(
        PROC / "embedding_pu_crossfit_v10_test_proba.parquet", index=False
    )

    expected = np.array([x["test_mixture"]["expected_macro_f1"] for x in fold_reports])
    priors = np.array([x["test_mixture"]["prior"] for x in fold_reports])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURES,
        "cv_shared_best": {"threshold": float(shared.index[0]), **shared.iloc[0].to_dict()},
        "folds": fold_reports,
        "test_estimated_macro_f1_mean": float(expected.mean()),
        "test_estimated_macro_f1_min": float(expected.min()),
        "test_estimated_macro_f1_max": float(expected.max()),
        "test_prior_mean": float(priors.mean()),
        "test_prior_min": float(priors.min()),
        "test_prior_max": float(priors.max()),
        "majority_positive_ratio": float(majority.mean()),
        "prior_candidate_positive_ratio": float(prior_pred.mean()),
        "candidate_agreement": float((majority == prior_pred).mean()),
        "recommended_candidate": str(majority_path),
        "alternative_candidate": str(prior_path),
        "submission_status": "generated_only_not_submitted",
        "overfit_guard": (
            "No lexical, category, brand, candidate-rank, or candidate-count features; "
            "term-grouped CV and test-query random negative component."
        ),
    }
    out = REPORTS / "embedding_pu_crossfit_v10_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
