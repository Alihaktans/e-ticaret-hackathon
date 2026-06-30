from __future__ import annotations

"""Cross-fit PU models and estimate test Macro-F1 from component score mixtures."""

import json
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "experiments"
SUBMISSIONS = ROOT / "submissions"
cfg = runpy.run_path(str(ROOT / "scripts" / "35_train_pu_random_negative_cv.py"))
FEATURES: list[str] = cfg["FEATURES"]
SEEDS = [42, 2026, 777]
CHUNK = 250_000
HIST_BINS = np.linspace(0.0, 1.0, 501)


def macro_from_rates(prior: float, tpr: float, fpr: float) -> float:
    tp = prior * tpr
    fn = prior * (1.0 - tpr)
    fp = (1.0 - prior) * fpr
    tn = (1.0 - prior) * (1.0 - fpr)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    f1_neg = 2 * tn / max(2 * tn + fp + fn, 1e-12)
    return float((f1_pos + f1_neg) / 2.0)


def normalized_hist(values: np.ndarray) -> np.ndarray:
    h = np.histogram(values, bins=HIST_BINS)[0].astype(np.float64)
    return h / h.sum()


def mixture_estimate(valid_y: np.ndarray, valid_p: np.ndarray, random_p: np.ndarray, test_p: np.ndarray) -> dict:
    neg = normalized_hist(random_p)
    pos = normalized_hist(valid_p[valid_y == 1])
    target = normalized_hist(test_p)
    matrix = np.stack([neg, pos], axis=1)
    weights, _ = nnls(
        np.vstack([matrix, np.ones((1, 2)) * 10.0]),
        np.concatenate([target, [10.0]]),
    )
    weights /= weights.sum()
    prior = float(weights[1])

    neg_tail = np.cumsum(neg[::-1])[::-1]
    pos_tail = np.cumsum(pos[::-1])[::-1]
    rows = []
    for idx in range(len(neg)):
        threshold = float(HIST_BINS[idx])
        score = macro_from_rates(prior, float(pos_tail[idx]), float(neg_tail[idx]))
        rows.append((score, threshold, float(pos_tail[idx]), float(neg_tail[idx])))
    best = max(rows)
    predicted_ratio = prior * best[2] + (1.0 - prior) * best[3]
    return {
        "estimated_positive_prior": prior,
        "estimated_best_macro_f1": float(best[0]),
        "best_threshold": float(best[1]),
        "estimated_tpr": float(best[2]),
        "estimated_fpr": float(best[3]),
        "estimated_pred_positive_ratio": float(predicted_ratio),
        "histogram_rmse": float(np.sqrt(np.mean((matrix @ weights - target) ** 2))),
    }


def predict_chunks(model: HistGradientBoostingClassifier, frame: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), CHUNK):
        end = min(start + CHUNK, len(frame))
        out[start:end] = model.predict_proba(frame.iloc[start:end][FEATURES])[:, 1]
    return out


def validate_and_write(ids: pd.Series, pred: np.ndarray, path: Path) -> None:
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv", usecols=["id"])
    if len(ids) != len(sample) or not ids.astype(str).reset_index(drop=True).equals(sample["id"].astype(str)):
        raise ValueError("Submission ID mismatch")
    if not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("Non-binary prediction")
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(path, index=False)


def main() -> None:
    train = pd.read_parquet(PROC / "v6_dual_e5_minilm_train_frame.parquet")
    train = train[train["negative_type"].isin(["positive", "easy_random"])].copy()
    train[FEATURES] = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_test_frame.parquet", columns=["id", "term_id"] + FEATURES
    )
    test[FEATURES] = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    random_test_query = pd.read_parquet(
        PROC / "test_query_random_feature_frame.parquet", columns=FEATURES
    )
    random_test_query[FEATURES] = (
        random_test_query[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    terms = np.array(sorted(train["term_id"].astype(str).unique()))

    test_probas = []
    fold_binary = []
    fold_reports = []
    for fold, seed in enumerate(SEEDS, start=1):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, valid_idx = next(splitter.split(terms, groups=terms))
        train_terms = set(terms[train_idx])
        valid_terms = set(terms[valid_idx])
        tr = train[train["term_id"].isin(train_terms)]
        va = train[train["term_id"].isin(valid_terms)]
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=3.0,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(tr[FEATURES], tr["label"].astype(int))
        valid_p = model.predict_proba(va[FEATURES])[:, 1]
        random_p = model.predict_proba(random_test_query[FEATURES])[:, 1]
        test_p = predict_chunks(model, test)
        estimate = mixture_estimate(
            va["label"].to_numpy(dtype=np.int8), valid_p, random_p, test_p
        )
        estimate.update({"fold": fold, "seed": seed, "valid_rows": int(len(va))})
        fold_reports.append(estimate)
        test_probas.append(test_p)
        fold_binary.append(test_p >= estimate["best_threshold"])
        print(f"Fold {fold}: {estimate}")

    ensemble = np.mean(np.stack(test_probas), axis=0)
    majority = (np.sum(np.stack(fold_binary), axis=0) >= 2).astype(np.int8)
    mean_prior = float(np.mean([x["estimated_positive_prior"] for x in fold_reports]))
    prior_cutoff = float(np.quantile(ensemble, 1.0 - mean_prior))
    prior_pred = (ensemble >= prior_cutoff).astype(np.int8)

    majority_path = SUBMISSIONS / "CANDIDATE_v9_pu_crossfit_majority.csv"
    prior_path = SUBMISSIONS / "CANDIDATE_v9_pu_crossfit_prior.csv"
    validate_and_write(test["id"], majority, majority_path)
    validate_and_write(test["id"], prior_pred, prior_path)
    pd.DataFrame({"id": test["id"], "term_id": test["term_id"], "proba": ensemble}).to_parquet(
        PROC / "pu_crossfit_hgb_v1_test_proba.parquet", index=False
    )

    scores = np.array([x["estimated_best_macro_f1"] for x in fold_reports])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "folds": fold_reports,
        "estimated_macro_f1_mean": float(scores.mean()),
        "estimated_macro_f1_min": float(scores.min()),
        "estimated_macro_f1_max": float(scores.max()),
        "estimated_prior_mean": mean_prior,
        "majority_positive_ratio": float(majority.mean()),
        "prior_candidate_positive_ratio": float(prior_pred.mean()),
        "candidate_agreement": float((majority == prior_pred).mean()),
        "recommended_candidate": str(majority_path),
        "alternative_candidate": str(prior_path),
        "submission_status": "generated_only_not_submitted",
        "warning": "Estimate depends on positive/random-negative component transfer to test.",
    }
    out = REPORTS / "pu_crossfit_hgb_v1_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
