from __future__ import annotations

"""Train the leakage-resistant PU model and create offline review candidates."""

import json
import runpy
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports" / "experiments"

cv_module = runpy.run_path(str(ROOT / "scripts" / "35_train_pu_random_negative_cv.py"))
FEATURES: list[str] = cv_module["FEATURES"]
TARGET_POS_PRIOR: float = cv_module["TARGET_POS_PRIOR"]
SHARED_CV_THRESHOLD = 0.55
CHUNK_SIZE = 250_000


def validate_submission(ids: pd.Series, prediction: np.ndarray) -> None:
    sample_ids = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv", usecols=["id"])["id"]
    if len(ids) != len(sample_ids):
        raise ValueError("Submission row count mismatch")
    if not ids.astype(str).reset_index(drop=True).equals(sample_ids.astype(str)):
        raise ValueError("Submission ID order mismatch")
    if not set(np.unique(prediction)).issubset({0, 1}):
        raise ValueError("Submission contains non-binary predictions")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(PROC / "v6_dual_e5_minilm_train_frame.parquet")
    train = train[train["negative_type"].isin(["positive", "easy_random"])].copy()
    train[FEATURES] = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=2026,
    )
    model.fit(train[FEATURES], train["label"].astype(int))
    model_path = MODELS / "pu_random_negative_hgb_v1.joblib"
    joblib.dump({"model": model, "features": FEATURES}, model_path)

    test = pd.read_parquet(
        PROC / "v6_dual_e5_minilm_test_frame.parquet",
        columns=["id", "term_id"] + FEATURES,
    )
    test[FEATURES] = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = np.empty(len(test), dtype=np.float32)
    for start in range(0, len(test), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(test))
        proba[start:end] = model.predict_proba(test.iloc[start:end][FEATURES])[:, 1]

    score_path = PROC / "pu_random_negative_hgb_v1_test_proba.parquet"
    pd.DataFrame(
        {"id": test["id"], "term_id": test["term_id"], "proba": proba}
    ).to_parquet(score_path, index=False)

    threshold_pred = (proba >= SHARED_CV_THRESHOLD).astype(np.int8)
    prior_cutoff = float(np.quantile(proba, 1.0 - TARGET_POS_PRIOR))
    prior_pred = (proba >= prior_cutoff).astype(np.int8)
    validate_submission(test["id"], threshold_pred)
    validate_submission(test["id"], prior_pred)

    threshold_path = SUBMISSIONS / "CANDIDATE_v8_pu_random_hgb_cv_threshold_0p550.csv"
    prior_path = SUBMISSIONS / "CANDIDATE_v8_pu_random_hgb_prior_0p425.csv"
    pd.DataFrame({"id": test["id"], "prediction": threshold_pred}).to_csv(
        threshold_path, index=False
    )
    pd.DataFrame({"id": test["id"], "prediction": prior_pred}).to_csv(prior_path, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "HistGradientBoostingClassifier",
        "training_rows": int(len(train)),
        "positive_training_rows": int(train["label"].sum()),
        "features": FEATURES,
        "cv_shared_threshold": SHARED_CV_THRESHOLD,
        "cv_mean_macro_f1_at_shared_threshold": 0.965568,
        "cv_min_macro_f1_at_shared_threshold": 0.963920,
        "target_positive_prior": TARGET_POS_PRIOR,
        "test": {
            "rows": int(len(test)),
            "cv_threshold_positive_ratio": float(threshold_pred.mean()),
            "prior_cutoff": prior_cutoff,
            "prior_constrained_positive_ratio": float(prior_pred.mean()),
            "candidate_agreement": float((threshold_pred == prior_pred).mean()),
        },
        "files": {
            "model": str(model_path),
            "scores": str(score_path),
            "cv_threshold_candidate": str(threshold_path),
            "prior_constrained_candidate": str(prior_path),
        },
        "submission_status": "generated_only_not_submitted",
        "warning": (
            "Local score is conditional on the independently estimated random-negative/positive "
            "test mixture; it is not a guaranteed leaderboard score."
        ),
    }
    report_path = REPORTS / "pu_random_negative_hgb_v1_final_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
