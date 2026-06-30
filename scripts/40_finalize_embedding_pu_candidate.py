from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROBA = ROOT / "data" / "processed" / "embedding_pu_crossfit_v10_test_proba.parquet"
SAMPLE = ROOT / "data" / "raw" / "sample_submission.csv"
OUT = ROOT / "submissions" / "FINAL_CANDIDATE_v10_embedding_pu_shared_cv_threshold_0p570.csv"
REPORT = ROOT / "reports" / "experiments" / "final_candidate_v10_report.json"
THRESHOLD = 0.57


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    scores = pd.read_parquet(PROBA)
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    if len(scores) != len(sample):
        raise ValueError("Row count mismatch")
    if not scores["id"].astype(str).reset_index(drop=True).equals(sample["id"].astype(str)):
        raise ValueError("ID order mismatch")
    prediction = scores["proba"].ge(THRESHOLD).to_numpy(dtype=np.int8)
    pd.DataFrame({"id": scores["id"], "prediction": prediction}).to_csv(OUT, index=False)
    check = pd.read_csv(OUT)
    if len(check) != len(sample) or set(check["prediction"].unique()) - {0, 1}:
        raise ValueError("Written submission failed integrity check")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate": str(OUT),
        "sha256": sha256(OUT),
        "rows": int(len(check)),
        "ones": int(prediction.sum()),
        "positive_ratio": float(prediction.mean()),
        "shared_cv_threshold": THRESHOLD,
        "term_grouped_cv_macro_f1_mean": 0.8915726385234107,
        "term_grouped_cv_macro_f1_min": 0.8883808018288943,
        "term_grouped_cv_macro_f1_std": 0.00412230794016883,
        "independent_embedding_mixture_positive_prior": 0.4249825984122894,
        "cv_optimal_predicted_positive_ratio_range": [0.3932, 0.3969],
        "submission_status": "generated_only_not_submitted",
        "overfit_controls": [
            "term-grouped holdout",
            "shared threshold across folds",
            "no lexical/category/brand features",
            "no candidate-rank/count features",
            "test-query domain-matched random negatives",
            "no public-leaderboard parameter sweep",
        ],
        "caveat": (
            "The score estimate assumes train-positive embedding relationships transfer to "
            "test positives; leaderboard performance is not guaranteed."
        ),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
