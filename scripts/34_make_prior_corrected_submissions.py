from __future__ import annotations

"""Create review candidates using the independently estimated test class prior.

No Kaggle submission is performed. Files are generated for offline inspection.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "experiments" / "test_class_prior_mixture.json"
OUT_DIR = ROOT / "submissions"


def validate_and_write(ids: pd.Series, pred: np.ndarray, path: Path) -> dict:
    sample = pd.read_csv(RAW / "sample_submission.csv", usecols=["id"])
    if len(ids) != len(sample):
        raise ValueError("Submission length mismatch")
    if not ids.astype(str).reset_index(drop=True).equals(sample["id"].astype(str)):
        raise ValueError("Submission id order mismatch")
    unique = set(np.unique(pred).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Invalid predictions: {unique}")
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(pred)),
        "ones": int(pred.sum()),
        "positive_ratio": float(pred.mean()),
    }


def main() -> None:
    prior_report = json.loads(REPORT.read_text(encoding="utf-8"))
    prior = float(prior_report["positive_prior_mean"])
    scores = pd.read_parquet(
        PROC / "v5_e5base_full900_test_proba.parquet",
        columns=["id", "term_id", "proba_avg"],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cutoff = float(scores["proba_avg"].quantile(1.0 - prior))
    global_pred = scores["proba_avg"].ge(cutoff).to_numpy(dtype=np.int8)

    group_size = scores.groupby("term_id")["proba_avg"].transform("size").to_numpy()
    within_rank = scores.groupby("term_id")["proba_avg"].rank(
        method="first", ascending=False
    ).to_numpy()
    per_term_k = np.maximum(1, np.rint(group_size * prior)).astype(np.int32)
    per_term_pred = (within_rank <= per_term_k).astype(np.int8)

    prior_tag = f"{prior:.3f}".replace(".", "p")
    summaries = [
        validate_and_write(
            scores["id"],
            global_pred,
            OUT_DIR / f"CANDIDATE_v7_v5_global_prior_{prior_tag}.csv",
        ),
        validate_and_write(
            scores["id"],
            per_term_pred,
            OUT_DIR / f"CANDIDATE_v7_v5_perterm_prior_{prior_tag}.csv",
        ),
    ]
    comparison = {
        "estimated_prior": prior,
        "global_probability_cutoff": cutoff,
        "candidate_summaries": summaries,
        "prediction_agreement": float((global_pred == per_term_pred).mean()),
        "recommended_for_first_review": summaries[0]["path"],
        "note": "Generated only; not submitted to Kaggle.",
    }
    out = ROOT / "reports" / "experiments" / "prior_corrected_submission_candidates.json"
    out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
