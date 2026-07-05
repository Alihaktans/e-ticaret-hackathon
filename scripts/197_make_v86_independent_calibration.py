"""Calibrate v85 scores globally without borrowing any v22/v71 query quota."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(".")
SCORES = ROOT / "data/processed/v85_independent_test_scores.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"
OUT_DIR = ROOT / "submissions/final_candidates_v86"
REPORT = ROOT / "reports/experiments/v86_independent_calibration.json"
RATIOS = [0.28, 0.30, 0.32, 0.34, 0.36, 0.40]


def main():
    s = pd.read_parquet(SCORES)
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    s["id"] = s["id"].astype(str); sample["id"] = sample["id"].astype(str)
    if not s["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("score/sample alignment failed")
    score_col = "v85_score" if "v85_score" in s.columns else "v82_score"
    score = s[score_col].to_numpy(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ratio in RATIOS:
        k = int(round(len(score) * ratio))
        take = np.argpartition(score, -k)[-k:]
        pred = np.zeros(len(score), dtype=np.int8); pred[take] = 1
        tag = str(ratio).replace(".", "p")
        path = OUT_DIR / f"FINAL_CANDIDATE_v86_v85_global_ratio{tag}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred}).to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        qk = pd.DataFrame({"term_id": s["term_id"], "p": pred}).groupby("term_id")["p"].sum()
        rows.append({"file": str(path), "ratio": ratio, "positives": int(pred.sum()),
                     "threshold": float(np.min(score[take])), "queries_zero": int((qk == 0).sum()),
                     "mean_k": float(qk.mean()), "median_k": float(qk.median()),
                     "p95_k": float(qk.quantile(.95)), "sha256": digest})
    REPORT.write_text(json.dumps({"quota_source": None, "anchor_source": None,
                                  "score_model": "v85 independent relative-history classifier",
                                  "candidates": rows}, indent=2), encoding="utf8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
