from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import RAW_DIR, PROCESSED_DIR, SUBMISSIONS_DIR, EXPERIMENT_REPORTS_DIR


SCORE_PATH = PROCESSED_DIR / "embedding_v1_minilm_topratio_008_scores.parquet"

RATIOS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.16]
FIXED_TOP_KS = [3, 5, 8, 10, 12, 15, 20]

MIN_POS_PER_QUERY = 1
MAX_POS_PER_QUERY = 20


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])
    assert set(sub["prediction"].unique()) <= {0, 1}


def build_submission(scores: pd.DataFrame, sample: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    tmp = scores[["id"]].copy()
    tmp["prediction"] = prediction.astype(np.int8)

    sub = sample[["id"]].merge(tmp, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)

    validate_submission(sub, sample)
    return sub


def make_top_ratio(scores: pd.DataFrame, sample: pd.DataFrame, ratio: float) -> dict:
    print(f"\nCreating top_ratio={ratio}")

    pred = np.zeros(len(scores), dtype=np.int8)
    score_series = scores["score"]

    for _, idx in tqdm(scores.groupby("term_id").groups.items(), desc=f"ratio {ratio}"):
        n = len(idx)

        k = round(n * ratio)
        k = max(MIN_POS_PER_QUERY, k)
        k = min(MAX_POS_PER_QUERY, k)
        k = min(k, n)

        top_idx = score_series.loc[idx].nlargest(k).index
        pred[top_idx] = 1

    sub = build_submission(scores, sample, pred)

    tag = str(ratio).replace("0.", "").zfill(3)
    out_path = SUBMISSIONS_DIR / f"embedding_v1_minilm_topratio_{tag}.csv"
    sub.to_csv(out_path, index=False)

    counts = sub["prediction"].value_counts().to_dict()
    ratios = sub["prediction"].value_counts(normalize=True).to_dict()

    print(f"Saved: {out_path}")
    print(counts)
    print(ratios)

    return {
        "strategy": "top_ratio",
        "value": ratio,
        "path": str(out_path),
        "prediction_counts": {str(k): int(v) for k, v in counts.items()},
        "prediction_ratios": {str(k): float(v) for k, v in ratios.items()},
    }


def make_fixed_topk(scores: pd.DataFrame, sample: pd.DataFrame, top_k: int) -> dict:
    print(f"\nCreating fixed_top_k={top_k}")

    pred = np.zeros(len(scores), dtype=np.int8)
    score_series = scores["score"]

    for _, idx in tqdm(scores.groupby("term_id").groups.items(), desc=f"topk {top_k}"):
        k = min(top_k, len(idx))
        top_idx = score_series.loc[idx].nlargest(k).index
        pred[top_idx] = 1

    sub = build_submission(scores, sample, pred)

    out_path = SUBMISSIONS_DIR / f"embedding_v1_minilm_fixed_top{top_k:02d}.csv"
    sub.to_csv(out_path, index=False)

    counts = sub["prediction"].value_counts().to_dict()
    ratios = sub["prediction"].value_counts(normalize=True).to_dict()

    print(f"Saved: {out_path}")
    print(counts)
    print(ratios)

    return {
        "strategy": "fixed_top_k",
        "value": top_k,
        "path": str(out_path),
        "prediction_counts": {str(k): int(v) for k, v in counts.items()},
        "prediction_ratios": {str(k): float(v) for k, v in ratios.items()},
    }


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORE_PATH.exists():
        raise FileNotFoundError(f"Score file not found: {SCORE_PATH}")

    print("=" * 100)
    print("03 MAKE EMBEDDING DECISION SUBMISSIONS")
    print("=" * 100)

    print(f"Reading scores: {SCORE_PATH}")
    scores = pd.read_parquet(SCORE_PATH)
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

    print(f"Scores shape: {scores.shape}")
    print(scores.head())
    print("\nScore summary:")
    print(scores["score"].describe())

    outputs = []

    for ratio in RATIOS:
        outputs.append(make_top_ratio(scores, sample, ratio))

    for top_k in FIXED_TOP_KS:
        outputs.append(make_fixed_topk(scores, sample, top_k))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_score_file": str(SCORE_PATH),
        "model_family": "embedding_v1_minilm",
        "ratios": RATIOS,
        "fixed_top_ks": FIXED_TOP_KS,
        "outputs": outputs,
        "notes": [
            "These submissions reuse cached MiniLM cosine scores.",
            "Only query-level decision strategy changes.",
            "No model re-training or re-embedding is performed.",
        ],
    }

    report_path = EXPERIMENT_REPORTS_DIR / "embedding_v1_decision_submissions_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
