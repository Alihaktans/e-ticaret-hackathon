from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SUBMISSIONS_DIR = Path("submissions")
REPORT_DIR = Path("reports/experiments")

PROBA_PATH = PROCESSED_DIR / "embedding_ranker_v2_catboost_text_test_proba.parquet"

EMBED_WEIGHTS = [0.90, 0.80, 0.70, 0.60]
TOP_RATIOS = [0.08, 0.10, 0.12, 0.14, 0.16]

BASE_FILE = SUBMISSIONS_DIR / "embedding_ranker_v2_catboost_text_blend_topratio_012.csv"


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])
    assert set(sub["prediction"].unique()) <= {0, 1}


def make_submission(df: pd.DataFrame, sample: pd.DataFrame, score_col: str, ratio: float) -> pd.DataFrame:
    pred = np.zeros(len(df), dtype=np.int8)
    score_series = df[score_col]

    for _, idx in tqdm(df.groupby("term_id").groups.items(), desc=f"{score_col} top {ratio}"):
        n = len(idx)

        k = round(n * ratio)
        k = max(1, k)
        k = min(20, k)
        k = min(k, n)

        top_idx = score_series.loc[idx].nlargest(k).index
        pred[top_idx] = 1

    tmp = df[["id"]].copy()
    tmp["prediction"] = pred

    sub = sample[["id"]].merge(tmp, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)

    validate_submission(sub, sample)
    return sub


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("11 MAKE V2 BLEND VARIANTS")
    print("=" * 100)

    if not PROBA_PATH.exists():
        raise FileNotFoundError(PROBA_PATH)

    print(f"Reading: {PROBA_PATH}")
    df = pd.read_parquet(PROBA_PATH)

    required = {"id", "term_id", "item_id", "embedding_rank_score", "proba"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

    base = None
    if BASE_FILE.exists():
        base = pd.read_csv(BASE_FILE)
        print(f"Base comparison file found: {BASE_FILE.name}")
    else:
        print("Base comparison file not found; diff will be empty.")

    rows = []

    for ew in EMBED_WEIGHTS:
        mw = 1.0 - ew
        score_col = f"blend_e{int(ew * 100):03d}_m{int(mw * 100):03d}"

        df[score_col] = (
            ew * df["embedding_rank_score"].astype(np.float32)
            + mw * df["proba"].astype(np.float32)
        ).astype(np.float32)

        print("\n" + "=" * 100)
        print(f"Blend: embedding={ew:.2f}, model={mw:.2f}")
        print(df[score_col].describe())

        for ratio in TOP_RATIOS:
            tag_ratio = f"{int(round(ratio * 100)):03d}"
            out_name = f"embedding_ranker_v2_catboost_text_{score_col}_topratio_{tag_ratio}.csv"
            out_path = SUBMISSIONS_DIR / out_name

            sub = make_submission(df, sample, score_col, ratio)
            sub.to_csv(out_path, index=False)

            pos_ratio = float(sub["prediction"].mean())
            ones = int(sub["prediction"].sum())
            zeros = int(len(sub) - ones)

            diff_vs_base = None
            diff_ratio_vs_base = None

            if base is not None:
                diff_vs_base = int((sub["prediction"] != base["prediction"]).sum())
                diff_ratio_vs_base = float(diff_vs_base / len(sub))

            row = {
                "file": out_name,
                "embed_weight": ew,
                "model_weight": mw,
                "top_ratio": ratio,
                "zeros": zeros,
                "ones": ones,
                "pos_ratio": pos_ratio,
                "diff_vs_base_top012": diff_vs_base,
                "diff_ratio_vs_base_top012": diff_ratio_vs_base,
            }
            rows.append(row)

            print(f"Saved: {out_path}")
            print(row)

    summary = pd.DataFrame(rows)
    summary_path = REPORT_DIR / "v2_blend_variants_summary.csv"
    summary.to_csv(summary_path, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_proba": str(PROBA_PATH),
        "base_file": str(BASE_FILE),
        "embed_weights": EMBED_WEIGHTS,
        "top_ratios": TOP_RATIOS,
        "summary_path": str(summary_path),
        "notes": [
            "These submissions reuse V2 CatBoost probabilities.",
            "No retraining is performed.",
            "Higher embed_weight is safer; higher model_weight trusts CatBoost text features more.",
        ],
    }

    report_path = REPORT_DIR / "v2_blend_variants_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report : {report_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
