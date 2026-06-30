from pathlib import Path
import pandas as pd
import numpy as np

RAW_SAMPLE = Path("data/raw/sample_submission.csv")

PROBA_FILES = [
    (
        "v4_single",
        Path("data/processed/embedding_ranker_v4_weighted_semantic_test_proba.parquet"),
        "proba",
    ),
    (
        "v4_ensemble_avg",
        Path("data/processed/embedding_ranker_v4_weight_seed_ensemble_test_ensemble_proba.parquet"),
        "ensemble_avg_proba",
    ),
    (
        "v4_ensemble_weighted",
        Path("data/processed/embedding_ranker_v4_weight_seed_ensemble_test_ensemble_proba.parquet"),
        "ensemble_weighted_proba",
    ),
]

THRESHOLDS = [
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
]

OUT_DIR = Path("submissions")
REPORT_PATH = Path("reports/experiments/threshold_submission_summary.csv")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(RAW_SAMPLE)
    rows = []

    for name, path, proba_col in PROBA_FILES:
        if not path.exists():
            print("MISSING:", path)
            continue

        print("=" * 100)
        print(name, path, proba_col)

        df = pd.read_parquet(path)

        if proba_col not in df.columns:
            print("COLUMN MISSING:", proba_col, "available:", df.columns.tolist())
            continue

        tmp = df[["id", proba_col]].copy()
        tmp = sample[["id"]].merge(tmp, on="id", how="left")

        if tmp[proba_col].isna().any():
            raise ValueError(f"Missing proba rows for {name}")

        for th in THRESHOLDS:
            sub = sample[["id"]].copy()
            sub["prediction"] = (tmp[proba_col].to_numpy() >= th).astype(int)

            tag = str(th).replace(".", "p")
            out_path = OUT_DIR / f"{name}_threshold_{tag}.csv"
            sub.to_csv(out_path, index=False)

            ones = int(sub["prediction"].sum())
            pos_ratio = float(sub["prediction"].mean())

            print(f"{out_path.name}: threshold={th}, ones={ones}, pos_ratio={pos_ratio:.6f}")

            rows.append(
                {
                    "name": name,
                    "proba_file": str(path),
                    "proba_col": proba_col,
                    "threshold": th,
                    "path": str(out_path),
                    "ones": ones,
                    "zeros": int(len(sub) - ones),
                    "pos_ratio": pos_ratio,
                }
            )

    report = pd.DataFrame(rows).sort_values(["name", "threshold"])
    report.to_csv(REPORT_PATH, index=False)

    print("=" * 100)
    print("DONE")
    print(report.to_string(index=False))
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
