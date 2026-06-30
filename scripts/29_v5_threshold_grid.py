from pathlib import Path
import pandas as pd
import numpy as np

PROBA_PATH = Path("data/processed/v5_e5base_full_test_proba.parquet")
SAMPLE_PATH = Path("data/raw/sample_submission.csv")
OUT_DIR = Path("submissions")
REPORT_PATH = Path("reports/experiments/v5_e5base_threshold_grid_summary.csv")

PROBA_COLS = [
    "proba_avg",
    "proba_seed42",
    "proba_seed2026",
    "proba_seed777",
]

THRESHOLDS = [
    0.920,
    0.925,
    0.930,
    0.935,
    0.940,
    0.945,
    0.950,
    0.955,
    0.960,
    0.965,
    0.970,
    0.975,
    0.980,
    0.985,
    0.990,
]

def tag(th: float) -> str:
    return f"{th:.3f}".replace(".", "p")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROBA_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    rows = []

    for col in PROBA_COLS:
        if col not in df.columns:
            print("MISSING COL:", col)
            continue

        tmp = sample[["id"]].merge(df[["id", col]], on="id", how="left")

        if tmp[col].isna().any():
            raise ValueError(f"Missing probabilities for {col}")

        for th in THRESHOLDS:
            sub = sample[["id"]].copy()
            sub["prediction"] = (tmp[col].to_numpy() >= th).astype(int)

            assert list(sub.columns) == ["id", "prediction"]
            assert len(sub) == len(sample)
            assert sub["id"].equals(sample["id"])
            assert set(sub["prediction"].unique()) <= {0, 1}

            out_path = OUT_DIR / f"v5_e5base_{col}_threshold_{tag(th)}.csv"
            sub.to_csv(out_path, index=False)

            ones = int(sub["prediction"].sum())
            pos_ratio = float(sub["prediction"].mean())

            rows.append(
                {
                    "proba_col": col,
                    "threshold": th,
                    "path": str(out_path),
                    "ones": ones,
                    "zeros": int(len(sub) - ones),
                    "pos_ratio": pos_ratio,
                    "distance_to_0p125": abs(pos_ratio - 0.125),
                    "distance_to_0p132": abs(pos_ratio - 0.132347),
                }
            )

            print(f"{col} th={th:.3f} ones={ones:,} pos_ratio={pos_ratio:.6f}")

    report = pd.DataFrame(rows)

    report = report.sort_values(["distance_to_0p132", "proba_col", "threshold"])
    report.to_csv(REPORT_PATH, index=False)

    print("=" * 100)
    print("DONE")
    print("Closest to CV best pred_pos_ratio ≈ 0.132347:")
    print(report.head(20).to_string(index=False))
    print("\nSaved:", REPORT_PATH)

if __name__ == "__main__":
    main()
