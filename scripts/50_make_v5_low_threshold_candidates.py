from pathlib import Path
import pandas as pd
import numpy as np

score = pd.read_parquet(
    "data/processed/v5_e5base_full900_test_proba.parquet",
    columns=["id", "proba_avg"]
)
sample = pd.read_csv("data/raw/sample_submission.csv")

assert score["id"].astype(str).reset_index(drop=True).equals(
    sample["id"].astype(str).reset_index(drop=True)
)

Path("submissions").mkdir(exist_ok=True)
Path("reports/manual_review").mkdir(parents=True, exist_ok=True)

rows = []

for th in [0.50, 0.525, 0.55, 0.575]:
    pred = (score["proba_avg"].to_numpy() >= th).astype(np.int8)
    th_tag = str(th).replace(".", "p")
    out = Path("submissions") / f"CANDIDATE_v5_full900_threshold_{th_tag}.csv"

    pd.DataFrame({
        "id": score["id"],
        "prediction": pred
    }).to_csv(out, index=False)

    rows.append({
        "file": str(out),
        "threshold": th,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean())
    })

summary = pd.DataFrame(rows)
summary.to_csv("reports/manual_review/v5_low_threshold_candidates_summary.csv", index=False)

print(summary.to_string(index=False))
