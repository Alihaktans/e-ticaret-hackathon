from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SUB = ROOT / "submissions"
OUT_DIR = ROOT / "reports" / "manual_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "manual_review_set_v1.csv"
N_PER_BUCKET = 120
SEED = 2026

CANDIDATES = {
    "old_v5_failed": SUB / "FINAL_v5_e5base_full900_avg_threshold_0p955.csv",
    "v7_global_prior": SUB / "CANDIDATE_v7_v5_global_prior_0p425.csv",
    "v10_prior": SUB / "CANDIDATE_v10_embedding_pu_prior.csv",
    "v10_majority": SUB / "FINAL_CANDIDATE_v10_embedding_pu_crossfit_majority.csv",
    "v10_shared_057": SUB / "FINAL_CANDIDATE_v10_embedding_pu_shared_cv_threshold_0p570.csv",
}

def read_pred(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Missing candidate: {path}")
        return None
    x = pd.read_csv(path, usecols=["id", "prediction"])
    x["id"] = x["id"].astype(str)
    return x.rename(columns={"prediction": f"pred_{name}"})

def sample_bucket(df: pd.DataFrame, mask, name: str, n: int) -> pd.DataFrame:
    part = df.loc[mask].copy()
    if len(part) == 0:
        print("Empty bucket:", name)
        return part
    take = min(n, len(part))
    out = part.sample(n=take, random_state=SEED)
    out["bucket"] = name
    print(name, len(part), "->", take)
    return out

def main():
    pairs = pd.read_csv(RAW / "submission_pairs.csv")
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    terms = pd.read_csv(RAW / "terms.csv")
    terms["term_id"] = terms["term_id"].astype(str)

    items = pd.read_csv(
        RAW / "items.csv",
        usecols=["item_id", "title", "category", "brand", "gender", "age_group", "attributes"],
    )
    items["item_id"] = items["item_id"].astype(str)

    df = pairs.merge(terms, on="term_id", how="left", validate="many_to_one")
    df = df.merge(items, on="item_id", how="left", validate="many_to_one")

    # Scores
    v5 = PROC / "v5_e5base_full900_test_proba.parquet"
    if v5.exists():
        s = pd.read_parquet(v5, columns=["id", "proba_avg"])
        s["id"] = s["id"].astype(str)
        df = df.merge(s.rename(columns={"proba_avg": "score_v5_full900"}), on="id", how="left")

    v10 = PROC / "embedding_pu_crossfit_v10_test_proba.parquet"
    if v10.exists():
        s = pd.read_parquet(v10, columns=["id", "proba"])
        s["id"] = s["id"].astype(str)
        df = df.merge(s.rename(columns={"proba": "score_v10"}), on="id", how="left")

    # Candidate predictions
    for name, path in CANDIDATES.items():
        pred = read_pred(path, name)
        if pred is not None:
            df = df.merge(pred, on="id", how="left")
            df[f"pred_{name}"] = df[f"pred_{name}"].fillna(0).astype(int)

    rows = []

    has_v10 = "pred_v10_prior" in df.columns
    has_old = "pred_old_v5_failed" in df.columns
    has_score_v10 = "score_v10" in df.columns

    if has_v10 and has_old:
        rows.append(sample_bucket(df, (df["pred_v10_prior"] == 1) & (df["pred_old_v5_failed"] == 0),
                                  "v10_yes_old_no", N_PER_BUCKET))
        rows.append(sample_bucket(df, (df["pred_v10_prior"] == 0) & (df["pred_old_v5_failed"] == 1),
                                  "old_yes_v10_no", N_PER_BUCKET))

    if has_score_v10:
        q90 = df["score_v10"].quantile(0.90)
        q70 = df["score_v10"].quantile(0.70)
        q50 = df["score_v10"].quantile(0.50)
        q30 = df["score_v10"].quantile(0.30)

        rows.append(sample_bucket(df, df["score_v10"] >= q90, "v10_top_10pct", N_PER_BUCKET))
        rows.append(sample_bucket(df, (df["score_v10"] >= q70) & (df["score_v10"] < q90),
                                  "v10_70_90pct", N_PER_BUCKET))
        rows.append(sample_bucket(df, (df["score_v10"] >= q50) & (df["score_v10"] < q70),
                                  "v10_50_70pct", N_PER_BUCKET))
        rows.append(sample_bucket(df, (df["score_v10"] >= q30) & (df["score_v10"] < q50),
                                  "v10_30_50pct", N_PER_BUCKET))

        rows.append(sample_bucket(df, (df["score_v10"] >= 0.52) & (df["score_v10"] <= 0.62),
                                  "v10_threshold_border", N_PER_BUCKET))

    rows.append(sample_bucket(df, np.ones(len(df), dtype=bool), "random_control", N_PER_BUCKET))

    review = pd.concat(rows, ignore_index=True)
    review = review.drop_duplicates("id").copy()

    wanted_cols = [
        "human_label",
        "bucket",
        "id",
        "term_id",
        "item_id",
        "query",
        "title",
        "category",
        "brand",
        "gender",
        "age_group",
        "attributes",
        "score_v5_full900",
        "score_v10",
    ]

    pred_cols = [c for c in review.columns if c.startswith("pred_")]
    wanted_cols += pred_cols

    for c in wanted_cols:
        if c not in review.columns:
            review[c] = ""

    review["human_label"] = ""
    review = review[wanted_cols]
    review.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Saved:", OUT)
    print("Rows:", len(review))
    print("Open this CSV and fill human_label with 1 or 0.")
    print("1 = query intent is satisfied")
    print("0 = wrong product / wrong category / wrong gender / important attribute mismatch")

if __name__ == "__main__":
    main()
