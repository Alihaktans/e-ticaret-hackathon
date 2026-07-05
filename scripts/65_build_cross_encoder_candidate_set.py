from pathlib import Path
import pandas as pd

ROOT = Path(".")
SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
OUT = ROOT / "data/processed/v13_cross_encoder_candidate_pairs.parquet"
REPORT = ROOT / "reports/manual_review/v13_cross_encoder_candidate_report.csv"

LOW = 0.45
HIGH = 0.80

scores = pd.read_parquet(SCORE, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

cand = scores[(scores["proba_avg"] >= LOW) & (scores["proba_avg"] <= HIGH)].copy()

pairs = pd.read_csv(PAIRS)
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(
    ITEMS,
    usecols=["item_id", "title", "category", "brand", "gender", "age_group", "attributes"],
    low_memory=False,
)
items["item_id"] = items["item_id"].astype(str)

cand = cand.merge(pairs, on="id", how="left", validate="one_to_one")
cand = cand.merge(terms, on="term_id", how="left", validate="many_to_one")
cand = cand.merge(items, on="item_id", how="left", validate="many_to_one")

cand["cross_text_query"] = cand["query"].fillna("").astype(str)
cand["cross_text_item"] = (
    "title: " + cand["title"].fillna("").astype(str)
    + " | category: " + cand["category"].fillna("").astype(str)
    + " | brand: " + cand["brand"].fillna("").astype(str)
    + " | gender: " + cand["gender"].fillna("").astype(str)
    + " | age_group: " + cand["age_group"].fillna("").astype(str)
    + " | attributes: " + cand["attributes"].fillna("").astype(str)
)

cand.to_parquet(OUT, index=False)

report = pd.DataFrame([{
    "low": LOW,
    "high": HIGH,
    "candidate_rows": len(cand),
    "total_rows": len(scores),
    "candidate_ratio": len(cand) / len(scores),
    "score_min": cand["proba_avg"].min(),
    "score_max": cand["proba_avg"].max(),
    "score_mean": cand["proba_avg"].mean(),
}])
report.to_csv(REPORT, index=False)

print(report.to_string(index=False))
print("saved:", OUT)
