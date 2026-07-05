from pathlib import Path
import pandas as pd

ROOT = Path(".")
SUB_05 = ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p5_GENDER_SAFE_BRAND_FILTERED.csv"
SUB_07 = ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_GENDER_SAFE_BRAND_FILTERED.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
SCORES = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
OUT = ROOT / "reports/manual_review/review_added_by_0p5_vs_0p7.csv"

s05 = pd.read_csv(SUB_05)
s07 = pd.read_csv(SUB_07)

assert s05["id"].astype(str).equals(s07["id"].astype(str))

mask = (s05["prediction"].eq(1)) & (s07["prediction"].eq(0))
ids = s05.loc[mask, ["id"]].copy()
ids["id"] = ids["id"].astype(str)

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

scores = pd.read_parquet(SCORES, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

df = ids.merge(pairs, on="id", how="left", validate="one_to_one")
df = df.merge(terms, on="term_id", how="left", validate="many_to_one")
df = df.merge(items, on="item_id", how="left", validate="many_to_one")
df = df.merge(scores, on="id", how="left", validate="many_to_one")

# Farktan rastgele 300 satır seçiyoruz.
sample = df.sample(n=min(300, len(df)), random_state=2026).copy()
sample.insert(0, "assistant_label", "")
sample.insert(1, "label_note", "")

cols = [
    "assistant_label", "label_note", "id", "proba_avg",
    "query", "title", "category", "brand", "gender", "age_group", "attributes",
    "term_id", "item_id",
]
sample = sample[[c for c in cols if c in sample.columns]]
sample.to_csv(OUT, index=False, encoding="utf-8-sig")

print("added_by_0p5_vs_0p7 total:", len(df))
print("sample saved:", OUT)
print(sample[["proba_avg", "query", "title", "category", "brand", "gender"]].head(20).to_string(index=False))
