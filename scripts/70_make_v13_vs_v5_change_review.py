from pathlib import Path
import pandas as pd

ROOT = Path(".")
V5 = ROOT / "submissions/FINAL_MAIN_v5_full900_0p6_GENDER_SAFE_BRAND.csv"
V13 = ROOT / "submissions/FINAL_MAIN_v13_banded_ce_w025_t0275_GSB.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
SCORES_V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
SCORES_CE = ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet"

OUT = ROOT / "reports/manual_review/review_v13_vs_v5_changes.csv"

v5 = pd.read_csv(V5)
v13 = pd.read_csv(V13)

assert v5["id"].astype(str).equals(v13["id"].astype(str))

v5["id"] = v5["id"].astype(str)
v13["id"] = v13["id"].astype(str)

df = pd.DataFrame({
    "id": v5["id"],
    "pred_v5": v5["prediction"].astype(int),
    "pred_v13": v13["prediction"].astype(int),
})

df = df[df["pred_v5"] != df["pred_v13"]].copy()

df["change_type"] = df.apply(
    lambda r: "v13_added_1" if r["pred_v5"] == 0 and r["pred_v13"] == 1 else "v13_removed_1",
    axis=1,
)

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

v5s = pd.read_parquet(SCORES_V5, columns=["id", "proba_avg"])
v5s["id"] = v5s["id"].astype(str)

ces = pd.read_parquet(SCORES_CE, columns=["id", "cross_sigmoid", "blend_w060"])
ces["id"] = ces["id"].astype(str)

df = df.merge(pairs, on="id", how="left", validate="one_to_one")
df = df.merge(terms, on="term_id", how="left", validate="many_to_one")
df = df.merge(items, on="item_id", how="left", validate="many_to_one")
df = df.merge(v5s, on="id", how="left", validate="many_to_one")
df = df.merge(ces, on="id", how="left", validate="many_to_one")

parts = []
for change_type, n in [("v13_added_1", 250), ("v13_removed_1", 250)]:
    part = df[df["change_type"].eq(change_type)].copy()
    part = part.sample(n=min(n, len(part)), random_state=2026)
    parts.append(part)

sample = pd.concat(parts, ignore_index=True)
sample.insert(0, "assistant_label", "")
sample.insert(1, "label_note", "")

cols = [
    "assistant_label", "label_note",
    "change_type", "id",
    "pred_v5", "pred_v13",
    "proba_avg", "cross_sigmoid", "blend_w060",
    "query", "title", "category", "brand", "gender", "age_group", "attributes",
    "term_id", "item_id",
]

sample = sample[[c for c in cols if c in sample.columns]]
sample.to_csv(OUT, index=False, encoding="utf-8-sig")

print("total changed:", len(df))
print(df["change_type"].value_counts().to_string())
print("sample saved:", OUT)
print(sample[["change_type", "proba_avg", "cross_sigmoid", "query", "title", "category", "brand"]].head(20).to_string(index=False))
