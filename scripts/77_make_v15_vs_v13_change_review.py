from pathlib import Path
import pandas as pd

ROOT = Path(".")

V13 = ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv"
V15 = ROOT / "submissions/FINAL_MAIN_v15_bge_w020_t018_GSB.csv"

PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V5S = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
BGE = ROOT / "data/processed/v15_bge_reranker_candidate_scores.parquet"
CE = ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet"

OUT = ROOT / "reports/manual_review/review_v15_vs_v13_changes.csv"

v13 = pd.read_csv(V13)
v15 = pd.read_csv(V15)

v13["id"] = v13["id"].astype(str)
v15["id"] = v15["id"].astype(str)

assert v13["id"].equals(v15["id"])

df = pd.DataFrame({
    "id": v13["id"],
    "pred_v13": v13["prediction"].astype(int),
    "pred_v15": v15["prediction"].astype(int),
})

df = df[df["pred_v13"] != df["pred_v15"]].copy()

df["change_type"] = df.apply(
    lambda r: "v15_added_1" if r["pred_v13"] == 0 and r["pred_v15"] == 1 else "v15_removed_1",
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

v5s = pd.read_parquet(V5S, columns=["id", "proba_avg"])
v5s["id"] = v5s["id"].astype(str)

bge = pd.read_parquet(BGE, columns=["id", "bge_sigmoid", "bge_blend_w020"])
bge["id"] = bge["id"].astype(str)

ce = pd.read_parquet(CE, columns=["id", "cross_sigmoid"])
ce["id"] = ce["id"].astype(str)

df = df.merge(pairs, on="id", how="left", validate="one_to_one")
df = df.merge(terms, on="term_id", how="left", validate="many_to_one")
df = df.merge(items, on="item_id", how="left", validate="many_to_one")
df = df.merge(v5s, on="id", how="left", validate="many_to_one")
df = df.merge(bge, on="id", how="left", validate="many_to_one")
df = df.merge(ce, on="id", how="left", validate="many_to_one")

parts = []
for change_type, n in [("v15_added_1", 300), ("v15_removed_1", 300)]:
    part = df[df["change_type"].eq(change_type)].copy()
    part = part.sample(n=min(n, len(part)), random_state=2027)
    parts.append(part)

sample = pd.concat(parts, ignore_index=True)
sample.insert(0, "assistant_label", "")
sample.insert(1, "label_note", "")

cols = [
    "assistant_label", "label_note",
    "change_type", "id",
    "pred_v13", "pred_v15",
    "proba_avg", "cross_sigmoid", "bge_sigmoid", "bge_blend_w020",
    "query", "title", "category", "brand", "gender", "age_group", "attributes",
    "term_id", "item_id",
]

sample = sample[[c for c in cols if c in sample.columns]]
sample.to_csv(OUT, index=False, encoding="utf-8-sig")

print("total changed:", len(df))
print(df["change_type"].value_counts().to_string())
print("sample saved:", OUT)
print(sample[["change_type", "proba_avg", "cross_sigmoid", "bge_sigmoid", "query", "title", "brand"]].head(20).to_string(index=False))
