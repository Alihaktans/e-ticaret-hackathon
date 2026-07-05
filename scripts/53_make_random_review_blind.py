from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "reports" / "manual_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 1000
SEED = 4242

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

sample = pairs.sample(n=N, random_state=SEED).copy()
sample = sample.merge(terms, on="term_id", how="left", validate="many_to_one")
sample = sample.merge(items, on="item_id", how="left", validate="many_to_one")

sample.insert(0, "human_label", "")
sample.insert(1, "assistant_label", "")
sample.insert(2, "label_note", "")

blind_cols = [
    "human_label",
    "assistant_label",
    "label_note",
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
]

sample = sample[blind_cols]

out = OUT_DIR / "random_review_blind_v2.csv"
sample.to_csv(out, index=False, encoding="utf-8-sig")

print("Saved:", out)
print("Rows:", len(sample))
print(sample[["query", "title", "category", "brand", "gender", "age_group"]].head(10).to_string(index=False))
