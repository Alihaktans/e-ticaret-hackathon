from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V24_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
]

TOP_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v29_qswap_v24_legacy_sparse_raw_g0p05_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v29_qswap_v24_legacy_sparse_raw_g0p05_b8000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v29_qswap_v24_legacy_sparse_raw_g0p03_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v29_qswap_v24_legacy_sparse_raw_g0p03_b8000.csv",
]

OUT = ROOT / "reports/manual_review/review_v29_qswap_targets.csv"
OUT_SUM = ROOT / "reports/manual_review/review_v29_qswap_summary.csv"

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()

print("loading sample")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)

v24_path = first_existing(V24_PATHS)
top_path = first_existing(TOP_PATHS)

if v24_path is None:
    raise FileNotFoundError("v24 not found")
if top_path is None:
    raise FileNotFoundError("top qswap candidate not found")

print("v24:", v24_path)
print("top:", top_path)

pred_v24 = load_pred(v24_path, sample)
pred_top = load_pred(top_path, sample)

pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
    raise RuntimeError("pairs id order mismatch")

df = pairs.copy()
df["pred_v24"] = pred_v24
df["pred_top"] = pred_top
df["change"] = df["pred_top"] - df["pred_v24"]

adds = df[df["change"] == 1].copy()
drops = df[df["change"] == -1].copy()

adds["add_rank"] = adds.groupby("term_id").cumcount()
drops["drop_rank"] = drops.groupby("term_id").cumcount()

swaps = adds.merge(
    drops,
    left_on=["term_id", "add_rank"],
    right_on=["term_id", "drop_rank"],
    suffixes=("_add", "_drop"),
    how="inner",
)

print("adds", len(adds), "drops", len(drops), "paired swaps", len(swaps))

terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)

for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

item_cols = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]

swaps = swaps.merge(terms[["term_id", "query"]], on="term_id", how="left", validate="many_to_one")

swaps = swaps.merge(
    items[item_cols].add_suffix("_add"),
    left_on="item_id_add",
    right_on="item_id_add",
    how="left",
    validate="many_to_one",
)

swaps = swaps.merge(
    items[item_cols].add_suffix("_drop"),
    left_on="item_id_drop",
    right_on="item_id_drop",
    how="left",
    validate="many_to_one",
)

# Review sampling:
# 1) ilk 500 swap
# 2) farklı termlerden örnekler
# 3) random karışık
swaps["review_bucket"] = "qswap_top_candidate"

top = swaps.head(500).copy()

rest = swaps.drop(top.index)
if len(rest) > 500:
    rest = rest.sample(500, random_state=2026)

review = pd.concat([top, rest], ignore_index=True).drop_duplicates(["id_add", "id_drop"])

review["assistant_swap_label"] = ""
review["assistant_confidence"] = ""
review["needs_recheck"] = ""
review["reason"] = ""

keep = [
    "review_bucket",
    "query",

    "id_add",
    "item_id_add",
    "title_add",
    "category_add",
    "brand_add",
    "gender_add",
    "age_group_add",

    "id_drop",
    "item_id_drop",
    "title_drop",
    "category_drop",
    "brand_drop",
    "gender_drop",
    "age_group_drop",

    "assistant_swap_label",
    "assistant_confidence",
    "needs_recheck",
    "reason",

    "term_id",
    "attributes_add",
    "attributes_drop",
]

for c in keep:
    if c not in review.columns:
        review[c] = ""

review = review[keep]
review.to_csv(OUT, index=False, encoding="utf-8-sig")

summary = pd.DataFrame([{
    "top_candidate_file": str(top_path),
    "total_adds": len(adds),
    "total_drops": len(drops),
    "paired_swaps": len(swaps),
    "review_rows": len(review),
}])
summary.to_csv(OUT_SUM, index=False)

print(summary.to_string(index=False))
print("saved:", OUT)
print("saved:", OUT_SUM)
