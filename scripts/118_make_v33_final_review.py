from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
]

TOP_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v33_v24_qswap_v33_pure_hybrid_safe_g0p06_b24446.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_v24_qswap_v33_pure_hybrid_safe_g0p03_b24446.csv",
]

OUT = ROOT / "reports/manual_review/review_v33_final_qswap_targets.csv"
OUT_SUM = ROOT / "reports/manual_review/review_v33_final_qswap_summary.csv"

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

sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)

anchor_path = first_existing(ANCHOR_PATHS)
top_path = first_existing(TOP_PATHS)

if anchor_path is None:
    raise FileNotFoundError("v24 anchor bulunamadı")
if top_path is None:
    raise FileNotFoundError("v33 top candidate bulunamadı")

print("anchor:", anchor_path)
print("top:", top_path)

pred_anchor = load_pred(anchor_path, sample)
pred_top = load_pred(top_path, sample)

pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
    raise RuntimeError("pairs id order mismatch")

df = pairs.copy()
df["pred_anchor"] = pred_anchor
df["pred_top"] = pred_top
df["change"] = df["pred_top"] - df["pred_anchor"]

adds = df[df["change"] == 1].copy()
drops = df[df["change"] == -1].copy()

adds["pair_rank"] = adds.groupby("term_id").cumcount()
drops["pair_rank"] = drops.groupby("term_id").cumcount()

swaps = adds.merge(
    drops,
    on=["term_id", "pair_rank"],
    suffixes=("_add", "_drop"),
    how="inner",
)

print("adds:", len(adds), "drops:", len(drops), "paired:", len(swaps))

v33 = pd.read_parquet(V33)
v33["id"] = v33["id"].astype(str)

score_cols = ["id", "v33_ft_score", "v33_ft_rank", "v33_ft_pct_rank", "v33_ft_term_z"]
v33_add = v33[score_cols].add_suffix("_add")
v33_drop = v33[score_cols].add_suffix("_drop")

swaps = swaps.merge(v33_add, on="id_add", how="left", validate="many_to_one")
swaps = swaps.merge(v33_drop, on="id_drop", how="left", validate="many_to_one")

swaps["v33_score_gain"] = swaps["v33_ft_score_add"] - swaps["v33_ft_score_drop"]
swaps["v33_rank_gain"] = swaps["v33_ft_pct_rank_drop"] - swaps["v33_ft_pct_rank_add"]
swaps["v33_z_gain"] = swaps["v33_ft_term_z_add"] - swaps["v33_ft_term_z_drop"]

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
    on="item_id_add",
    how="left",
    validate="many_to_one",
)

swaps = swaps.merge(
    items[item_cols].add_suffix("_drop"),
    on="item_id_drop",
    how="left",
    validate="many_to_one",
)

# Review sampling:
# 400 highest gain, 300 middle, 300 weakest among selected.
swaps = swaps.sort_values("v33_score_gain", ascending=False).reset_index(drop=True)

top = swaps.head(400).copy()
top["review_bucket"] = "top_gain"

mid_start = max(0, len(swaps)//2 - 150)
mid = swaps.iloc[mid_start:mid_start+300].copy()
mid["review_bucket"] = "mid_gain"

tail = swaps.tail(300).copy()
tail["review_bucket"] = "tail_gain"

review = pd.concat([top, mid, tail], ignore_index=True)
review = review.drop_duplicates(["id_add", "id_drop"]).reset_index(drop=True)

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
    "v33_ft_score_add",
    "v33_ft_rank_add",
    "v33_ft_pct_rank_add",
    "v33_ft_term_z_add",

    "id_drop",
    "item_id_drop",
    "title_drop",
    "category_drop",
    "brand_drop",
    "gender_drop",
    "age_group_drop",
    "v33_ft_score_drop",
    "v33_ft_rank_drop",
    "v33_ft_pct_rank_drop",
    "v33_ft_term_z_drop",

    "v33_score_gain",
    "v33_rank_gain",
    "v33_z_gain",

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
    "anchor_file": str(anchor_path),
    "top_file": str(top_path),
    "total_adds": len(adds),
    "total_drops": len(drops),
    "paired_swaps": len(swaps),
    "review_rows": len(review),
    "score_gain_mean": float(swaps["v33_score_gain"].mean()),
    "score_gain_min": float(swaps["v33_score_gain"].min()),
    "score_gain_max": float(swaps["v33_score_gain"].max()),
}])

summary.to_csv(OUT_SUM, index=False)

print(summary.to_string(index=False))
print("saved:", OUT)
print("saved:", OUT_SUM)
