from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
SPARSE_SCORE = ROOT / "data/processed/v26_sparse_text_scores.parquet"

OUT_REVIEW = ROOT / "reports/manual_review/review_v26_sparse_additions_targets.csv"
OUT_SUMMARY = ROOT / "reports/manual_review/review_v26_sparse_additions_summary.csv"

V24_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
]

CANDIDATES = [
    "sparse_addonly_supported_budget500",
    "sparse_addonly_supported_budget1000",
    "sparse_addonly_supported_budget2000",
    "sparse_addonly_supported_budget3000",
    "sparse_addonly_supported_budget5000",
    "sparse_addonly_strictpct3_budget500",
    "sparse_addonly_strictpct3_budget1000",
    "sparse_addonly_strictpct3_budget2000",
    "sparse_addonly_strictpct3_budget3000",
    "sparse_addonly_budget2000",
    "sparse_addonly_budget3000",
    "sparse_addonly_budget5000",
]

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

def add_bucket(df, mask, bucket, n_take=180):
    g = df[mask].copy()
    if len(g) == 0:
        return None

    g["review_bucket"] = bucket
    g = g.sort_values(["sparse_blend", "sparse_pct_rank"], ascending=[False, True])

    top_n = n_take // 2
    top = g.head(top_n)
    rest = g.drop(top.index)

    if len(rest) > n_take - len(top):
        rest = rest.sample(n=n_take - len(top), random_state=2026)

    return pd.concat([top, rest], ignore_index=True)

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()

v24_path = first_existing(V24_PATHS)
if v24_path is None:
    raise FileNotFoundError("v24 anchor not found")

print("loading v24:", v24_path)
pred_v24 = load_pred(v24_path, sample)

print("loading sparse score...")
score = pd.read_parquet(SPARSE_SCORE)
score["id"] = score["id"].astype(str)

# score içinde term_id/item_id zaten olabilir; yoksa pairs'ten ekle
if "term_id" not in score.columns or "item_id" not in score.columns:
    pairs = pd.read_csv(PAIRS)
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)
    score = score.merge(pairs[["id", "term_id", "item_id"]], on="id", how="left", validate="one_to_one")
else:
    score["term_id"] = score["term_id"].astype(str)
    score["item_id"] = score["item_id"].astype(str)

df = pd.DataFrame({
    "id": ids,
    "pred_v24": pred_v24,
}).merge(score, on="id", how="left", validate="one_to_one")

print("loading candidate predictions...")
for nm in CANDIDATES:
    p = ROOT / "submissions" / f"FINAL_CANDIDATE_v26_{nm}.csv"
    if not p.exists():
        print("missing:", p)
        continue

    print("loading", nm)
    df[f"pred_{nm}"] = load_pred(p, sample)

parts = []

def put(mask, bucket, n_take):
    part = add_bucket(df, mask, bucket, n_take=n_take)
    if part is not None and len(part):
        parts.append(part)

# En önemli katmanlar: 0-500, 500-1000, 1000-2000, 2000-3000
if "pred_sparse_addonly_supported_budget500" in df.columns:
    put(
        (df["pred_v24"] == 0) & (df["pred_sparse_addonly_supported_budget500"] == 1),
        "supported_0_500_added",
        220,
    )

if {"pred_sparse_addonly_supported_budget1000", "pred_sparse_addonly_supported_budget500"}.issubset(df.columns):
    put(
        (df["pred_sparse_addonly_supported_budget1000"] == 1)
        & (df["pred_sparse_addonly_supported_budget500"] == 0),
        "supported_500_1000_extra",
        180,
    )

if {"pred_sparse_addonly_supported_budget2000", "pred_sparse_addonly_supported_budget1000"}.issubset(df.columns):
    put(
        (df["pred_sparse_addonly_supported_budget2000"] == 1)
        & (df["pred_sparse_addonly_supported_budget1000"] == 0),
        "supported_1000_2000_extra",
        220,
    )

if {"pred_sparse_addonly_supported_budget3000", "pred_sparse_addonly_supported_budget2000"}.issubset(df.columns):
    put(
        (df["pred_sparse_addonly_supported_budget3000"] == 1)
        & (df["pred_sparse_addonly_supported_budget2000"] == 0),
        "supported_2000_3000_extra",
        180,
    )

# strict ile supported farkını ölç
if {"pred_sparse_addonly_strictpct3_budget2000", "pred_sparse_addonly_supported_budget2000"}.issubset(df.columns):
    put(
        (df["pred_sparse_addonly_strictpct3_budget2000"] == 1)
        & (df["pred_sparse_addonly_supported_budget2000"] == 0),
        "strictpct3_2000_extra_not_supported",
        180,
    )

if {"pred_sparse_addonly_supported_budget2000", "pred_sparse_addonly_strictpct3_budget2000"}.issubset(df.columns):
    put(
        (df["pred_sparse_addonly_supported_budget2000"] == 1)
        & (df["pred_sparse_addonly_strictpct3_budget2000"] == 0),
        "supported_2000_extra_not_strictpct3",
        180,
    )

# 3000-5000 risk katmanı
if {"pred_sparse_addonly_supported_budget5000", "pred_sparse_addonly_supported_budget3000"}.issubset(df.columns):
    put(
        (df["pred_sparse_addonly_supported_budget5000"] == 1)
        & (df["pred_sparse_addonly_supported_budget3000"] == 0),
        "supported_3000_5000_extra_risky",
        180,
    )

if {"pred_sparse_addonly_budget5000", "pred_sparse_addonly_supported_budget5000"}.issubset(df.columns):
    put(
        (df["pred_sparse_addonly_budget5000"] == 1)
        & (df["pred_sparse_addonly_supported_budget5000"] == 0),
        "raw_sparse_5000_extra_not_supported",
        150,
    )

if not parts:
    raise RuntimeError("no review parts created")

review = pd.concat(parts, ignore_index=True).drop_duplicates("id").reset_index(drop=True)

print("loading metadata...")
terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)

for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

review["term_id"] = review["term_id"].astype(str)
review["item_id"] = review["item_id"].astype(str)

review = (
    review
    .merge(terms[["term_id", "query"]], on="term_id", how="left", validate="many_to_one")
    .merge(
        items[["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
)

review["assistant_label"] = ""
review["assistant_confidence"] = ""
review["needs_recheck"] = ""
review["reason"] = ""

keep_cols = [
    "id",
    "review_bucket",
    "query",
    "title",
    "category",
    "brand",
    "gender",
    "age_group",
    "sparse_word",
    "sparse_char",
    "sparse_blend",
    "sparse_rank",
    "sparse_pct_rank",
    "sparse_delta_from_max",
    "assistant_label",
    "assistant_confidence",
    "needs_recheck",
    "reason",
    "term_id",
    "item_id",
    "attributes",
]

for c in keep_cols:
    if c not in review.columns:
        review[c] = ""

review = review[keep_cols]
review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

summary = (
    review.groupby("review_bucket")
    .agg(
        rows=("id", "count"),
        sparse_mean=("sparse_blend", "mean"),
        sparse_min=("sparse_blend", "min"),
        pct_rank_mean=("sparse_pct_rank", "mean"),
        pct_rank_max=("sparse_pct_rank", "max"),
    )
    .reset_index()
    .sort_values("review_bucket")
)

summary.to_csv(OUT_SUMMARY, index=False)

print("\nREVIEW SUMMARY")
print(summary.to_string(index=False))
print("saved:", OUT_REVIEW)
print("saved:", OUT_SUMMARY)
