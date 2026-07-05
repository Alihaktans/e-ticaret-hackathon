from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
SUB_PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V21 = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"

OUT = ROOT / "reports/manual_review/review_v24_vote_refinement_targets.csv"
REPORT = ROOT / "reports/manual_review/review_v24_vote_refinement_summary.csv"

RANDOM_SEED = 2026

PATHS = {
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
    ],
    "add5": [
        ROOT / "submissions/FINAL_MAIN_v21_v20_add5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add5000.csv",
    ],
    "vote_base": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "swap5000": [
        ROOT / "submissions/FINAL_MAIN_v23_vote_swap5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_swap_restore_trim_budget5000.csv",
    ],
    "swap8000": [
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_swap_restore_trim_budget8000.csv",
    ],
    "swap12000": [
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_swap_restore_trim_budget12000.csv",
    ],
    "restore5000": [
        ROOT / "submissions/FINAL_MAIN_v23_vote_restore5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_restore_blend_budget5000.csv",
    ],
    "restore12000": [
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_restore_blend_budget12000.csv",
    ],
    "v18ge09": [
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_restore_v18ge0p9.csv",
    ],
}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_sub(name, sample, required=True):
    p = first_existing(PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(f"missing {name}")
        print("missing optional:", name)
        return None

    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)

    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {name}")

    return d["prediction"].astype(np.int8).to_numpy()

def take_sample(df, n, sort_col=None, ascending=False):
    if len(df) <= n:
        return df.copy()

    if sort_col is None or sort_col not in df.columns:
        return df.sample(n=n, random_state=RANDOM_SEED)

    top_n = n // 2
    top = df.sort_values(sort_col, ascending=ascending).head(top_n)
    rest = df.drop(top.index)

    if len(rest) <= n - len(top):
        return pd.concat([top, rest], ignore_index=True)

    rnd = rest.sample(n=n - len(top), random_state=RANDOM_SEED)
    return pd.concat([top, rnd], ignore_index=True)

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

pred_v20 = load_sub("v20", sample)
pred_add5 = load_sub("add5", sample)
pred_base = load_sub("vote_base", sample)
pred_swap5 = load_sub("swap5000", sample)
pred_swap8 = load_sub("swap8000", sample, required=False)
pred_swap12 = load_sub("swap12000", sample, required=False)
pred_restore5 = load_sub("restore5000", sample, required=False)
pred_restore12 = load_sub("restore12000", sample, required=False)
pred_v18ge09 = load_sub("v18ge09", sample, required=False)

if pred_swap8 is None:
    pred_swap8 = pred_swap5.copy()
if pred_swap12 is None:
    pred_swap12 = pred_swap8.copy()
if pred_restore5 is None:
    pred_restore5 = pred_base.copy()
if pred_restore12 is None:
    pred_restore12 = pred_restore5.copy()
if pred_v18ge09 is None:
    pred_v18ge09 = pred_base.copy()

print("loading scores...")
v21 = pd.read_parquet(V21)
v21["id"] = v21["id"].astype(str)
assert v21["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
v21_score = v21["v21_score"].astype("float32").to_numpy()

v5_score = np.full(n, -1.0, dtype=np.float32)
if V5.exists():
    v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
    v5["id"] = v5["id"].astype(str)
    assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
    v5_score = v5["proba_avg"].astype("float32").to_numpy()

v18_score = np.full(n, -1.0, dtype=np.float32)
if V18.exists():
    v18 = pd.read_parquet(V18)
    v18["id"] = v18["id"].astype(str)
    idx = v18["id"].map(id_to_idx)
    ok = idx.notna()
    v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()

base = pd.DataFrame({
    "id": sample["id"].to_numpy(),
    "pred_v20": pred_v20,
    "pred_add5": pred_add5,
    "pred_vote_base": pred_base,
    "pred_swap5": pred_swap5,
    "pred_swap8": pred_swap8,
    "pred_swap12": pred_swap12,
    "pred_restore5": pred_restore5,
    "pred_restore12": pred_restore12,
    "pred_v18ge09": pred_v18ge09,
    "v21_score": v21_score,
    "v5_score": v5_score,
    "v18_score": v18_score,
})

# swap5000: base'e göre geri eklenenler
g1 = base[(base.pred_vote_base == 0) & (base.pred_swap5 == 1)].copy()
g1["review_bucket"] = "swap5_restored_base0_to1"

# swap5000: base'den silinenler
g2 = base[(base.pred_vote_base == 1) & (base.pred_swap5 == 0)].copy()
g2["review_bucket"] = "swap5_trimmed_base1_to0"

# swap8000'in swap5000 üstüne ekstra restore ettiği
g3 = base[(base.pred_swap5 == 0) & (base.pred_swap8 == 1)].copy()
g3["review_bucket"] = "swap8_extra_restored_over_swap5"

# swap8000'in swap5000 üstüne ekstra trimlediği
g4 = base[(base.pred_swap5 == 1) & (base.pred_swap8 == 0)].copy()
g4["review_bucket"] = "swap8_extra_trimmed_over_swap5"

# restore5000'in ekledikleri
g5 = base[(base.pred_vote_base == 0) & (base.pred_restore5 == 1)].copy()
g5["review_bucket"] = "restore5_added_over_base"

# restore12000'in restore5000 üstüne ekstra ekledikleri
g6 = base[(base.pred_restore5 == 0) & (base.pred_restore12 == 1)].copy()
g6["review_bucket"] = "restore12_extra_over_restore5"

# v18 yüksek restore
g7 = base[(base.pred_vote_base == 0) & (base.pred_v18ge09 == 1)].copy()
g7["review_bucket"] = "v18ge09_restored_over_base"

# base pozitif ama hem v21 hem v18 zayıf: silme adayı
g8 = base[
    (base.pred_vote_base == 1)
    & (base.v21_score < 0.05)
    & ((base.v18_score < 0.10) | (base.v18_score < 0))
    & (base.v5_score < 0.25)
].copy()
g8["review_bucket"] = "base_positive_weak_all_trim_candidate"

parts = [
    take_sample(g1, 250, "v5_score", ascending=False),
    take_sample(g2, 250, "v5_score", ascending=True),
    take_sample(g3, 180, "v5_score", ascending=False),
    take_sample(g4, 180, "v5_score", ascending=True),
    take_sample(g5, 200, "v5_score", ascending=False),
    take_sample(g6, 200, "v5_score", ascending=False),
    take_sample(g7, 180, "v18_score", ascending=False),
    take_sample(g8, 180, "v21_score", ascending=True),
]

review = pd.concat(parts, ignore_index=True)
review = review.drop_duplicates("id").reset_index(drop=True)

print("selected:", len(review))
print(review["review_bucket"].value_counts().to_string())

print("loading metadata...")
pairs = pd.read_csv(SUB_PAIRS)
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)

for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

review = (
    review
    .merge(pairs[["id", "term_id", "item_id"]], on="id", how="left", validate="one_to_one")
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

front_cols = [
    "id", "review_bucket",
    "query", "title", "category", "brand", "gender", "age_group",
    "pred_v20", "pred_add5", "pred_vote_base",
    "pred_swap5", "pred_swap8", "pred_swap12",
    "pred_restore5", "pred_restore12", "pred_v18ge09",
    "v21_score", "v5_score", "v18_score",
    "assistant_label", "assistant_confidence", "needs_recheck", "reason",
    "term_id", "item_id", "attributes",
]

review = review[front_cols]
review.to_csv(OUT, index=False, encoding="utf-8-sig")

summary = (
    review.groupby("review_bucket")
    .agg(
        rows=("id", "count"),
        v21_mean=("v21_score", "mean"),
        v21_min=("v21_score", "min"),
        v21_max=("v21_score", "max"),
        v5_mean=("v5_score", "mean"),
        v18_mean=("v18_score", "mean"),
    )
    .reset_index()
)

summary.to_csv(REPORT, index=False)

print(summary.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
