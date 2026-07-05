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

OUT = ROOT / "reports/manual_review/review_v21_active_learning_targets.csv"
REPORT = ROOT / "reports/manual_review/review_v21_active_learning_targets_summary.csv"

RANDOM_SEED = 2026

PATHS = {
    "v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
    "v19p1": [
        ROOT / "submissions/FINAL_MAIN_v19p1_honest_addonly_ba012_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba012_r1_v18th0p9.csv",
    ],
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
    ],
    "v21_add5k": [
        ROOT / "submissions/FINAL_MAIN_v21_v20_add5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add5000.csv",
    ],
    "v21_add12k": [
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add12000.csv",
    ],
    "v21_asym10k1k": [
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_asym_add10000_rem1000.csv",
    ],
    "v21_vote": [
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_vote_v16_v17_v21_2of3_323.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_vote_v16_v17_v21_2of3_325.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_vote_v16_v17_v21_2of3_032.csv",
    ],
    "v17p4": [
        ROOT / "submissions/FINAL_MAIN_v17p4_independent_b04_ba008_r1.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v17p4_v17p4_qt_b0p4_ba0p08_sa0p0_ga0p0_nocap_r1.csv",
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
            raise FileNotFoundError(f"Missing submission for {name}")
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

pred_v16 = load_sub("v16", sample)
pred_v19 = load_sub("v19p1", sample)
pred_v20 = load_sub("v20", sample)
pred_add5 = load_sub("v21_add5k", sample)
pred_add12 = load_sub("v21_add12k", sample, required=False)
pred_asym = load_sub("v21_asym10k1k", sample, required=False)
pred_vote = load_sub("v21_vote", sample, required=False)
pred_v17 = load_sub("v17p4", sample, required=False)

if pred_add12 is None:
    pred_add12 = pred_add5.copy()
if pred_asym is None:
    pred_asym = pred_add5.copy()
if pred_vote is None:
    pred_vote = pred_add5.copy()
if pred_v17 is None:
    pred_v17 = pred_v16.copy()

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
    "pred_v16": pred_v16,
    "pred_v19p1": pred_v19,
    "pred_v20": pred_v20,
    "pred_v17p4": pred_v17,
    "pred_v21_add5k": pred_add5,
    "pred_v21_add12k": pred_add12,
    "pred_v21_asym10k1k": pred_asym,
    "pred_v21_vote": pred_vote,
    "v21_score": v21_score,
    "v5_score": v5_score,
    "v18_score": v18_score,
})

# 1) En güvenli v21 katkısı: add5k v20 üstüne ne ekledi?
g1 = base[(base.pred_v20 == 0) & (base.pred_v21_add5k == 1)].copy()
g1["review_bucket"] = "v21_add5k_added_over_v20"

# 2) 5k'dan 12k'ya genişletilen ekstra eklemeler doğru mu?
g2 = base[
    (base.pred_v20 == 0)
    & (base.pred_v21_add5k == 0)
    & (base.pred_v21_add12k == 1)
].copy()
g2["review_bucket"] = "v21_add12k_extra_over_add5k"

# 3) Asym adayın ekledikleri
g3 = base[(base.pred_v20 == 0) & (base.pred_v21_asym10k1k == 1)].copy()
g3["review_bucket"] = "v21_asym10k1k_added_over_v20"

# 4) Asym adayın sildikleri
g4 = base[(base.pred_v20 == 1) & (base.pred_v21_asym10k1k == 0)].copy()
g4["review_bucket"] = "v21_asym10k1k_removed_from_v20"

# 5) Büyük potansiyel aday: vote modeli v20'ye ne ekliyor?
g5 = base[(base.pred_v20 == 0) & (base.pred_v21_vote == 1)].copy()
g5["review_bucket"] = "v21_vote_added_over_v20"

# 6) Büyük potansiyel aday: vote modeli v20'den ne siliyor?
g6 = base[(base.pred_v20 == 1) & (base.pred_v21_vote == 0)].copy()
g6["review_bucket"] = "v21_vote_removed_from_v20"

# 7) v21 yüksek ama v18 düşük: riskli addition kontrolü
g7 = base[
    (base.pred_v20 == 0)
    & (base.v21_score >= np.quantile(base.v21_score, 0.995))
    & (base.v18_score >= 0)
    & (base.v18_score <= 0.10)
].copy()
g7["review_bucket"] = "v21_high_but_v18_low_risky_add"

parts = [
    take_sample(g1, 250, "v21_score", ascending=False),
    take_sample(g2, 250, "v21_score", ascending=False),
    take_sample(g3, 150, "v21_score", ascending=False),
    take_sample(g4, 150, "v21_score", ascending=True),
    take_sample(g5, 300, "v21_score", ascending=False),
    take_sample(g6, 300, "v21_score", ascending=True),
    take_sample(g7, 150, "v21_score", ascending=False),
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
    "pred_v16", "pred_v19p1", "pred_v20", "pred_v17p4",
    "pred_v21_add5k", "pred_v21_add12k", "pred_v21_asym10k1k", "pred_v21_vote",
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
        v18_mean=("v18_score", "mean"),
        v5_mean=("v5_score", "mean"),
    )
    .reset_index()
)

summary.to_csv(REPORT, index=False)

print(summary.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
