from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
SUB_PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"
V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"

SUBS = {
    "v13": [
        ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
    ],
    "v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
    "v17p4": [
        ROOT / "submissions/FINAL_MAIN_v17p4_independent_b04_ba008_r1.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v17p4_v17p4_qt_b0p4_ba0p08_sa0p0_ga0p0_nocap_r1.csv",
    ],
    "v19p1": [
        ROOT / "submissions/FINAL_MAIN_v19p1_honest_addonly_ba012_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba012_r1_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba008_r1_v18th0p9.csv",
    ],
}

OUT = ROOT / "reports/manual_review/review_v20_active_learning_targets.csv"
REPORT = ROOT / "reports/manual_review/review_v20_active_learning_targets_summary.csv"

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_sub(name, sample):
    p = first_existing(SUBS[name])
    if p is None:
        raise FileNotFoundError(f"missing {name}")
    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    assert d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), name
    return d["prediction"].astype(np.int8).to_numpy(), p

def take_sample(df, n, sort_col=None, ascending=False):
    if len(df) <= n:
        return df.copy()
    if sort_col is not None and sort_col in df.columns:
        # yarısı en uçlardan, yarısı random
        top = df.sort_values(sort_col, ascending=ascending).head(n // 2)
        rest = df.drop(top.index)
        rnd = rest.sample(n=n - len(top), random_state=RANDOM_SEED)
        return pd.concat([top, rnd], ignore_index=True)
    return df.sample(n=n, random_state=RANDOM_SEED)

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
n = len(sample)

pred_v13, p13 = load_sub("v13", sample)
pred_v16, p16 = load_sub("v16", sample)
pred_v17, p17 = load_sub("v17p4", sample)
pred_v19, p19 = load_sub("v19p1", sample)

id_to_idx = pd.Series(np.arange(n), index=sample["id"])

print("loading v18...")
v18 = pd.read_parquet(V18)
v18["id"] = v18["id"].astype(str)
v18_score = np.full(n, -1.0, dtype=np.float32)
idx = v18["id"].map(id_to_idx)
ok = idx.notna()
v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()
has_v18 = v18_score >= 0

print("loading v5...")
v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
v5["id"] = v5["id"].astype(str)
assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
v5_score = v5["proba_avg"].astype("float32").to_numpy()

base = pd.DataFrame({
    "id": sample["id"].to_numpy(),
    "pred_v13": pred_v13,
    "pred_v16": pred_v16,
    "pred_v17p4": pred_v17,
    "pred_v19p1": pred_v19,
    "v18_score": v18_score,
    "v5_score": v5_score,
})

# Grup 1: v19p1'in eklediği satırlar. Bunlar en kritik: gerçekten doğruysa public artar.
g1 = base[(base.pred_v16 == 0) & (base.pred_v19p1 == 1)].copy()
g1["review_bucket"] = "v19p1_added_over_v16"

# Grup 2: v17p4 + v18 çok pozitif ama v19p1 hâlâ 0. Bunlar kaçan pozitif olabilir.
g2 = base[
    (base.pred_v19p1 == 0)
    & (base.pred_v17p4 == 1)
    & has_v18
    & (base.v18_score >= 0.90)
].copy()
g2["review_bucket"] = "missed_add_v17p4_1_v18_high"

# Grup 3: v16 pozitif ama v17p4 negatif + v18 çok düşük. Bunlar silinebilecek yanlış pozitif olabilir.
g3 = base[
    (base.pred_v16 == 1)
    & (base.pred_v17p4 == 0)
    & has_v18
    & (base.v18_score <= 0.10)
    & (base.v5_score < 0.80)
].copy()
g3["review_bucket"] = "possible_remove_v16_1_v17p4_0_v18_low"

# Grup 4: v13=1 ama v16=0 olmuş yerler. v16 yanlışlıkla pozitifleri silmiş olabilir.
g4 = base[
    (base.pred_v13 == 1)
    & (base.pred_v16 == 0)
    & has_v18
    & (base.v18_score >= 0.80)
].copy()
g4["review_bucket"] = "possible_restore_v13_1_v16_0_v18_high"

# Grup 5: zor kalite kontrol, modeller çok karışık.
g5 = base[
    has_v18
    & (base.v18_score.between(0.40, 0.60))
    & ((base.pred_v16 + base.pred_v17p4 + base.pred_v19p1).between(1, 2))
].copy()
g5["review_bucket"] = "uncertain_mid_v18_model_disagreement"

parts = [
    take_sample(g1, 300, "v18_score", ascending=False),
    take_sample(g2, 250, "v18_score", ascending=False),
    take_sample(g3, 250, "v18_score", ascending=True),
    take_sample(g4, 150, "v18_score", ascending=False),
    take_sample(g5, 150, "v18_score", ascending=False),
]

review = pd.concat(parts, ignore_index=True)
review = review.drop_duplicates("id").reset_index(drop=True)

print("selected rows:", len(review))
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

cols = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
for c in cols:
    if c not in items.columns:
        items[c] = ""

review = (
    review
    .merge(pairs[["id", "term_id", "item_id"]], on="id", how="left", validate="one_to_one")
    .merge(terms[["term_id", "query"]], on="term_id", how="left", validate="many_to_one")
    .merge(items[cols], on="item_id", how="left", validate="many_to_one")
)

review["assistant_label"] = ""
review["assistant_confidence"] = ""
review["needs_recheck"] = ""
review["reason"] = ""

front_cols = [
    "id", "review_bucket", "query", "title", "category", "brand", "gender", "age_group",
    "pred_v13", "pred_v16", "pred_v17p4", "pred_v19p1", "v18_score", "v5_score",
    "assistant_label", "assistant_confidence", "needs_recheck", "reason",
    "term_id", "item_id", "attributes",
]

review = review[front_cols]
review.to_csv(OUT, index=False, encoding="utf-8-sig")

summary = (
    review.groupby("review_bucket")
    .agg(
        rows=("id", "count"),
        v18_mean=("v18_score", "mean"),
        v18_min=("v18_score", "min"),
        v18_max=("v18_score", "max"),
        v5_mean=("v5_score", "mean"),
    )
    .reset_index()
)
summary.to_csv(REPORT, index=False)

print(summary.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
