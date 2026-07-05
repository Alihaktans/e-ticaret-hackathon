from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V21 = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"

OUT_SCORE = ROOT / "data/processed/v26_sparse_text_scores.parquet"
OUT_SUMMARY = ROOT / "reports/manual_review/v26_sparse_text_summary.csv"
OUT_FULL = ROOT / "reports/manual_review/v26_sparse_candidates_summary.csv"

PATHS = {
    "vote_base": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "v24_best": [
        ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ],
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
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
            raise FileNotFoundError(name)
        return None
    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    assert d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), name
    return d["prediction"].astype(np.int8).to_numpy()

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def row_dot(A, B, chunk=400000):
    out = np.zeros(A.shape[0], dtype=np.float32)
    for s in range(0, A.shape[0], chunk):
        e = min(s + chunk, A.shape[0])
        out[s:e] = np.asarray(A[s:e].multiply(B[s:e]).sum(axis=1)).ravel().astype(np.float32)
        print("dot", e, "/", A.shape[0])
    return out

def clean_name(x):
    return str(x).replace(".", "p").replace(" ", "_")

print("loading sample/pairs...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)

pairs = pd.read_csv(PAIRS)
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

assert pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))

terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)
terms["query_clean"] = terms["query"].map(clean_text)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)

for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

items["title_clean"] = items["title"].map(clean_text)
items["category_clean"] = items["category"].map(clean_text)
items["brand_clean"] = items["brand"].map(clean_text)
items["attr_clean"] = items["attributes"].map(clean_text)
items["gender_clean"] = items["gender"].map(clean_text)
items["age_clean"] = items["age_group"].map(clean_text)

# Title daha önemli, brand/category yardımcı.
items["item_text"] = (
    items["title_clean"] + " " + items["title_clean"] + " " +
    items["brand_clean"] + " " +
    items["category_clean"] + " " +
    items["gender_clean"] + " " +
    items["age_clean"] + " " +
    items["attr_clean"].str.slice(0, 350)
).str.strip()

print("mapping rows...")
term_pos = pd.Series(np.arange(len(terms)), index=terms["term_id"])
item_pos = pd.Series(np.arange(len(items)), index=items["item_id"])

term_idx = pairs["term_id"].map(term_pos).astype(np.int32).to_numpy()
item_idx = pairs["item_id"].map(item_pos).astype(np.int32).to_numpy()

print("vectorizing word...")
word_vec = HashingVectorizer(
    n_features=2**20,
    alternate_sign=False,
    norm="l2",
    analyzer="word",
    ngram_range=(1, 2),
    lowercase=False,
)

Qw_all = word_vec.transform(terms["query_clean"].fillna("").tolist())
Iw_all = word_vec.transform(items["item_text"].fillna("").tolist())

Qw = Qw_all[term_idx]
Iw = Iw_all[item_idx]

word_score = row_dot(Qw, Iw)

print("vectorizing char...")
char_vec = HashingVectorizer(
    n_features=2**21,
    alternate_sign=False,
    norm="l2",
    analyzer="char_wb",
    ngram_range=(3, 5),
    lowercase=False,
)

Qc_all = char_vec.transform(terms["query_clean"].fillna("").tolist())
Ic_all = char_vec.transform(items["item_text"].fillna("").tolist())

Qc = Qc_all[term_idx]
Ic = Ic_all[item_idx]

char_score = row_dot(Qc, Ic)

score = pd.DataFrame({
    "id": sample["id"].to_numpy(),
    "term_id": pairs["term_id"].to_numpy(),
    "item_id": pairs["item_id"].to_numpy(),
    "sparse_word": word_score,
    "sparse_char": char_score,
})

score["sparse_blend"] = (
    0.55 * score["sparse_word"].astype("float32") +
    0.45 * score["sparse_char"].astype("float32")
).astype("float32")

print("adding per-term ranks...")
score["candidate_count"] = score.groupby("term_id")["id"].transform("count").astype(np.int32)
score["sparse_rank"] = score.groupby("term_id")["sparse_blend"].rank(method="first", ascending=False).astype(np.int32)
score["sparse_pct_rank"] = (score["sparse_rank"] / score["candidate_count"]).astype("float32")
score["sparse_term_max"] = score.groupby("term_id")["sparse_blend"].transform("max").astype("float32")
score["sparse_delta_from_max"] = (score["sparse_term_max"] - score["sparse_blend"]).astype("float32")

score.to_parquet(OUT_SCORE, index=False)
print("saved score:", OUT_SCORE)

summary = score[["sparse_word", "sparse_char", "sparse_blend", "sparse_pct_rank"]].describe().T
summary.to_csv(OUT_SUMMARY)
print(summary)

print("loading anchors...")
pred_base = load_sub("vote_base", sample)
pred_v24 = load_sub("v24_best", sample)
pred_v20 = load_sub("v20", sample, required=False)
if pred_v20 is None:
    pred_v20 = pred_base.copy()

n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

print("loading other model scores...")
v21_score = np.zeros(n, dtype=np.float32)
if V21.exists():
    v21 = pd.read_parquet(V21)
    v21["id"] = v21["id"].astype(str)
    assert v21["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
    v21_score = v21["v21_score"].astype("float32").to_numpy()

v5_score = np.zeros(n, dtype=np.float32)
if V5.exists():
    v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
    v5["id"] = v5["id"].astype(str)
    assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
    v5_score = v5["proba_avg"].astype("float32").to_numpy()

v18_score = np.zeros(n, dtype=np.float32)
if V18.exists():
    v18 = pd.read_parquet(V18)
    v18["id"] = v18["id"].astype(str)
    idx = v18["id"].map(id_to_idx)
    ok = idx.notna()
    v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()

score["pred_base"] = pred_base
score["pred_v24"] = pred_v24
score["pred_v20"] = pred_v20
score["v21_score"] = v21_score
score["v5_score"] = v5_score
score["v18_score"] = v18_score

# Yeni sinyal ağırlıklı add skoru.
# Sparse ana sinyal; diğerleri sadece güvenlik filtresi.
score["add_score"] = (
    1.00 * score["sparse_blend"].astype("float32") +
    0.15 * score["v5_score"].astype("float32") +
    0.15 * score["v18_score"].astype("float32") +
    0.10 * score["v21_score"].astype("float32") -
    0.10 * score["sparse_pct_rank"].astype("float32")
).astype("float32")

# v24'ün 0 dediği ama sparse'ın güçlü dediği yerler.
add_pool = score[
    (score["pred_v24"] == 0)
    & (score["sparse_pct_rank"] <= 0.08)
    & (score["sparse_blend"] >= 0.18)
].copy()

# Çok zayıf semantic çelişkiyi engelle.
add_pool = add_pool[
    ~(
        (add_pool["v5_score"] < 0.05)
        & (add_pool["v18_score"] < 0.05)
        & (add_pool["v21_score"] < 0.05)
    )
].copy()

add_pool = add_pool.sort_values("add_score", ascending=False)

print("add_pool:", len(add_pool))

variants = {}

def save_variant(name, pred):
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v26_{clean_name(name)}.csv"
    pd.DataFrame({"id": sample["id"].to_numpy(), "prediction": pred.astype(np.int8)}).to_csv(out, index=False)
    variants[name] = pred.astype(np.int8)
    print("saved", out, "ones", int(pred.sum()), "pos_ratio", float(pred.mean()), "diff_v24", int((pred != pred_v24).sum()))

# Baselines
save_variant("v24_anchor", pred_v24)

for budget in [500, 1000, 2000, 3000, 5000, 8000, 12000, 16000, 24000]:
    pred = pred_v24.copy()
    take = add_pool.head(budget)
    idx = take.index.to_numpy()
    pred[idx] = 1
    save_variant(f"sparse_addonly_budget{budget}", pred)

# Daha sıkı: sadece term içi ilk %3
strict_pool = add_pool[add_pool["sparse_pct_rank"] <= 0.03].copy()
for budget in [500, 1000, 2000, 3000, 5000, 8000]:
    pred = pred_v24.copy()
    take = strict_pool.head(budget)
    idx = take.index.to_numpy()
    pred[idx] = 1
    save_variant(f"sparse_addonly_strictpct3_budget{budget}", pred)

# Daha kaliteli: sparse yüksek + en az bir eski model destekliyor
supported_pool = add_pool[
    (add_pool["sparse_pct_rank"] <= 0.05)
    & (
        (add_pool["v5_score"] >= 0.25)
        | (add_pool["v18_score"] >= 0.25)
        | (add_pool["v21_score"] >= 0.80)
        | (add_pool["pred_v20"] == 1)
    )
].copy()

for budget in [500, 1000, 2000, 3000, 5000, 8000, 12000]:
    pred = pred_v24.copy()
    take = supported_pool.head(budget)
    idx = take.index.to_numpy()
    pred[idx] = 1
    save_variant(f"sparse_addonly_supported_budget{budget}", pred)

rows = []
for name, pred in variants.items():
    rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
        "diff_vs_base": int((pred != pred_base).sum()),
        "diff_vs_v20": int((pred != pred_v20).sum()),
    })

full = pd.DataFrame(rows)
full.to_csv(OUT_FULL, index=False)
print(full.to_string(index=False))
print("saved:", OUT_FULL)
