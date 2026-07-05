from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")

PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
CE = ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet"
BGE = ROOT / "data/processed/v15_bge_reranker_candidate_scores.parquet"

OUT = ROOT / "data/processed/v17_query_level_features.parquet"
REPORT = ROOT / "reports/manual_review/v17_query_level_features_report.csv"

PRED_FILES = {
    "pred_v13": [
        ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
    ],
    "pred_v15": [
        ROOT / "submissions/FINAL_MAIN_v15_bge_w020_t018_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v15_bge_bge_w020_t018_GSB.csv",
    ],
    "pred_v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
}

SAFE_BRANDS = [
    "nike","adidas","puma","reebok","vans","skechers","new balance","converse",
    "apple","samsung","xiaomi","huawei","lenovo","asus","acer","hp","logitech",
    "sony","philips","dyson","beko","arcelik","bosch","siemens","tefal","fakir",
    "vestel","karaca","korkmaz","arzum","ikea","english home","madame coco",
    "cerave","la roche posay","bioderma","avene","vichy","neutrogena","nivea",
    "maybelline","loreal","l oreal","golden rose","flormar","clinique",
    "defacto","koton","lc waikiki","zara","bershka","stradivarius","mango",
    "citizen","casio","seiko","daniel klein","calvin klein","tommy hilfiger",
    "hot wheels","wella","exquise","nutri feline"
]

COLOR_WORDS = [
    "siyah","beyaz","kirmizi","kırmızı","mavi","lacivert","yesil","yeşil",
    "sari","sarı","gri","bej","kahverengi","pembe","mor","turuncu","gold",
    "gümüş","gumus","füme","fume","krem"
]

GENDER_WORDS = [
    "erkek","kadin","kadın","bayan","bay","kiz","kız","cocuk","çocuk",
    "bebek","women","woman","men","male","female"
]

def norm_series(s):
    tr = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.translate(tr)
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def make_contains(series, words):
    words = sorted(set([
        re.escape(
            str(w).lower()
            .translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
        )
        for w in words
    ]), key=len, reverse=True)
    pat = r"(?:^| )(?:%s)(?: |$)" % "|".join(words)
    return series.str.contains(pat, regex=True, na=False).astype(np.int8)

def load_pred(name, paths, ids):
    for p in paths:
        if p.exists():
            print(f"loading {name}:", p)
            d = pd.read_csv(p, usecols=["id", "prediction"])
            d["id"] = d["id"].astype(str)
            assert d["id"].reset_index(drop=True).equals(ids.reset_index(drop=True)), p
            return d["prediction"].astype(np.int8).to_numpy()
    print("missing prediction:", name)
    return np.full(len(ids), -1, dtype=np.int8)

print("loading base...")
pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
assert pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))

v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
v5["id"] = v5["id"].astype(str)

df = pairs.merge(v5, on="id", how="left", validate="one_to_one")
assert df["proba_avg"].notna().all()

print("loading ce/bge...")
ce = pd.read_parquet(CE, columns=["id", "cross_sigmoid"])
ce["id"] = ce["id"].astype(str)

bge = pd.read_parquet(BGE, columns=["id", "bge_sigmoid", "bge_blend_w020", "bge_blend_w030"])
bge["id"] = bge["id"].astype(str)

df = df.merge(ce, on="id", how="left", validate="one_to_one")
df = df.merge(bge, on="id", how="left", validate="one_to_one")

df["cross_sigmoid"] = df["cross_sigmoid"].fillna(-1.0).astype("float32")
df["bge_sigmoid"] = df["bge_sigmoid"].fillna(-1.0).astype("float32")
df["bge_blend_w020"] = df["bge_blend_w020"].fillna(-1.0).astype("float32")
df["bge_blend_w030"] = df["bge_blend_w030"].fillna(-1.0).astype("float32")

print("loading predictions...")
for name, paths in PRED_FILES.items():
    df[name] = load_pred(name, paths, df["id"])

print("query features...")
terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)
terms["query_norm"] = norm_series(terms["query"])

df = df.merge(terms[["term_id", "query", "query_norm"]], on="term_id", how="left", validate="many_to_one")

q = df["query_norm"].fillna("")
df["query_token_count"] = q.str.split().str.len().fillna(0).astype("int16")
df["query_char_len"] = q.str.len().fillna(0).astype("int16")
df["query_has_brand"] = make_contains(q, SAFE_BRANDS)
df["query_has_color"] = make_contains(q, COLOR_WORDS)
df["query_has_gender"] = make_contains(q, GENDER_WORDS)
df["query_has_digit"] = q.str.contains(r"\d", regex=True, na=False).astype(np.int8)
df["query_is_generic"] = (
    (df["query_token_count"] <= 2)
    & (df["query_has_brand"] == 0)
    & (df["query_has_digit"] == 0)
).astype(np.int8)
df["query_is_specific"] = (
    (df["query_token_count"] >= 3)
    | (df["query_has_brand"] == 1)
    | (df["query_has_digit"] == 1)
).astype(np.int8)

print("score features...")
# Mid-band dışında BGE yok. Ranking için güvenli skorlar.
df["bge_rank_score"] = np.where(
    df["bge_sigmoid"] >= 0,
    df["bge_sigmoid"],
    np.where(df["proba_avg"] > 0.80, 1.0, -0.1)
).astype("float32")

df["bge_blend_score"] = np.where(
    df["bge_blend_w020"] >= 0,
    df["bge_blend_w020"],
    np.where(df["proba_avg"] > 0.80, 0.95, 0.05)
).astype("float32")

df["ce_rank_score"] = np.where(
    df["cross_sigmoid"] >= 0,
    df["cross_sigmoid"],
    np.where(df["proba_avg"] > 0.80, 1.0, -0.1)
).astype("float32")

df["v5_high"] = (df["proba_avg"] > 0.80).astype(np.int8)
df["v5_mid"] = ((df["proba_avg"] >= 0.45) & (df["proba_avg"] <= 0.80)).astype(np.int8)
df["has_bge"] = (df["bge_sigmoid"] >= 0).astype(np.int8)
df["has_ce"] = (df["cross_sigmoid"] >= 0).astype(np.int8)

print("group ranks...")
g = df.groupby("term_id", sort=False)

df["candidate_count"] = g["id"].transform("size").astype("int16")

for col in ["proba_avg", "bge_rank_score", "bge_blend_score", "ce_rank_score"]:
    rank_col = f"{col}_rank"
    pct_col = f"{col}_pct_rank"
    max_col = f"{col}_term_max"
    delta_col = f"{col}_delta_top"

    df[rank_col] = g[col].rank(method="first", ascending=False).astype("float32")
    df[pct_col] = (df[rank_col] / df["candidate_count"]).astype("float32")
    df[max_col] = g[col].transform("max").astype("float32")
    df[delta_col] = (df[max_col] - df[col]).astype("float32")

for pred_col in ["pred_v13", "pred_v15", "pred_v16"]:
    if pred_col in df.columns:
        df[f"{pred_col}_term_sum"] = g[pred_col].transform(lambda x: (x == 1).sum()).astype("int16")
        df[f"{pred_col}_term_ratio"] = (df[f"{pred_col}_term_sum"] / df["candidate_count"]).astype("float32")

keep = [
    "id", "term_id", "item_id",
    "proba_avg", "cross_sigmoid", "bge_sigmoid", "bge_blend_w020", "bge_blend_w030",
    "bge_rank_score", "bge_blend_score", "ce_rank_score",
    "candidate_count",
    "query_token_count", "query_char_len",
    "query_has_brand", "query_has_color", "query_has_gender", "query_has_digit",
    "query_is_generic", "query_is_specific",
    "v5_high", "v5_mid", "has_bge", "has_ce",
    "pred_v13", "pred_v15", "pred_v16",
    "pred_v13_term_sum", "pred_v13_term_ratio",
    "pred_v15_term_sum", "pred_v15_term_ratio",
    "pred_v16_term_sum", "pred_v16_term_ratio",
]

for col in ["proba_avg", "bge_rank_score", "bge_blend_score", "ce_rank_score"]:
    keep += [
        f"{col}_rank", f"{col}_pct_rank",
        f"{col}_term_max", f"{col}_delta_top"
    ]

df[keep].to_parquet(OUT, index=False)

report = pd.DataFrame([{
    "rows": len(df),
    "terms": df["term_id"].nunique(),
    "mean_candidate_count": float(df["candidate_count"].mean()),
    "pred_v13_ones": int((df["pred_v13"] == 1).sum()),
    "pred_v15_ones": int((df["pred_v15"] == 1).sum()),
    "pred_v16_ones": int((df["pred_v16"] == 1).sum()),
    "query_has_brand_ratio": float(df["query_has_brand"].mean()),
    "query_is_generic_ratio": float(df["query_is_generic"].mean()),
}])
report.to_csv(REPORT, index=False)

print(report.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
