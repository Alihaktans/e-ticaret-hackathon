from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")

V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
BGE = ROOT / "data/processed/v15_bge_reranker_candidate_scores.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V13_MAIN = ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv"
V5_MAIN = ROOT / "submissions/FINAL_MAIN_v5_full900_0p6_GENDER_SAFE_BRAND.csv"

OUT_SUMMARY = ROOT / "reports/manual_review/v15_bge_submission_summary.csv"
FLIPPED_OUT = ROOT / "reports/manual_review/v15_bge_gsb_flipped_rows.csv"

LOW = 0.45
HIGH = 0.80

VARIANTS = [
    ("bge_w020_t018", 0.20, 0.18),
    ("bge_w040_t032", 0.40, 0.32),
    ("bge_w030_t030", 0.30, 0.30),
    ("bge_w030_t034", 0.30, 0.34),
    ("bge_w020_t020", 0.20, 0.20),
    ("bge_w030_t026", 0.30, 0.26),
]

SAFE_BRANDS = [
    "nike","adidas","puma","reebok","vans","skechers","new balance","converse",
    "apple","samsung","xiaomi","huawei","lenovo","asus","acer","hp","logitech",
    "sony","philips","dyson","beko","arcelik","bosch","siemens","tefal","fakir",
    "vestel","karaca","korkmaz","arzum","ikea","english home","madame coco",
    "cerave","la roche posay","bioderma","avene","vichy","neutrogena","nivea",
    "maybelline","loreal","l oreal","golden rose","flormar","clinique",
    "defacto","koton","lc waikiki","zara","bershka","stradivarius","mango",
    "citizen","casio","seiko","daniel klein","calvin klein","tommy hilfiger",
    "hot wheels",
]

ALIASES = {
    "apple":["iphone","ipad","macbook","airpods","ios"],
    "samsung":["galaxy"],
    "xiaomi":["redmi","poco"],
    "nike":["jordan","air max"],
    "loreal":["l oreal"],
    "l oreal":["loreal"],
    "la roche posay":["roche posay"],
    "hot wheels":["hotwheels"],
}

COMPAT_WORDS = [
    "uyumlu","kilif","kılıf","sarj","şarj","kablo","adapter","adaptor",
    "adaptör","ekran koruyucu","koruyucu cam","yedek parca","yedek parça",
    "kapak","kayis","kayış"
]

INTRINSIC_WORDS = [
    "ayakkabi","ayakkabı","bot","terlik","sneaker","tisort","t shirt",
    "sweatshirt","ceket","mont","gomlek","gömlek","pantolon","elbise",
    "canta","çanta","saat","parfum","parfüm","krem","serum","sampuan",
    "şampuan","ruj","maskara","fondoten","oyuncak","bebek","dolap",
    "masa","sandalye","hali","halı","nevresim","forma"
]

def norm(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_series(s):
    tr = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    return (
        s.fillna("").astype(str).str.lower().str.translate(tr)
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

def word_contains(series, words):
    words = [re.escape(norm(w)) for w in words if norm(w)]
    if not words:
        return pd.Series(False, index=series.index)
    pat = r"(?:^| )(?:%s)(?: |$)" % "|".join(words)
    return series.str.contains(pat, regex=True, na=False)

def any_contains(series, words):
    words = [re.escape(norm(w)) for w in words if norm(w)]
    if not words:
        return pd.Series(False, index=series.index)
    return series.str.contains("|".join(words), regex=True, na=False)

SAFE_BRANDS = sorted(set(norm(x) for x in SAFE_BRANDS), key=len, reverse=True)
ALIASES = {norm(k): [norm(v) for v in vals] for k, vals in ALIASES.items()}

print("loading scores...")
v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
v5["id"] = v5["id"].astype(str)

bge = pd.read_parquet(BGE, columns=["id", "bge_sigmoid"])
bge["id"] = bge["id"].astype(str)

sample = pd.read_csv(SAMPLE)
sample["id"] = sample["id"].astype(str)

assert v5["id"].astype(str).reset_index(drop=True).equals(sample["id"].reset_index(drop=True))

df = v5.merge(bge, on="id", how="left", validate="one_to_one")

v5_score = df["proba_avg"].to_numpy()
bge_score = df["bge_sigmoid"].to_numpy()

preds_before_filter = {}

for name, w, th in VARIANTS:
    pred = np.zeros(len(df), dtype=np.int8)

    high = v5_score > HIGH
    mid = (v5_score >= LOW) & (v5_score <= HIGH) & ~np.isnan(bge_score)

    pred[high] = 1

    blend = w * v5_score[mid] + (1.0 - w) * bge_score[mid]
    pred[mid] = (blend >= th).astype(np.int8)

    preds_before_filter[name] = pred

union_pos = np.zeros(len(df), dtype=bool)
for pred in preds_before_filter.values():
    union_pos |= pred.astype(bool)

union_idx = np.flatnonzero(union_pos)
print("union positive rows before filter:", len(union_idx))

pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

assert pairs["id"].reset_index(drop=True).equals(df["id"].reset_index(drop=True))

pos = pairs.iloc[union_idx].copy()
pos["row_pos"] = union_idx

terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(
    ITEMS,
    usecols=["item_id", "title", "category", "brand", "gender", "attributes"],
    low_memory=False,
)
items["item_id"] = items["item_id"].astype(str)

pos = pos.merge(terms[["term_id", "query"]], on="term_id", how="left", validate="many_to_one")
pos = pos.merge(items, on="item_id", how="left", validate="many_to_one")

q = norm_series(pos["query"])
item_gender_text = norm_series(
    pos["gender"].fillna("").astype(str) + " " +
    pos["title"].fillna("").astype(str) + " " +
    pos["category"].fillna("").astype(str)
)
item_text = norm_series(
    pos["title"].fillna("").astype(str) + " " +
    pos["category"].fillna("").astype(str) + " " +
    pos["attributes"].fillna("").astype(str) + " " +
    pos["brand"].fillna("").astype(str)
)
item_brand = norm_series(pos["brand"].fillna("").astype(str))

q_male = word_contains(q, ["erkek", "bay", "men", "male"])
q_female = word_contains(q, ["kadin", "bayan", "kiz", "women", "woman", "female"])
item_male = word_contains(item_gender_text, ["erkek", "bay", "men", "male"])
item_female = word_contains(item_gender_text, ["kadin", "bayan", "kiz", "women", "woman", "female"])

gender_mismatch = (
    (q_male & ~q_female & item_female & ~item_male)
    |
    (q_female & ~q_male & item_male & ~item_female)
)

safe_brand_mismatch = pd.Series(False, index=pos.index)

for brand in SAFE_BRANDS:
    q_has = word_contains(q, [brand])
    if not q_has.any():
        continue

    item_brand_match = (
        item_brand.eq(brand)
        | item_brand.str.contains(re.escape(brand), regex=True, na=False)
    )
    text_match = word_contains(item_text, [brand] + ALIASES.get(brand, []))
    safe_brand_mismatch = safe_brand_mismatch | (q_has & ~item_brand_match & ~text_match)

has_compat = any_contains(q + " " + item_text, COMPAT_WORDS)
has_intrinsic = any_contains(q + " " + item_text, INTRINSIC_WORDS)

safe_brand_filter = safe_brand_mismatch & has_intrinsic & ~has_compat
combined_filter = gender_mismatch | safe_brand_filter

filter_positions = pos.loc[combined_filter, "row_pos"].to_numpy()

flipped = pos.loc[combined_filter].copy()
flipped["reason_gender_mismatch"] = gender_mismatch[combined_filter].to_numpy()
flipped["reason_safe_brand_filter"] = safe_brand_filter[combined_filter].to_numpy()
flipped.to_csv(FLIPPED_OUT, index=False, encoding="utf-8-sig")

v13 = None
if V13_MAIN.exists():
    v13 = pd.read_csv(V13_MAIN, usecols=["id", "prediction"])
    v13["id"] = v13["id"].astype(str)
    assert v13["id"].reset_index(drop=True).equals(df["id"].reset_index(drop=True))
    v13_pred = v13["prediction"].astype(np.int8).to_numpy()
else:
    v13_pred = None

v5main = None
if V5_MAIN.exists():
    v5main = pd.read_csv(V5_MAIN, usecols=["id", "prediction"])
    v5main["id"] = v5main["id"].astype(str)
    assert v5main["id"].reset_index(drop=True).equals(df["id"].reset_index(drop=True))
    v5main_pred = v5main["prediction"].astype(np.int8).to_numpy()
else:
    v5main_pred = None

rows = []

for name, w, th in VARIANTS:
    before = preds_before_filter[name]
    pred = before.copy()
    pred[filter_positions] = 0

    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v15_bge_{name}_GSB.csv"
    pd.DataFrame({"id": df["id"], "prediction": pred}).to_csv(out, index=False)

    row = {
        "variant": name,
        "w_v5": w,
        "threshold": th,
        "file": str(out),
        "ones_before_filter": int(before.sum()),
        "ones_after_filter": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "flipped_1_to_0": int(before.sum() - pred.sum()),
        "id_order_ok": bool(df["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))),
    }

    if v13_pred is not None:
        row["diff_vs_v13"] = int((pred != v13_pred).sum())
        row["diff_vs_v13_ratio"] = float((pred != v13_pred).mean())

    if v5main_pred is not None:
        row["diff_vs_v5"] = int((pred != v5main_pred).sum())
        row["diff_vs_v5_ratio"] = float((pred != v5main_pred).mean())

    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(OUT_SUMMARY, index=False)

print(summary.to_string(index=False))
print("summary:", OUT_SUMMARY)
print("flipped rows:", FLIPPED_OUT)
