from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")

PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT = ROOT / "data/processed/v17p4_independent_lexical_features.parquet"
REPORT = ROOT / "reports/manual_review/v17p4_independent_lexical_features_report.csv"

SAFE_BRANDS = [
    "nike","adidas","puma","reebok","vans","skechers","new balance","converse",
    "apple","samsung","xiaomi","huawei","lenovo","asus","acer","hp","logitech",
    "sony","philips","dyson","beko","arcelik","arçelik","bosch","siemens","tefal","fakir",
    "vestel","karaca","korkmaz","arzum","ikea","english home","madame coco",
    "cerave","la roche posay","bioderma","avene","vichy","neutrogena","nivea",
    "maybelline","loreal","l oreal","l'oréal","golden rose","flormar","clinique",
    "defacto","koton","lc waikiki","zara","bershka","stradivarius","mango",
    "citizen","casio","seiko","daniel klein","calvin klein","tommy hilfiger",
    "hot wheels","wella","exquise","nutri feline"
]

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def toks(x):
    if not x:
        return set()
    return set(x.split())

BRANDS_NORM = sorted(set(norm(x) for x in SAFE_BRANDS if norm(x)), key=len, reverse=True)

def extract_brand(q):
    if not q:
        return ""
    padded = " " + q + " "
    for b in BRANDS_NORM:
        if " " + b + " " in padded:
            return b
    return ""

def gender_type(x):
    s = norm(x)
    if any(w in s.split() for w in ["erkek", "bay", "men", "man", "male"]):
        return 1
    if any(w in s.split() for w in ["kadin", "bayan", "women", "woman", "female"]):
        return 2
    if any(w in s.split() for w in ["cocuk", "kiz", "oglan", "bebek", "baby", "kids", "kid"]):
        return 3
    return 0

print("loading...")
pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

terms = pd.read_csv(TERMS, usecols=["term_id", "query"])
terms["term_id"] = terms["term_id"].astype(str)
terms["query_norm"] = terms["query"].map(norm)
terms["query_tokens"] = terms["query_norm"].map(toks)
terms["query_brand"] = terms["query_norm"].map(extract_brand)
terms["query_gender_type"] = terms["query_norm"].map(gender_type).astype("int8")
terms["query_token_len"] = terms["query_tokens"].map(len).astype("int16")

needed_items = set(pairs["item_id"].unique())

items = pd.read_csv(
    ITEMS,
    usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    low_memory=False,
)
items["item_id"] = items["item_id"].astype(str)
items = items[items["item_id"].isin(needed_items)].copy()

items["title_norm"] = items["title"].map(norm)
items["category_norm"] = items["category"].map(norm)
items["brand_norm"] = items["brand"].map(norm)
items["gender_norm"] = items["gender"].map(norm)

items["title_tokens"] = items["title_norm"].map(toks)
items["category_tokens"] = items["category_norm"].map(toks)
items["item_gender_type"] = items["gender_norm"].map(gender_type).astype("int8")
items["item_title_len"] = items["title_tokens"].map(len).astype("int16")
items["item_category_len"] = items["category_tokens"].map(len).astype("int16")
items["item_brand_empty"] = (items["brand_norm"].str.len() == 0).astype("int8")

term_map = terms.set_index("term_id")[[
    "query_tokens", "query_brand", "query_gender_type", "query_token_len"
]].to_dict("index")

item_map = items.set_index("item_id")[[
    "title_tokens", "category_tokens", "brand_norm",
    "item_gender_type", "item_title_len", "item_category_len", "item_brand_empty"
]].to_dict("index")

print("computing pair lexical features...")

n = len(pairs)

title_overlap = np.zeros(n, dtype=np.int16)
cat_overlap = np.zeros(n, dtype=np.int16)
title_cov = np.zeros(n, dtype=np.float32)
cat_cov = np.zeros(n, dtype=np.float32)
title_jacc = np.zeros(n, dtype=np.float32)
cat_jacc = np.zeros(n, dtype=np.float32)

brand_in_query = np.zeros(n, dtype=np.int8)
brand_match = np.zeros(n, dtype=np.int8)
brand_mismatch = np.zeros(n, dtype=np.int8)

gender_match = np.zeros(n, dtype=np.int8)
gender_mismatch = np.zeros(n, dtype=np.int8)

item_title_len = np.zeros(n, dtype=np.int16)
item_category_len = np.zeros(n, dtype=np.int16)
item_brand_empty = np.ones(n, dtype=np.int8)

term_ids = pairs["term_id"].to_numpy()
item_ids = pairs["item_id"].to_numpy()

for i, (tid, iid) in enumerate(zip(term_ids, item_ids)):
    tm = term_map.get(tid)
    im = item_map.get(iid)

    if tm is None or im is None:
        continue

    qt = tm["query_tokens"]
    tt = im["title_tokens"]
    ct = im["category_tokens"]

    qlen = max(1, int(tm["query_token_len"]))
    to = len(qt & tt)
    co = len(qt & ct)

    title_overlap[i] = to
    cat_overlap[i] = co
    title_cov[i] = to / qlen
    cat_cov[i] = co / qlen

    tu = len(qt | tt)
    cu = len(qt | ct)
    title_jacc[i] = to / max(1, tu)
    cat_jacc[i] = co / max(1, cu)

    qbrand = tm["query_brand"]
    ibrand = im["brand_norm"]

    if qbrand:
        brand_in_query[i] = 1
        if qbrand in ibrand or qbrand in " ".join(tt):
            brand_match[i] = 1
        else:
            brand_mismatch[i] = 1

    qg = int(tm["query_gender_type"])
    ig = int(im["item_gender_type"])

    if qg != 0 and ig != 0:
        if qg == ig:
            gender_match[i] = 1
        else:
            gender_mismatch[i] = 1

    item_title_len[i] = int(im["item_title_len"])
    item_category_len[i] = int(im["item_category_len"])
    item_brand_empty[i] = int(im["item_brand_empty"])

    if i % 500000 == 0 and i > 0:
        print("processed", i, "/", n)

out = pd.DataFrame({
    "id": pairs["id"].to_numpy(),
    "title_overlap": title_overlap,
    "cat_overlap": cat_overlap,
    "title_cov": title_cov,
    "cat_cov": cat_cov,
    "title_jacc": title_jacc,
    "cat_jacc": cat_jacc,
    "brand_in_query": brand_in_query,
    "brand_match": brand_match,
    "brand_mismatch": brand_mismatch,
    "gender_match": gender_match,
    "gender_mismatch": gender_mismatch,
    "item_title_len": item_title_len,
    "item_category_len": item_category_len,
    "item_brand_empty": item_brand_empty,
})

out.to_parquet(OUT, index=False)

report = pd.DataFrame([{
    "rows": len(out),
    "title_cov_mean": float(out["title_cov"].mean()),
    "cat_cov_mean": float(out["cat_cov"].mean()),
    "brand_in_query_ratio": float(out["brand_in_query"].mean()),
    "brand_match_ratio": float(out["brand_match"].mean()),
    "brand_mismatch_ratio": float(out["brand_mismatch"].mean()),
    "gender_mismatch_ratio": float(out["gender_mismatch"].mean()),
}])
report.to_csv(REPORT, index=False)

print(report.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
