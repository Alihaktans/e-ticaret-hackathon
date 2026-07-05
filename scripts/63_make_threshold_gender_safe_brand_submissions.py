from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT_SUMMARY = ROOT / "reports/manual_review/v5_threshold_gender_safe_brand_submission_summary.csv"

THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80]

SAFE_BRANDS = [
    "nike", "adidas", "puma", "reebok", "vans", "skechers", "new balance", "converse",
    "apple", "samsung", "xiaomi", "huawei", "lenovo", "asus", "acer", "hp", "logitech",
    "sony", "philips", "dyson", "beko", "arcelik", "bosch", "siemens", "tefal", "fakir",
    "vestel", "karaca", "korkmaz", "arzum",
    "ikea", "english home", "madame coco", "schafer",
    "cerave", "la roche posay", "bioderma", "avene", "vichy", "neutrogena", "nivea",
    "maybelline", "loreal", "l oreal", "golden rose", "flormar", "clinique",
    "defacto", "koton", "lc waikiki", "zara", "bershka", "stradivarius", "mango",
    "pull bear", "jack jones",
    "citizen", "casio", "seiko", "daniel klein", "calvin klein", "tommy hilfiger",
    "hot wheels",
]

ALIASES = {
    "apple": ["iphone", "ipad", "macbook", "airpods", "ios"],
    "samsung": ["galaxy"],
    "xiaomi": ["redmi", "poco"],
    "nike": ["jordan", "air max"],
    "loreal": ["l oreal"],
    "l oreal": ["loreal"],
    "la roche posay": ["roche posay"],
    "hot wheels": ["hotwheels"],
}

COMPAT_WORDS = [
    "uyumlu", "kilif", "kılıf", "sarj", "şarj", "kablo", "adapter", "adaptor",
    "adaptör", "ekran koruyucu", "koruyucu cam", "yedek parca", "yedek parça",
    "kapak", "kayis", "kayış", "telefon aksesuar", "tablet aksesuar"
]

INTRINSIC_WORDS = [
    "ayakkabi", "ayakkabı", "bot", "terlik", "sneaker", "tisort", "t shirt",
    "sweatshirt", "ceket", "mont", "gomlek", "gömlek", "pantolon", "elbise",
    "canta", "çanta", "saat", "parfum", "parfüm", "krem", "serum", "sampuan",
    "şampuan", "ruj", "maskara", "fondoten", "oyuncak", "bebek", "dolap",
    "masa", "sandalye", "hali", "halı", "nevresim", "forma"
]

def norm_str(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

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

def word_contains(series, words):
    words = [re.escape(norm_str(w)) for w in words if norm_str(w)]
    if not words:
        return pd.Series(False, index=series.index)
    pat = r"(?:^| )(?:%s)(?: |$)" % "|".join(words)
    return series.str.contains(pat, regex=True, na=False)

def any_contains(series, words):
    words = [re.escape(norm_str(w)) for w in words if norm_str(w)]
    if not words:
        return pd.Series(False, index=series.index)
    return series.str.contains("|".join(words), regex=True, na=False)

SAFE_BRANDS = sorted(set(norm_str(x) for x in SAFE_BRANDS), key=len, reverse=True)
ALIASES = {norm_str(k): [norm_str(v) for v in vals] for k, vals in ALIASES.items()}

scores = pd.read_parquet(SCORE, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

sample = pd.read_csv(SAMPLE)
assert scores["id"].astype(str).reset_index(drop=True).equals(
    sample["id"].astype(str).reset_index(drop=True)
)

pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
assert pairs["id"].astype(str).reset_index(drop=True).equals(
    scores["id"].astype(str).reset_index(drop=True)
)

# En düşük threshold 0.50 olduğu için sadece bu adayların metadata'sını hazırlıyoruz.
base_mask_050 = scores["proba_avg"].to_numpy() >= min(THRESHOLDS)
pos_idx = np.flatnonzero(base_mask_050)

pos = pairs.iloc[pos_idx].copy()
pos["term_id"] = pos["term_id"].astype(str)
pos["item_id"] = pos["item_id"].astype(str)

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

q_norm = norm_series(pos["query"])
item_brand_norm = norm_series(pos["brand"])

item_text_norm = norm_series(
    pos["title"].fillna("").astype(str) + " " +
    pos["category"].fillna("").astype(str) + " " +
    pos["attributes"].fillna("").astype(str) + " " +
    pos["brand"].fillna("").astype(str)
)

item_gender_text_norm = norm_series(
    pos["gender"].fillna("").astype(str) + " " +
    pos["title"].fillna("").astype(str) + " " +
    pos["category"].fillna("").astype(str)
)

q_male = word_contains(q_norm, ["erkek", "bay", "men", "male"])
q_female = word_contains(q_norm, ["kadin", "bayan", "kiz", "women", "woman", "female"])
item_male = word_contains(item_gender_text_norm, ["erkek", "bay", "men", "male"])
item_female = word_contains(item_gender_text_norm, ["kadin", "bayan", "kiz", "women", "woman", "female"])

gender_mismatch = (
    (q_male & ~q_female & item_female & ~item_male)
    |
    (q_female & ~q_male & item_male & ~item_female)
)

safe_brand_mismatch = pd.Series(False, index=pos.index)

for brand in SAFE_BRANDS:
    q_has = word_contains(q_norm, [brand])
    if not q_has.any():
        continue

    item_brand_match = (
        item_brand_norm.eq(brand)
        | item_brand_norm.str.contains(re.escape(brand), regex=True, na=False)
    )
    text_match = word_contains(item_text_norm, [brand] + ALIASES.get(brand, []))

    safe_brand_mismatch = safe_brand_mismatch | (q_has & ~item_brand_match & ~text_match)

has_compat = any_contains(q_norm + " " + item_text_norm, COMPAT_WORDS)
has_intrinsic = any_contains(q_norm + " " + item_text_norm, INTRINSIC_WORDS)

safe_brand_filter = safe_brand_mismatch & has_intrinsic & ~has_compat
combined_filter_050_universe = gender_mismatch | safe_brand_filter

rows = []

for th in THRESHOLDS:
    pred = (scores["proba_avg"].to_numpy() >= th).astype(np.int8)

    # Bu threshold'un pozitifleri, 0.50 pozitif indexleri içinde subset'tir.
    th_inside_050 = scores["proba_avg"].to_numpy()[pos_idx] >= th
    flip_inside = combined_filter_050_universe.to_numpy() & th_inside_050
    flip_indices = pos_idx[flip_inside]

    pred[flip_indices] = 0

    tag = str(th).replace(".", "p")
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v5_full900_threshold_{tag}_GENDER_SAFE_BRAND_FILTERED.csv"
    pd.DataFrame({"id": scores["id"], "prediction": pred}).to_csv(out, index=False)

    rows.append({
        "threshold": th,
        "file": str(out),
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "flipped_1_to_0": int(len(flip_indices)),
        "id_order_ok": bool(pd.Series(scores["id"]).astype(str).reset_index(drop=True).equals(sample["id"].astype(str).reset_index(drop=True))),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_SUMMARY, index=False)

print(summary.to_string(index=False))
print("saved summary:", OUT_SUMMARY)
