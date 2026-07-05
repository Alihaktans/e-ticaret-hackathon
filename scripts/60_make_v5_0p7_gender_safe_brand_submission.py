from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
BASE_SUB = ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_RANDOM_VALIDATED.csv"
SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT = ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_GENDER_SAFE_BRAND_FILTERED.csv"
REPORT = ROOT / "reports/manual_review/v5_0p7_gender_safe_brand_full_report.csv"
FLIPPED = ROOT / "reports/manual_review/v5_0p7_gender_safe_brand_flipped_rows.csv"

SAFE_BRANDS = [
    "nike", "adidas", "puma", "reebok", "vans", "skechers", "new balance", "converse",
    "under armour", "columbia", "lacoste",
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

def word_contains_norm(series, words):
    words = [re.escape(norm_str(w)) for w in words if norm_str(w)]
    if not words:
        return pd.Series(False, index=series.index)
    pat = r"(?:^| )(?:%s)(?: |$)" % "|".join(words)
    return series.str.contains(pat, regex=True, na=False)

def any_contains_norm(series, words):
    words = [re.escape(norm_str(w)) for w in words if norm_str(w)]
    if not words:
        return pd.Series(False, index=series.index)
    pat = "|".join(words)
    return series.str.contains(pat, regex=True, na=False)

SAFE_BRANDS = sorted(set(norm_str(x) for x in SAFE_BRANDS), key=len, reverse=True)
ALIASES = {norm_str(k): [norm_str(v) for v in vals] for k, vals in ALIASES.items()}

base = pd.read_csv(BASE_SUB)
sample = pd.read_csv(SAMPLE)

assert base["id"].astype(str).reset_index(drop=True).equals(
    sample["id"].astype(str).reset_index(drop=True)
), "Base submission id order sample_submission ile aynı değil."

pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
assert pairs["id"].astype(str).reset_index(drop=True).equals(
    base["id"].astype(str).reset_index(drop=True)
), "submission_pairs id order base ile aynı değil."

pred = base["prediction"].to_numpy().astype(np.int8)
pos_idx = np.flatnonzero(pred == 1)

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
item_text_raw = (
    pos["title"].fillna("").astype(str) + " " +
    pos["category"].fillna("").astype(str) + " " +
    pos["attributes"].fillna("").astype(str) + " " +
    pos["brand"].fillna("").astype(str)
)
item_text_norm = norm_series(item_text_raw)

# gender mismatch
item_gender_text_norm = norm_series(
    pos["gender"].fillna("").astype(str) + " " +
    pos["title"].fillna("").astype(str) + " " +
    pos["category"].fillna("").astype(str)
)

male_words = ["erkek", "bay", "men", "male"]
female_words = ["kadin", "bayan", "kiz", "women", "woman", "female"]

q_male = word_contains_norm(q_norm, male_words)
q_female = word_contains_norm(q_norm, female_words)
item_male = word_contains_norm(item_gender_text_norm, male_words)
item_female = word_contains_norm(item_gender_text_norm, female_words)

gender_mismatch = (
    (q_male & ~q_female & item_female & ~item_male)
    |
    (q_female & ~q_male & item_male & ~item_female)
)

# strict brand mismatch
safe_brand_mismatch = pd.Series(False, index=pos.index)

for brand in SAFE_BRANDS:
    q_has = word_contains_norm(q_norm, [brand])
    if not q_has.any():
        continue

    item_brand_match = (
        item_brand_norm.eq(brand)
        | item_brand_norm.str.contains(re.escape(brand), regex=True, na=False)
    )

    brand_text_match = word_contains_norm(item_text_norm, [brand] + ALIASES.get(brand, []))

    flag = q_has & ~item_brand_match & ~brand_text_match
    safe_brand_mismatch = safe_brand_mismatch | flag

has_compat = any_contains_norm(q_norm + " " + item_text_norm, COMPAT_WORDS)
has_intrinsic = any_contains_norm(q_norm + " " + item_text_norm, INTRINSIC_WORDS)

safe_brand_filter = safe_brand_mismatch & has_intrinsic & ~has_compat

combined_filter = gender_mismatch | safe_brand_filter

flip_indices = pos_idx[combined_filter.to_numpy()]
pred[flip_indices] = 0

out = pd.DataFrame({
    "id": base["id"],
    "prediction": pred,
})
out.to_csv(OUT, index=False)

flipped = pos.loc[combined_filter].copy()
flipped["reason_gender_mismatch"] = gender_mismatch[combined_filter].to_numpy()
flipped["reason_safe_brand_filter"] = safe_brand_filter[combined_filter].to_numpy()
flipped["old_prediction"] = 1
flipped["new_prediction"] = 0
flipped.to_csv(FLIPPED, index=False, encoding="utf-8-sig")

report = pd.DataFrame([{
    "base_file": str(BASE_SUB),
    "out_file": str(OUT),
    "rows": len(out),
    "base_ones": int(base["prediction"].sum()),
    "new_ones": int(pred.sum()),
    "flipped_1_to_0_total": int(len(flip_indices)),
    "gender_flips": int(gender_mismatch.sum()),
    "safe_brand_flips": int(safe_brand_filter.sum()),
    "overlap_flips": int((gender_mismatch & safe_brand_filter).sum()),
    "base_pos_ratio": float(base["prediction"].mean()),
    "new_pos_ratio": float(pred.mean()),
    "id_order_ok": bool(out["id"].astype(str).reset_index(drop=True).equals(sample["id"].astype(str).reset_index(drop=True))),
}])
report.to_csv(REPORT, index=False)

print(report.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
print("flipped rows:", FLIPPED)

if int(safe_brand_filter.sum()) > 80000:
    print("WARNING: safe_brand_filter çok fazla satır çevirdi. Submit etmeden önce flipped rows dosyasını incele.")
