from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

LABEL_PATH = Path("reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv")
SCORE_PATH = Path("data/processed/v5_e5base_full900_test_proba.parquet")
OUT = Path("reports/manual_review/v5_0p7_safe_brand_rule_test.csv")
FLAG_OUT = Path("reports/manual_review/v5_0p7_safe_brand_flagged_rows.csv")

TH = 0.70

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

def norm(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

SAFE_BRANDS = sorted(set(norm(x) for x in SAFE_BRANDS), key=len, reverse=True)
ALIASES = {norm(k): [norm(v) for v in vals] for k, vals in ALIASES.items()}

def has_phrase(text, phrase):
    t = " " + norm(text) + " "
    p = " " + norm(phrase) + " "
    return p in t

def has_any(text, words):
    t = norm(text)
    return any(norm(w) in t for w in words)

def query_brands(q):
    qn = " " + norm(q) + " "
    hits = []
    for b in SAFE_BRANDS:
        if " " + b + " " in qn:
            hits.append(b)
    return hits

def brand_or_alias_in_text(brand, text):
    if has_phrase(text, brand):
        return True
    for a in ALIASES.get(brand, []):
        if has_phrase(text, a):
            return True
    return False

def item_brand_matches_query_brand(item_brand, q_brand):
    ib = norm(item_brand)
    qb = norm(q_brand)
    if not ib or not qb:
        return False
    if ib == qb:
        return True
    if len(ib) >= 4 and len(qb) >= 4 and (ib in qb or qb in ib):
        return True
    return False

def metric(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro"),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

df = pd.read_csv(LABEL_PATH)
df["id"] = df["id"].astype(str)
df = df[df["assistant_label"].isin([0, 1, "0", "1"])].copy()
df["assistant_label"] = df["assistant_label"].astype(int)

scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)
df = df.merge(scores, on="id", how="left", validate="many_to_one").dropna(subset=["proba_avg"])

df["base_pred"] = (df["proba_avg"] >= TH).astype(int)

item_text = (
    df.get("title", "").fillna("").astype(str) + " " +
    df.get("category", "").fillna("").astype(str) + " " +
    df.get("attributes", "").fillna("").astype(str) + " " +
    df.get("brand", "").fillna("").astype(str)
)

brand_flags = []
brand_names = []

for _, row in df.iterrows():
    q = row.get("query", "")
    item_brand = row.get("brand", "")
    text = " ".join([
        str(row.get("title", "")),
        str(row.get("category", "")),
        str(row.get("attributes", "")),
        str(row.get("brand", "")),
    ])

    hits = query_brands(q)
    flag = False
    hit_name = ""

    for qb in hits:
        if item_brand_matches_query_brand(item_brand, qb):
            continue
        if brand_or_alias_in_text(qb, text):
            continue

        flag = True
        hit_name = qb
        break

    brand_flags.append(flag)
    brand_names.append(hit_name)

df["safe_brand_mismatch"] = brand_flags
df["query_brand_hit"] = brand_names
df["has_compat_word"] = [
    has_any(str(q) + " " + str(t), COMPAT_WORDS)
    for q, t in zip(df.get("query", ""), item_text)
]
df["has_intrinsic_word"] = [
    has_any(str(q) + " " + str(t), INTRINSIC_WORDS)
    for q, t in zip(df.get("query", ""), item_text)
]

# Gender flag from previous successful rule
male_words = ["erkek", "bay", "men", "male"]
female_words = ["kadin", "bayan", "kiz", "women", "woman", "female"]

def word_contains(text, words):
    t = " " + norm(text) + " "
    return any((" " + norm(w) + " ") in t for w in words)

gender_flags = []
for _, row in df.iterrows():
    q = row.get("query", "")
    it = " ".join([str(row.get("gender", "")), str(row.get("title", "")), str(row.get("category", ""))])
    q_male = word_contains(q, male_words)
    q_female = word_contains(q, female_words)
    item_male = word_contains(it, male_words)
    item_female = word_contains(it, female_words)
    gender_flags.append((q_male and item_female) or (q_female and item_male))

df["gender_mismatch"] = gender_flags

base = df["base_pred"].copy()
gender_base = base.copy()
gender_base[(gender_base == 1) & df["gender_mismatch"]] = 0

rules = {
    "base_0p7": base.to_numpy(),
    "gender_only": gender_base.to_numpy(),
}

variants = {
    "safe_brand_all": df["safe_brand_mismatch"],
    "safe_brand_no_compat": df["safe_brand_mismatch"] & ~df["has_compat_word"],
    "safe_brand_intrinsic": df["safe_brand_mismatch"] & df["has_intrinsic_word"],
    "safe_brand_intrinsic_no_compat": df["safe_brand_mismatch"] & df["has_intrinsic_word"] & ~df["has_compat_word"],
    "safe_brand_intrinsic_no_compat_lt085": df["safe_brand_mismatch"] & df["has_intrinsic_word"] & ~df["has_compat_word"] & (df["proba_avg"] < 0.85),
    "safe_brand_intrinsic_no_compat_lt080": df["safe_brand_mismatch"] & df["has_intrinsic_word"] & ~df["has_compat_word"] & (df["proba_avg"] < 0.80),
}

for name, flag in variants.items():
    p = gender_base.copy()
    p[(p == 1) & flag] = 0
    rules["gender_plus_" + name] = p.to_numpy()

rows = []
y = df["assistant_label"].to_numpy()

for name, pred in rules.items():
    row = {"rule": name, "n": len(df)}
    row.update(metric(y, pred))
    rows.append(row)

res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
res.to_csv(OUT, index=False)

flagged = df[
    (df["base_pred"] == 1)
    & df["safe_brand_mismatch"]
][[
    "assistant_label", "base_pred", "proba_avg", "query_brand_hit",
    "query", "title", "category", "brand", "gender",
    "has_compat_word", "has_intrinsic_word"
]].copy()
flagged.to_csv(FLAG_OUT, index=False, encoding="utf-8-sig")

print(res.to_string(index=False))
print()
print("flag counts among base positives:")
print("safe_brand_mismatch:", int(((df["base_pred"] == 1) & df["safe_brand_mismatch"]).sum()))
print("safe_brand_no_compat:", int(((df["base_pred"] == 1) & variants["safe_brand_no_compat"]).sum()))
print("safe_brand_intrinsic_no_compat_lt085:", int(((df["base_pred"] == 1) & variants["safe_brand_intrinsic_no_compat_lt085"]).sum()))
print("saved:", OUT)
print("flagged rows:", FLAG_OUT)
