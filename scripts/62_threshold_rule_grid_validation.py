from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")
SCORE_PATH = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
OUT = ROOT / "reports/manual_review/v5_threshold_rule_grid_validation.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
}

THRESHOLDS = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

SAFE_BRANDS = [
    "nike", "adidas", "puma", "reebok", "vans", "skechers", "new balance", "converse",
    "apple", "samsung", "xiaomi", "huawei", "lenovo", "asus", "acer", "hp", "logitech",
    "sony", "philips", "dyson", "beko", "arcelik", "bosch", "siemens", "tefal", "fakir",
    "vestel", "karaca", "korkmaz", "arzum", "ikea", "english home", "madame coco",
    "cerave", "la roche posay", "bioderma", "avene", "vichy", "neutrogena", "nivea",
    "maybelline", "loreal", "l oreal", "golden rose", "flormar", "clinique",
    "defacto", "koton", "lc waikiki", "zara", "bershka", "stradivarius", "mango",
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
    "kapak", "kayis", "kayış"
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
    words = [re.escape(norm(w)) for w in words if norm(w)]
    if not words:
        return pd.Series(False, index=series.index)
    pat = r"(?:^| )(?:%s)(?: |$)" % "|".join(words)
    return series.str.contains(pat, regex=True, na=False)

def any_contains(series, words):
    words = [re.escape(norm(w)) for w in words if norm(w)]
    if not words:
        return pd.Series(False, index=series.index)
    pat = "|".join(words)
    return series.str.contains(pat, regex=True, na=False)

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1]),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

SAFE_BRANDS = sorted(set(norm(x) for x in SAFE_BRANDS), key=len, reverse=True)
ALIASES = {norm(k): [norm(v) for v in vals] for k, vals in ALIASES.items()}

scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

rows = []

for label_name, label_path in LABEL_FILES.items():
    df = pd.read_csv(label_path)
    df["id"] = df["id"].astype(str)
    df = df[df["assistant_label"].isin([0, 1, "0", "1"])].copy()
    df["assistant_label"] = df["assistant_label"].astype(int)
    df = df.merge(scores, on="id", how="left", validate="many_to_one").dropna(subset=["proba_avg"])

    q = norm_series(df["query"])
    item_gender_text = norm_series(
        df.get("gender", "").fillna("").astype(str) + " " +
        df.get("title", "").fillna("").astype(str) + " " +
        df.get("category", "").fillna("").astype(str)
    )
    item_text = norm_series(
        df.get("title", "").fillna("").astype(str) + " " +
        df.get("category", "").fillna("").astype(str) + " " +
        df.get("attributes", "").fillna("").astype(str) + " " +
        df.get("brand", "").fillna("").astype(str)
    )
    item_brand = norm_series(df.get("brand", "").fillna("").astype(str))

    q_male = word_contains(q, ["erkek", "bay", "men", "male"])
    q_female = word_contains(q, ["kadin", "bayan", "kiz", "women", "woman", "female"])
    item_male = word_contains(item_gender_text, ["erkek", "bay", "men", "male"])
    item_female = word_contains(item_gender_text, ["kadin", "bayan", "kiz", "women", "woman", "female"])

    gender_mismatch = (
        (q_male & ~q_female & item_female & ~item_male)
        |
        (q_female & ~q_male & item_male & ~item_female)
    )

    safe_brand_mismatch = pd.Series(False, index=df.index)

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

    y = df["assistant_label"].to_numpy()
    s = df["proba_avg"].to_numpy()

    for th in THRESHOLDS:
        base = (s >= th).astype(int)

        gender = base.copy()
        gender[(gender == 1) & gender_mismatch.to_numpy()] = 0

        gender_brand = base.copy()
        combined = (gender_mismatch | safe_brand_filter).to_numpy()
        gender_brand[(gender_brand == 1) & combined] = 0

        for rule_name, pred in [
            ("base", base),
            ("gender", gender),
            ("gender_safe_brand", gender_brand),
        ]:
            row = {
                "label_set": label_name,
                "threshold": th,
                "rule": rule_name,
                "n": len(df),
            }
            row.update(metrics(y, pred))
            rows.append(row)

res = pd.DataFrame(rows)
res = res.sort_values(["label_set", "macro_f1"], ascending=[True, False])
res.to_csv(OUT, index=False)

print(res.groupby("label_set").head(20).to_string(index=False))
print("saved:", OUT)
