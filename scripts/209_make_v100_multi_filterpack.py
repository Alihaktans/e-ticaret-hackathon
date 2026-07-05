"""Build a new multi-filter pack around the public-0.82 v95 anchor."""
from pathlib import Path
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(".")

BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

TEST_SCORE = ROOT / "data/processed/v83_independent_test_scores.parquet"
TEST_FEAT = ROOT / "data/processed/v21_test_features.parquet"

TRAIN_FEATURES = ROOT / "data/processed/v82_v34_features.parquet"
HOLD = ROOT / "models/v76_trendyol_contrastive/holdout_terms.txt"
FEATURE_REPORT = ROOT / "reports/experiments/v83_independent_pairlocal_classifier.json"
MODEL = ROOT / "models/v83_independent_pairlocal_classifier.cbm"

OUT_DIR = ROOT / "submissions/final_candidates_v105"
REPORT = ROOT / "reports/experiments/v105_multi_filterpack.json"

ATTR_RULE_COLS = [
    "query_has_color",
    "query_has_known_brand",
    "gender_mismatch",
    "color_mismatch",
    "query_brand_mismatch",
    "weighted_overlap_pct_rank",
    "last_token_in_title",
    "brand_in_query_pair",
]
ATTR_FLAG_COLS = [
    "query_has_color",
    "query_has_known_brand",
    "gender_mismatch",
    "color_mismatch",
    "query_brand_mismatch",
    "last_token_in_title",
    "brand_in_query_pair",
]
ATTR_FLOAT_COLS = ["weighted_overlap_pct_rank"]


TR_MAP = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ş": "s",
        "Ş": "s",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    }
)

STOP = {
    "ve",
    "ile",
    "icin",
    "bir",
    "adet",
    "set",
    "takim",
    "model",
    "uyumlu",
    "orjinal",
    "orijinal",
    "yeni",
    "renk",
    "boy",
    "numara",
    "beden",
    "cm",
    "mm",
    "lt",
    "kg",
    "gr",
    "ml",
    "x",
    "no",
    "olan",
    "veya",
    "plus",
    "pro",
    "max",
    "mini",
    "urun",
    "da",
    "de",
    "mi",
    "mu",
    "ve",
}

COLOR_WORDS = {
    "siyah",
    "beyaz",
    "kirmizi",
    "mavi",
    "lacivert",
    "yesil",
    "sari",
    "pembe",
    "mor",
    "turuncu",
    "gri",
    "antrasit",
    "kahverengi",
    "bej",
    "krem",
    "gold",
    "gumus",
    "silver",
    "bordo",
    "ekru",
}

MALE_WORDS = {"erkek", "bay", "adam"}
FEMALE_WORDS = {"kadin", "bayan", "kiz"}
UNISEX_WORDS = {"unisex"}
CHILD_WORDS = {"cocuk", "bebek", "junior", "kids", "kid"}

ACCESSORY_QUERY_PHRASES = (
    "ekran koruyucu",
    "kirilmaz ekran",
    "kirilmaz cam",
    "koruyucu cam",
    "sarj aleti",
    "sarz aleti",
    "hizli sarj aleti",
    "sarj adaptoru",
    "sarz adaptoru",
    "tablet kalemi",
    "tablet kalemleri",
    "dokunmatik kalem",
    "telefon tutucu",
    "tablet tutucu",
    "monitor yukseltici",
    "yazici murekkebi",
    "kamera lens koruyucu",
    "süpürge torbasi",
    "supurge torbasi",
    "süpürge filtresi",
    "supurge filtresi",
)

ACCESSORY_DEVICE_TOKEN_PREFIXES = {
    "aksesu",
    "kilif",
    "kapak",
    "case",
    "powerbank",
    "kordon",
    "kayis",
    "kartus",
    "toner",
    "filtre",
    "tripod",
    "lens",
    "yedek",
    "tutucu",
    "murekkep",
    "yukseltici",
}

ACCESSORY_DEVICE_EXACT_TOKENS = {
    "kablo",
    "sarj",
    "adaptor",
    "adapter",
    "aparat",
    "kalem",
    "torba",
    "torbasi",
}

ACCESSORY_ITEM_CATEGORY_HINTS = (
    "kapak kilif",
    "tablet kilifi",
    "tablet ekran koruyucu",
    "ekran koruyucu film",
    "kamera lens koruyucu",
    "telefon tutucu",
    "sarj aleti",
    "sarj cihazlari",
    "powerbank",
    "telefon bataryasi",
    "telefon ekrani",
    "telefon yedek parcalari",
    "yazici murekkebi",
    "kartus",
    "toner",
    "notebook sogutucu",
    "notebook koruyucu",
    "notebook adaptoru",
    "notebook cantasi",
    "tablet aksesuarlari",
    "akilli saat kordon",
    "akilli saat ekran koruyucu",
    "akilli saat aksesuarlari",
    "playstation aksesuari",
    "konsol aksesuarlari",
    "tv aksesuarlari",
    "foto kamera aksesuari",
    "supurge aksesuari",
)

ACCESSORY_CATEGORY_BROAD_HINTS = (
    "aksesuar",
    "aksesuarlari",
    "yedek parca",
    "sarj cihazlari",
    "tablet aksesuarlari",
    "konsol aksesuarlari",
)

MAIN_ITEM_CATEGORY_HINTS = (
    "android cep telefonu",
    "ios cep telefonlari",
    "ios cep telefonu",
    "tuslu cep telefonu",
    "tablet grubu tablet",
    "yenilenmis tablet",
    "cocuk cizim tableti",
    "grafik tablet",
    "oyuncu monitor",
    "monitorler monitor",
    "dizustu bilgisayar",
    "oyuncu dizustu bilgisayari",
    "murekkep puskurtmeli yazici",
    "tankli yazici",
    "lazer yazici",
    "mini yazici",
    "giyilebilir teknoloji akilli saat",
    "akilli ev aletleri robot supurge",
    "dik supurge",
    "torbasiz supurge",
    "kablosuz dikey supurge",
    "kulak ici tws bluetooth kulaklik",
    "kulak ici kablolu kulaklik",
    "oyuncu kulaklik",
)

ITEM_ACCESSORY_TITLE_PHRASES = (
    "ekran koruyucu",
    "kirilmaz cam",
    "sarj aleti",
    "sarj adaptoru",
    "telefon tutucu",
    "monitor yukseltici",
    "yazici murekkebi",
    "tablet kalemi",
)

DEVICE_HINTS = {
    "iphone",
    "telefon",
    "cep",
    "galaxy",
    "samsung",
    "xiaomi",
    "redmi",
    "ipad",
    "tablet",
    "matepad",
    "watch",
    "playstation",
    "ps4",
    "ps5",
    "xbox",
    "roborock",
    "supurge",
    "dyson",
    "printer",
    "yazici",
    "monitor",
    "laptop",
    "notebook",
}

STRICT_HEAD_GROUPS = {"drinkware", "cookware"}

STORAGE_NUMBERS = {"32", "64", "128", "256", "512", "1024"}
UNIT_TOKENS = {"gb", "tb", "mb", "w", "wh", "mah", "ml", "lt", "cm", "mm", "kg", "gr", "mp", "hz"}
SPEC_ALNUM_BLOCKLIST = {
    "3g",
    "4g",
    "5g",
    "2k",
    "4k",
    "8k",
    "hd",
    "fhd",
    "qhd",
    "uhd",
    "oled",
    "amoled",
    "ips",
    "lcd",
    "led",
    "wifi5",
    "wifi6",
}
PAIR_MODEL_BASES = {
    "iphone",
    "ipad",
    "watch",
    "galaxy",
    "redmi",
    "note",
    "tab",
    "matepad",
    "honor",
    "lenovo",
    "dyson",
    "roborock",
    "epson",
    "canon",
    "hp",
}

PROMOTION_RULES = {
    ("device_accessory", "phone"): {"target": "accessory", "min_score": 0.45},
    ("device_accessory", "tablet"): {"target": "accessory", "min_score": 0.45},
    ("device_accessory", "vacuum"): {"target": "accessory", "min_score": 0.28},
    ("device_main", "phone"): {"target": "main", "min_score": 0.22},
}

FORMULA_WORDS = {"mama", "devam", "sutu", "sut", "combiotik", "aptamil", "bebelac", "hero", "milupa", "sma"}

HEAD_SYNONYMS = {
    "nevresim_takimi": [["nevresim", "takimi"], ["nevresim", "takim"]],
    "carsaf": [["carsaf"]],
    "yorgan": [["yorgan"]],
    "battaniye": [["battaniye"]],
    "pike": [["pike"]],
    "yatak_ortusu": [["yatak", "ortusu"]],
    "parfum": [["parfum"]],
    "deodorant": [["deodorant"]],
    "sampuan": [["sampuan"]],
    "serum": [["serum"]],
    "termos": [["termos"]],
    "matara": [["matara"]],
    "kupa": [["kupa"]],
    "bardak": [["bardak"]],
    "tencere": [["tencere"]],
    "tava": [["tava"]],
    "cezve": [["cezve"]],
    "defter": [["defter"]],
    "kalem": [["kalem"]],
    "kitap": [["kitap"]],
    "soru_bankasi": [["soru", "bankasi"]],
    "deneme": [["deneme"]],
    "konu_anlatim": [["konu", "anlatim"]],
    "pantolon": [["pantolon"]],
    "etek": [["etek"]],
    "elbise": [["elbise"]],
    "gomlek": [["gomlek"]],
    "kazak": [["kazak"]],
    "ceket": [["ceket"]],
    "mont": [["mont"]],
    "esofman": [["esofman"]],
    "ayakkabi": [["ayakkabi"]],
    "bot": [["bot"]],
    "terlik": [["terlik"]],
    "sandalet": [["sandalet"]],
    "krampon": [["krampon"]],
    "figur": [["figur"]],
    "pelus": [["pelus"]],
    "oyuncak": [["oyuncak"]],
}

HEAD_GROUP = {
    "nevresim_takimi": "bedding",
    "carsaf": "bedding",
    "yorgan": "bedding",
    "battaniye": "bedding",
    "pike": "bedding",
    "yatak_ortusu": "bedding",
    "parfum": "fragrance",
    "deodorant": "fragrance",
    "sampuan": "haircare",
    "serum": "haircare",
    "termos": "drinkware",
    "matara": "drinkware",
    "kupa": "drinkware",
    "bardak": "drinkware",
    "tencere": "cookware",
    "tava": "cookware",
    "cezve": "cookware",
    "defter": "stationery",
    "kalem": "stationery",
    "kitap": "book",
    "soru_bankasi": "book",
    "deneme": "book",
    "konu_anlatim": "book",
    "pantolon": "apparel",
    "etek": "apparel",
    "elbise": "apparel",
    "gomlek": "apparel",
    "kazak": "apparel",
    "ceket": "apparel",
    "mont": "apparel",
    "esofman": "apparel",
    "ayakkabi": "footwear",
    "bot": "footwear",
    "terlik": "footwear",
    "sandalet": "footwear",
    "krampon": "footwear",
    "figur": "toy",
    "pelus": "toy",
    "oyuncak": "toy",
}

FAMILY_DROP = STOP | COLOR_WORDS | MALE_WORDS | FEMALE_WORDS | UNISEX_WORDS


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    try:
        x = x.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    x = x.translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def tokens(x):
    return [t for t in norm_text(x).split() if len(t) >= 2]


def token_set(x):
    return set(tokens(x))


def has_phrase(toks_set, phrase_tokens):
    return all(tok in toks_set for tok in phrase_tokens)


def person_flag(text):
    t = token_set(text)
    if "tek" in t and "kisilik" in t:
        return "tek"
    if "cift" in t and "kisilik" in t:
        return "cift"
    return ""


def class_grade_set(text):
    n = norm_text(text)
    grades = set()
    for m in re.finditer(r"sinif", n):
        prefix = n[max(0, m.start() - 28): m.start()]
        grades.update(re.findall(r"\b([1-9]|1[0-2])\b", prefix))
    return grades


def formula_stage_set(text):
    n = norm_text(text)
    if not (token_set(n) & FORMULA_WORDS):
        return set()
    stages = set(re.findall(r"\b(?:no|numara)\s*([1-6])\b", n))
    stages.update(re.findall(r"\b([1-6])\s*(?:no|numara)\b", n))
    stages.update(re.findall(r"\b(?:aptamil|bebelac|hero|milupa|sma)\s*([1-6])\b", n))
    stages.update(re.findall(r"\b([1-6])\s*(?:devam|bebek)\b", n))
    return stages


def diaper_no_set(text):
    n = norm_text(text)
    if "bez" not in n:
        return set()
    vals = set()
    for start, end in re.findall(r"\b([1-8])\s*-\s*([1-8])\s*(?:numara|no)\b", n):
        lo = min(int(start), int(end))
        hi = max(int(start), int(end))
        vals.update(str(v) for v in range(lo, hi + 1))
    vals.update(re.findall(r"\b(?:no|numara)\s*([1-8])\+?\b", n))
    vals.update(re.findall(r"\b([1-8])\+?\s*(?:numara|no)\b", n))
    return vals


def tire_signature(text):
    n = norm_text(text)
    m = re.search(r"\b(\d{3})\s*(\d{2})\s*r\s*(\d{2})\b", n)
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}-r{m.group(3)}"


def jant_size(text):
    m = re.search(r"\b(\d{2})\s*jant\b", norm_text(text))
    return m.group(1) if m else ""


def hair_dye_code(text):
    n = norm_text(text)
    m = re.search(r"\b\d{1,2}(?:\.\d{1,2}){1,2}\b", n)
    return m.group(0) if m else ""


def oil_viscosity(text):
    n = norm_text(text).replace(" ", "")
    m = re.search(r"\b\d{1,2}w\d{2}\b", n)
    return m.group(0) if m else ""


def head_labels_from_tokens(toks_set):
    out = set()
    for label, patterns in HEAD_SYNONYMS.items():
        for pattern in patterns:
            if has_phrase(toks_set, pattern):
                out.add(label)
                break
    return out


def gender_state(query_tokens, item_gender=""):
    if query_tokens & UNISEX_WORDS:
        return "unisex"
    gender_token = norm_text(item_gender)
    male = bool(query_tokens & MALE_WORDS)
    female = bool(query_tokens & FEMALE_WORDS)
    if male and female:
        return "unisex"
    if male and not female:
        return "male"
    if female and not male:
        return "female"
    if gender_token == "unisex":
        return "unisex"
    if gender_token == "erkek":
        return "male"
    if gender_token == "kadin":
        return "female"
    return "unknown"


def token_has_prefix(toks_set, prefixes):
    return any(any(tok.startswith(prefix) for prefix in prefixes) for tok in toks_set)


def token_in_set(toks_set, exact_tokens):
    return any(tok in exact_tokens for tok in toks_set)


def item_is_explicit_main(category_norm):
    return any(x in category_norm for x in MAIN_ITEM_CATEGORY_HINTS)


def item_is_explicit_accessory(title_norm, category_norm):
    if item_is_explicit_main(category_norm):
        return False
    if any(x in category_norm for x in ACCESSORY_ITEM_CATEGORY_HINTS):
        return True
    if not any(x in category_norm for x in ACCESSORY_CATEGORY_BROAD_HINTS):
        return False
    title_tokens = set(title_norm.split())
    if token_has_prefix(title_tokens, ACCESSORY_DEVICE_TOKEN_PREFIXES):
        return True
    if token_in_set(title_tokens, ACCESSORY_DEVICE_EXACT_TOKENS):
        return True
    return any(x in title_norm for x in ITEM_ACCESSORY_TITLE_PHRASES)


def query_is_accessory(query_norm, toks_set):
    if any(x in query_norm for x in ACCESSORY_QUERY_PHRASES):
        return True
    if token_has_prefix(toks_set, ACCESSORY_DEVICE_TOKEN_PREFIXES) and toks_set & DEVICE_HINTS:
        return True
    if token_in_set(toks_set, ACCESSORY_DEVICE_EXACT_TOKENS) and toks_set & DEVICE_HINTS:
        return True
    if any(tok.startswith("cant") for tok in toks_set) and toks_set & {"laptop", "notebook", "tablet"}:
        return True
    if any(tok.startswith("kalem") for tok in toks_set) and toks_set & {"ipad", "iphone", "tablet", "apple"}:
        return True
    return False


def query_device_domain(toks_set):
    return bool(toks_set & DEVICE_HINTS)


def query_has_main_device_signal(query_norm, toks_set, compat_tokens):
    if compat_tokens:
        return True
    if toks_set & {"telefon", "telefonu", "tablet", "monitor", "yazici", "printer", "supurge", "laptop", "notebook", "watch"}:
        return True
    return any(
        phrase in query_norm
        for phrase in (
            "cep telefonu",
            "akilli cep telefonu",
            "tuslu cep telefonu",
            "robot supurge",
            "dikey elektrikli supurge",
            "tankli yazici",
            "mini yazici",
            "akilli saat",
        )
    )


def query_device_family(query_norm, toks_set):
    if toks_set & {"ipad", "tablet", "matepad"} or "tab" in toks_set:
        return "tablet"
    if "watch" in toks_set or "akilli saat" in query_norm:
        return "watch"
    if toks_set & {"yazici", "printer"}:
        return "printer"
    if "monitor" in toks_set:
        return "monitor"
    if toks_set & {"supurge", "dyson", "roborock"}:
        return "vacuum"
    if toks_set & {"playstation", "ps4", "ps5", "xbox"}:
        return "console"
    if toks_set & {"laptop", "notebook"}:
        return "laptop"
    if "kamera" in toks_set or "camera" in toks_set:
        return "camera"
    if "kulaklik" in toks_set:
        return "audio"
    if "iphone" in toks_set or toks_set & {"telefon", "telefonu", "cep", "galaxy", "xiaomi", "redmi", "samsung"}:
        return "phone"
    return "other"


def item_device_family(category_path, combined_text):
    if any(x in category_path for x in {"cep telefonu", "telefon aksesuarlari", "ios cep telefonu", "android cep telefonu"}):
        return "phone"
    if any(x in category_path for x in {"tablet", "tablet aksesuarlari", "grafik tablet"}):
        return "tablet"
    if "akilli saat" in category_path:
        return "watch"
    if any(x in category_path for x in {"yazici", "kartus", "toner", "murekkep"}):
        return "printer"
    if "monitor" in category_path:
        return "monitor"
    if any(x in category_path for x in {"supurge", "robot supurge", "supurge aksesuari"}):
        return "vacuum"
    if any(x in category_path for x in {"playstation", "konsol", "xbox"}):
        return "console"
    if any(x in category_path for x in {"dizustu bilgisayar", "notebook", "bilgisayarlar"}):
        return "laptop"
    if any(x in category_path for x in {"kamera", "foto"}):
        return "camera"
    if "kulaklik" in category_path:
        return "audio"
    if "iphone" in combined_text or "galaxy" in combined_text or "redmi" in combined_text:
        return "phone"
    if "ipad" in combined_text or "matepad" in combined_text or "tablet" in combined_text:
        return "tablet"
    if "watch" in combined_text:
        return "watch"
    if "monitor" in combined_text:
        return "monitor"
    if "yazici" in combined_text or "printer" in combined_text:
        return "printer"
    if "supurge" in combined_text or "roborock" in combined_text or "dyson" in combined_text:
        return "vacuum"
    return "other"


def query_intent(query_norm, device_family, is_accessory_query, has_main_device_signal, gender_state, grade_set, head_labels):
    if device_family != "other" and is_accessory_query:
        return "device_accessory"
    if device_family != "other" and has_main_device_signal:
        return "device_main"
    if grade_set or any(h in {"kitap", "soru_bankasi", "deneme", "konu_anlatim"} for h in head_labels):
        return "education"
    if gender_state in {"male", "female", "unisex"} and any(h in {"pantolon", "etek", "elbise", "gomlek", "kazak", "ceket", "mont", "esofman", "ayakkabi", "bot", "terlik", "sandalet", "krampon"} for h in head_labels):
        return "gendered_fashion"
    return "general"


def strong_model_tokens(text):
    toks_list = tokens(text)
    out = set()
    for i, tok in enumerate(toks_list):
        if tok in UNIT_TOKENS:
            continue
        if tok in SPEC_ALNUM_BLOCKLIST:
            continue
        if tok in STORAGE_NUMBERS and ((i + 1 < len(toks_list) and toks_list[i + 1] in {"gb", "tb"}) or (i > 0 and toks_list[i - 1] in {"gb", "tb"})):
            continue
        if re.fullmatch(r"\d+(?:gb|tb|mb|mah|hz|w|wh|mp)", tok):
            continue
        if re.search(r"[a-z]", tok) and re.search(r"\d", tok):
            if re.fullmatch(r"\d+[a-z]{1,3}", tok) and tok not in {"ps4", "ps5"} and tok[1:] in {"g", "k"}:
                continue
            out.add(tok)
        elif tok.isdigit() and tok not in STORAGE_NUMBERS and 1 <= len(tok) <= 3:
            prev = toks_list[i - 1] if i > 0 else ""
            if prev in PAIR_MODEL_BASES:
                out.add(f"{prev}_{tok}")
            elif prev == "ps":
                out.add(f"ps{tok}")
    return out


def compatibility_tokens(query_text):
    toks_list = token_set(query_text)
    models = strong_model_tokens(query_text)
    for tok in toks_list & {"iphone", "ipad", "galaxy", "matepad", "playstation", "ps4", "ps5", "watch", "roborock"}:
        models.add(tok)
    return models


def build_brand_token_set():
    brands = pd.read_csv(ITEMS, usecols=["brand"])
    brands["brand_norm"] = brands["brand"].map(norm_text)
    counts = Counter()
    for brand, cnt in brands["brand_norm"].value_counts().items():
        if not brand or brand == "unknown":
            continue
        parts = [p for p in brand.split() if len(p) >= 3]
        for part in parts:
            counts[part] += int(cnt)
    return {tok for tok, cnt in counts.items() if cnt >= 25 and tok not in FAMILY_DROP and tok not in {"home", "shop", "store", "tekstil"}}


def family_key(query_text, brand_tokens):
    toks_list = tokens(query_text)
    kept = [t for t in toks_list if t not in FAMILY_DROP and t not in brand_tokens]
    if len(kept) < 2:
        kept = [t for t in toks_list if t not in FAMILY_DROP]
    return " ".join(kept[:10])


def build_item_meta(needed_item_ids):
    cols = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    items = pd.read_csv(ITEMS, usecols=cols, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)
    items = items[items["item_id"].isin(needed_item_ids)].copy()
    items["title_norm"] = items["title"].map(norm_text)
    items["category_norm"] = items["category"].map(norm_text)
    items["brand_norm"] = items["brand"].map(norm_text)
    items["attr_norm"] = items["attributes"].map(norm_text)
    items["root"] = items["category"].fillna("").astype(str).str.split("/").str[0].map(norm_text)

    item_info = {}
    for row in items.itertuples(index=False):
        cat_parts = [norm_text(part) for part in str(row.category).split("/") if norm_text(part)]
        category_path = "/".join(cat_parts)
        path2 = "/".join(cat_parts[:2])
        path3 = "/".join(cat_parts[:3])
        combined_tokens = set((row.title_norm + " " + row.category_norm + " " + row.brand_norm + " " + row.attr_norm).split())
        combined_text = " ".join(sorted(combined_tokens))
        item_info[row.item_id] = {
            "title_norm": row.title_norm,
            "category_norm": row.category_norm,
            "category_path": category_path,
            "path2": path2,
            "path3": path3,
            "brand_norm": row.brand_norm,
            "attr_norm": row.attr_norm,
            "root": row.root,
            "gender_state": gender_state(combined_tokens, row.gender),
            "person": person_flag(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
            "grade_set": class_grade_set(row.title_norm + " " + row.category_norm),
            "formula_stage_set": formula_stage_set(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
            "tire_sig": tire_signature(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
            "jant": jant_size(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
            "head_labels": head_labels_from_tokens(combined_tokens),
            "is_accessory": item_is_explicit_accessory(row.title_norm, row.category_norm),
            "is_main_device": item_is_explicit_main(row.category_norm),
            "device_family": item_device_family(category_path, combined_text),
            "compat_tokens": compatibility_tokens(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
            "diaper_no_set": diaper_no_set(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
            "hair_dye_code": hair_dye_code(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
            "oil_viscosity": oil_viscosity(row.title_norm + " " + row.category_norm + " " + row.attr_norm),
        }
    return item_info


def build_query_info(term_ids, brand_tokens):
    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    terms = terms[terms["term_id"].isin(term_ids)].copy()
    terms["query_norm"] = terms["query"].map(norm_text)
    query_info = {}
    for row in terms.itertuples(index=False):
        q_tokens = set(row.query_norm.split())
        compat = compatibility_tokens(row.query_norm)
        device_family = query_device_family(row.query_norm, q_tokens)
        is_acc = query_is_accessory(row.query_norm, q_tokens)
        has_main = query_has_main_device_signal(row.query_norm, q_tokens, compat)
        heads = head_labels_from_tokens(q_tokens)
        query_info[row.term_id] = {
            "query": row.query,
            "query_norm": row.query_norm,
            "tokens": q_tokens,
            "gender_state": gender_state(q_tokens),
            "person": person_flag(row.query_norm),
            "grade_set": class_grade_set(row.query_norm),
            "formula_stage_set": formula_stage_set(row.query_norm),
            "tire_sig": tire_signature(row.query_norm),
            "jant": jant_size(row.query_norm),
            "head_labels": heads,
            "is_accessory_query": is_acc,
            "is_device_domain": query_device_domain(q_tokens),
            "has_main_device_signal": has_main,
            "device_family": device_family,
            "intent": query_intent(
                row.query_norm,
                device_family,
                is_acc,
                has_main,
                gender_state(q_tokens),
                class_grade_set(row.query_norm),
                heads,
            ),
            "compat_tokens": compat,
            "family_key": family_key(row.query_norm, brand_tokens),
            "diaper_no_set": diaper_no_set(row.query_norm),
            "hair_dye_code": hair_dye_code(row.query_norm),
            "oil_viscosity": oil_viscosity(row.query_norm),
        }
    return query_info


def build_family_stats(train_pairs, query_info, item_info):
    stats = defaultdict(
        lambda: {
            "n": 0,
            "queries": set(),
            "roots": Counter(),
            "heads": Counter(),
            "accessory": Counter(),
            "p2": Counter(),
            "p3": Counter(),
        }
    )
    for row in train_pairs.itertuples(index=False):
        q = query_info.get(row.term_id)
        it = item_info.get(row.item_id)
        if q is None or it is None:
            continue
        key = q["family_key"]
        if not key:
            continue
        st = stats[key]
        st["n"] += 1
        st["queries"].add(row.term_id)
        st["roots"][it["root"]] += 1
        for head in it["head_labels"]:
            st["heads"][head] += 1
        st["accessory"]["accessory" if it["is_accessory"] else "main"] += 1
        st["p2"][it["path2"]] += 1
        st["p3"][it["path3"]] += 1
    return stats


def compute_holdout_score():
    feature_cols = json.loads(FEATURE_REPORT.read_text())["features"]
    valid = pd.read_parquet(TRAIN_FEATURES)
    valid["term_id"] = valid["term_id"].astype(str)
    hold = set(HOLD.read_text().splitlines())
    valid = valid[valid["term_id"].isin(hold)].copy()
    model = CatBoostClassifier()
    model.load_model(str(MODEL))
    valid["rule_score"] = model.predict_proba(valid[feature_cols].fillna(-1).astype("float32"))[:, 1]
    return valid[["term_id", "item_id", "label", "rule_score"]].copy()


def prepare_test_rows():
    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    for col in ["id", "term_id", "item_id"]:
        pairs[col] = pairs[col].astype(str)
    score = pd.read_parquet(TEST_SCORE, columns=["id", "v82_score"])
    score["id"] = score["id"].astype(str)
    feat = pd.read_parquet(TEST_FEAT, columns=["id", "title_cov_pct_rank"])
    feat["id"] = feat["id"].astype(str)
    test = base.merge(pairs, on="id").merge(score, on="id").merge(feat, on="id")
    test = test[test["prediction"].eq(1)].copy()
    test["score"] = test["v82_score"].astype(np.float32)
    test["label"] = -1
    return test[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]].copy()


def prepare_full_test_pool(term_ids=None):
    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    for col in ["id", "term_id", "item_id"]:
        pairs[col] = pairs[col].astype(str)
    if term_ids is not None:
        term_ids = set(term_ids)
        pairs = pairs[pairs["term_id"].isin(term_ids)].copy()
        base = base[base["id"].isin(set(pairs["id"]))].copy()
    score = pd.read_parquet(TEST_SCORE, columns=["id", "v82_score"])
    score["id"] = score["id"].astype(str)
    feat = pd.read_parquet(TEST_FEAT, columns=["id", "title_cov_pct_rank"])
    feat["id"] = feat["id"].astype(str)
    test = base.merge(pairs, on="id").merge(score, on="id").merge(feat, on="id")
    test["score"] = test["v82_score"].astype(np.float32)
    test["label"] = -1
    test = test.rename(columns={"prediction": "base_prediction"})
    return test[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label", "base_prediction"]].copy()


def prepare_valid_rows(valid_scores):
    valid = valid_scores.copy()
    valid["item_id"] = valid["item_id"].astype(str)
    valid["score"] = valid["rule_score"].astype(np.float32)
    valid["title_cov_pct_rank"] = np.float32(0.5)
    valid["id"] = ""
    return valid[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]].copy()


def family_support(item_meta, family_stat):
    total = max(1, family_stat["n"])
    root_support = family_stat["roots"].get(item_meta["root"], 0) / total
    acc_key = "accessory" if item_meta["is_accessory"] else "main"
    acc_support = family_stat["accessory"].get(acc_key, 0) / total
    p2_support = family_stat["p2"].get(item_meta["path2"], 0) / total
    p3_support = family_stat["p3"].get(item_meta["path3"], 0) / total
    top_root_ratio = max(family_stat["roots"].values()) / total if family_stat["roots"] else 0.0
    top_accessory_ratio = max(family_stat["accessory"].values()) / total if family_stat["accessory"] else 0.0
    top_p2_ratio = max(family_stat["p2"].values()) / total if family_stat["p2"] else 0.0
    top_p3_ratio = max(family_stat["p3"].values()) / total if family_stat["p3"] else 0.0
    return root_support, acc_support, p2_support, p3_support, top_root_ratio, top_accessory_ratio, top_p2_ratio, top_p3_ratio


def evaluate_rows(rows, query_info, item_info, family_stats):
    results = []
    for row in rows.itertuples(index=False):
        q = query_info.get(row.term_id)
        it = item_info.get(row.item_id)
        if q is None or it is None:
            continue

        reasons = []
        flags = {
            "gender_veto": 0,
            "main_accessory_veto": 0,
            "head_conflict_veto": 0,
            "package_variant_veto": 0,
            "compatibility_veto": 0,
            "family_veto_strict": 0,
            "family_veto_balanced": 0,
        }

        # 4) gender rule
        qg = q["gender_state"]
        ig = it["gender_state"]
        if qg == "male" and ig == "female":
            flags["gender_veto"] = 1
            reasons.append("gender_male_vs_female")
        elif qg == "female" and ig == "male":
            flags["gender_veto"] = 1
            reasons.append("gender_female_vs_male")
        elif qg == "unisex" and ig in {"male", "female"}:
            flags["gender_veto"] = 1
            reasons.append("gender_unisex_vs_single")

        # 6) main/accessory direction
        if q["intent"] == "device_accessory":
            if it["is_main_device"]:
                flags["main_accessory_veto"] = 1
                reasons.append("query_accessory_item_main")
        elif q["intent"] == "device_main" and it["is_accessory"]:
            if q["is_device_domain"] or q["compat_tokens"]:
                flags["main_accessory_veto"] = 1
                reasons.append("query_main_item_accessory")

        # 2) head noun conflict
        q_by_group = defaultdict(set)
        it_by_group = defaultdict(set)
        for head in q["head_labels"]:
            q_by_group[HEAD_GROUP[head]].add(head)
        for head in it["head_labels"]:
            it_by_group[HEAD_GROUP[head]].add(head)
        for group in set(q_by_group) & set(it_by_group):
            if group not in STRICT_HEAD_GROUPS:
                continue
            if len(q_by_group[group]) == 1 and q_by_group[group].isdisjoint(it_by_group[group]):
                flags["head_conflict_veto"] = 1
                reasons.append(f"head_conflict_{group}")
                break

        # 1+3) strict package / variant
        if q["person"] and it["person"] and q["person"] != it["person"]:
            flags["package_variant_veto"] = 1
            reasons.append("person_count_mismatch")
        if q["grade_set"] and it["grade_set"] and not (q["grade_set"] & it["grade_set"]):
            flags["package_variant_veto"] = 1
            reasons.append("class_grade_mismatch")
        if q["formula_stage_set"] and it["formula_stage_set"] and not (q["formula_stage_set"] & it["formula_stage_set"]):
            flags["package_variant_veto"] = 1
            reasons.append("formula_stage_mismatch")
        if q["diaper_no_set"] and it["diaper_no_set"] and not (q["diaper_no_set"] & it["diaper_no_set"]):
            flags["package_variant_veto"] = 1
            reasons.append("diaper_no_mismatch")
        if q["tire_sig"] and it["tire_sig"] and q["tire_sig"] != it["tire_sig"]:
            flags["package_variant_veto"] = 1
            reasons.append("tire_size_mismatch")
        if q["jant"] and it["jant"] and q["jant"] != it["jant"]:
            flags["package_variant_veto"] = 1
            reasons.append("jant_mismatch")
        if q["hair_dye_code"] and it["hair_dye_code"] and q["hair_dye_code"] != it["hair_dye_code"]:
            flags["package_variant_veto"] = 1
            reasons.append("hair_dye_code_mismatch")
        if q["oil_viscosity"] and it["oil_viscosity"] and q["oil_viscosity"] != it["oil_viscosity"]:
            flags["package_variant_veto"] = 1
            reasons.append("oil_viscosity_mismatch")

        # deeper compatibility rule for device-family accessory intents
        if q["intent"] == "device_accessory" and q["compat_tokens"]:
            overlap = q["compat_tokens"] & it["compat_tokens"]
            if not overlap and it["compat_tokens"] and q["is_device_domain"]:
                flags["compatibility_veto"] = 1
                reasons.append("compatibility_model_mismatch")

        # 5) query-family consistency
        fam = family_stats.get(q["family_key"])
        if fam and fam["n"] >= 6 and len(fam["queries"]) >= 3:
            root_support, acc_support, p2_support, p3_support, top_root_ratio, top_acc_ratio, top_p2_ratio, top_p3_ratio = family_support(it, fam)
            if fam["n"] >= 10 and top_root_ratio >= 0.85 and root_support == 0.0 and row.score <= 0.12:
                flags["family_veto_strict"] = 1
                reasons.append("family_root_unsupported")
            if fam["n"] >= 10 and top_acc_ratio >= 0.85 and acc_support == 0.0 and row.score <= 0.12:
                flags["family_veto_strict"] = 1
                reasons.append("family_accessory_direction")
            if fam["n"] >= 10 and top_p3_ratio >= 0.8 and p3_support == 0.0 and row.score <= 0.12:
                flags["family_veto_strict"] = 1
                reasons.append("family_p3_unsupported")
            if fam["n"] >= 8 and top_root_ratio >= 0.75 and root_support <= 0.05 and row.score <= 0.18:
                flags["family_veto_balanced"] = 1
                reasons.append("family_root_soft_unsupported")
            if fam["n"] >= 8 and top_p3_ratio >= 0.7 and p3_support == 0.0 and row.score <= 0.15:
                flags["family_veto_balanced"] = 1
                reasons.append("family_p3_soft_unsupported")

        clean_core_veto = (
            flags["gender_veto"]
            or flags["main_accessory_veto"]
            or flags["package_variant_veto"]
            or flags["compatibility_veto"]
        )
        head_augmented_veto = int(clean_core_veto or flags["head_conflict_veto"])

        results.append(
            {
                "id": row.id,
                "term_id": row.term_id,
                "item_id": row.item_id,
                "label": int(row.label),
                "score": float(row.score),
                "query_intent": q["intent"],
                "query_device_family": q["device_family"],
                "item_device_family": it["device_family"],
                "item_is_accessory": int(it["is_accessory"]),
                "item_is_main_device": int(it["is_main_device"]),
                "gender_veto": flags["gender_veto"],
                "main_accessory_veto": flags["main_accessory_veto"],
                "head_conflict_veto": flags["head_conflict_veto"],
                "package_variant_veto": flags["package_variant_veto"],
                "compatibility_veto": flags["compatibility_veto"],
                "family_veto_strict": flags["family_veto_strict"],
                "family_veto_balanced": flags["family_veto_balanced"],
                "clean_core_veto": int(clean_core_veto),
                "head_augmented_veto": int(head_augmented_veto),
                "full_veto_strict": int(head_augmented_veto or flags["family_veto_strict"]),
                "full_veto_balanced": int(head_augmented_veto or flags["family_veto_balanced"]),
                "reason_text": "|".join(sorted(set(reasons))),
            }
        )
    return pd.DataFrame(results)


def calibration_table(df, col):
    mask = df[col].eq(1)
    rows = int(mask.sum())
    if rows == 0:
        return {"rule": col, "holdout_rows": 0, "holdout_positive_errors": 0, "holdout_negative_precision": np.nan}
    return {
        "rule": col,
        "holdout_rows": rows,
        "holdout_positive_errors": int(df.loc[mask, "label"].sum()),
        "holdout_negative_precision": float((1 - df.loc[mask, "label"]).mean()),
    }


def attach_attribute_rule_features(valid_eval, test_eval):
    hold = set(HOLD.read_text().splitlines())
    valid_feat = pd.read_parquet(TRAIN_FEATURES, columns=["term_id", "item_id"] + ATTR_RULE_COLS)
    valid_feat["term_id"] = valid_feat["term_id"].astype(str)
    valid_feat["item_id"] = valid_feat["item_id"].astype(str)
    valid_feat = valid_feat[valid_feat["term_id"].isin(hold)].drop_duplicates(["term_id", "item_id"]).copy()

    test_feat = pd.read_parquet(TEST_FEAT, columns=["id"] + ATTR_RULE_COLS)
    test_feat["id"] = test_feat["id"].astype(str)
    test_feat = test_feat.drop_duplicates(["id"]).copy()

    valid_eval = valid_eval.merge(valid_feat, on=["term_id", "item_id"], how="left")
    test_eval = test_eval.merge(test_feat, on="id", how="left")

    for df in (valid_eval, test_eval):
        for col in ATTR_FLAG_COLS:
            df[col] = df[col].fillna(0).astype(np.int8)
        for col in ATTR_FLOAT_COLS:
            df[col] = df[col].fillna(1.0).astype(np.float32)

        df["gender_score_veto"] = ((df["gender_mismatch"].eq(1)) & df["score"].le(0.15)).astype(np.int8)
        df["color_score_veto"] = (
            (df["query_has_color"].eq(1)) & (df["color_mismatch"].eq(1)) & df["score"].le(0.10)
        ).astype(np.int8)
        df["gender_color_score_veto"] = df[["gender_score_veto", "color_score_veto"]].max(axis=1).astype(np.int8)
        df["brand_tail_veto"] = (
            (df["query_has_known_brand"].eq(1))
            & (df["query_brand_mismatch"].eq(1))
            & df["score"].le(0.08)
            & df["weighted_overlap_pct_rank"].le(0.50)
            & df["last_token_in_title"].eq(0)
            & df["brand_in_query_pair"].eq(0)
        ).astype(np.int8)
        df["safe_attr_brand_veto"] = df[["gender_color_score_veto", "brand_tail_veto"]].max(axis=1).astype(np.int8)

    return valid_eval, test_eval


def save_variant(ids, pred, name, base_pred, rows):
    path = OUT_DIR / f"FINAL_CANDIDATE_v105_{name}.csv"
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(path, index=False)
    rows.append(
        {
            "variant": name,
            "removed_vs_base": int((base_pred == 1).sum() - pred.sum()),
            "positives": int(pred.sum()),
            "ratio": float(pred.mean()),
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )


def diff_summary(base_df, pred_df, test_eval):
    merged = base_df[["id", "prediction"]].merge(pred_df, on="id", suffixes=("_base", "_new"))
    changed = merged[merged["prediction_base"] != merged["prediction_new"]].copy()
    if changed.empty:
        return {"changed_rows": 0}
    changed = changed.merge(test_eval[["id", "reason_text"]], on="id", how="left")
    reasons = Counter()
    for text in changed["reason_text"].fillna(""):
        for part in str(text).split("|"):
            if part:
                reasons[part] += 1
    return {
        "changed_rows": int(len(changed)),
        "top_reasons": dict(reasons.most_common(20)),
    }


def build_replacement_promotions(full_eval, variant_col):
    promoted_ids = []
    promotion_meta = []
    for term_id, grp in full_eval.groupby("term_id", sort=False):
        route = grp["query_intent"].iloc[0]
        family = grp["query_device_family"].iloc[0]
        rule = PROMOTION_RULES.get((route, family))
        if rule is None:
            continue

        current = grp[(grp["base_prediction"].eq(1)) & (grp[variant_col].eq(0))].copy()
        if rule["target"] == "accessory":
            kept_target = current[current["item_is_accessory"].eq(1) & current["item_device_family"].eq(family)]
            candidates = grp[
                grp["base_prediction"].eq(0)
                & grp[variant_col].eq(0)
                & grp["item_is_accessory"].eq(1)
                & grp["item_device_family"].eq(family)
                & grp["score"].ge(rule["min_score"])
            ].copy()
        else:
            kept_target = current[current["item_is_main_device"].eq(1) & current["item_device_family"].eq(family)]
            candidates = grp[
                grp["base_prediction"].eq(0)
                & grp[variant_col].eq(0)
                & grp["item_is_main_device"].eq(1)
                & grp["item_device_family"].eq(family)
                & grp["score"].ge(rule["min_score"])
            ].copy()

        if not kept_target.empty or candidates.empty:
            continue

        chosen = candidates.sort_values("score", ascending=False).iloc[0]
        promoted_ids.append(chosen["id"])
        promotion_meta.append(
            {
                "term_id": term_id,
                "query_intent": route,
                "query_device_family": family,
                "target": rule["target"],
                "id": chosen["id"],
                "item_id": chosen["item_id"],
                "score": float(chosen["score"]),
            }
        )
    return promoted_ids, promotion_meta


def main():
    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    test_rows = prepare_test_rows()
    valid_scores = compute_holdout_score()
    valid_rows = prepare_valid_rows(valid_scores)

    train_pairs = pd.read_csv(TRAIN_PAIRS, usecols=["term_id", "item_id"])
    train_pairs["term_id"] = train_pairs["term_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)

    brand_tokens = build_brand_token_set()

    needed_term_ids = set(test_rows["term_id"]) | set(valid_rows["term_id"]) | set(train_pairs["term_id"])
    needed_item_ids = set(test_rows["item_id"]) | set(valid_rows["item_id"]) | set(train_pairs["item_id"])

    query_info = build_query_info(needed_term_ids, brand_tokens)
    item_info = build_item_meta(needed_item_ids)
    family_stats = build_family_stats(train_pairs, query_info, item_info)

    valid_eval = evaluate_rows(valid_rows, query_info, item_info, family_stats)
    test_eval = evaluate_rows(test_rows, query_info, item_info, family_stats)
    valid_eval, test_eval = attach_attribute_rule_features(valid_eval, test_eval)

    derived_variants = {
        "precision_core_veto": ["gender_veto", "package_variant_veto", "compatibility_veto"],
        "precision_plus_head_veto": ["gender_veto", "package_variant_veto", "compatibility_veto", "head_conflict_veto"],
        "precision_plus_family_strict_veto": ["gender_veto", "package_variant_veto", "compatibility_veto", "family_veto_strict"],
        "precision_plus_family_balanced_veto": ["gender_veto", "package_variant_veto", "compatibility_veto", "family_veto_balanced"],
        "precision_full_strict_veto": [
            "gender_veto",
            "package_variant_veto",
            "compatibility_veto",
            "head_conflict_veto",
            "family_veto_strict",
        ],
        "precision_full_balanced_veto": [
            "gender_veto",
            "package_variant_veto",
            "compatibility_veto",
            "head_conflict_veto",
            "family_veto_balanced",
        ],
        "gender_package_veto": ["gender_veto", "package_variant_veto"],
    }
    safe_variants = {
        "safe_clean_core_veto": ["safe_attr_brand_veto", "main_accessory_veto", "package_variant_veto"],
        "safe_clean_core_plus_compat_veto": [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "compatibility_veto",
        ],
        "safe_head_augmented_veto": [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "head_conflict_veto",
        ],
        "safe_head_plus_compat_veto": [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "compatibility_veto",
            "head_conflict_veto",
        ],
        "safe_full_strict_veto": [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "head_conflict_veto",
            "family_veto_strict",
        ],
        "safe_full_balanced_veto": [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "head_conflict_veto",
            "family_veto_balanced",
        ],
        "safe_full_strict_plus_compat_veto": [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "compatibility_veto",
            "head_conflict_veto",
            "family_veto_strict",
        ],
        "safe_full_balanced_plus_compat_veto": [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "compatibility_veto",
            "head_conflict_veto",
            "family_veto_balanced",
        ],
    }
    for out_col, source_cols in derived_variants.items():
        valid_eval[out_col] = valid_eval[source_cols].max(axis=1).astype(np.int8)
        test_eval[out_col] = test_eval[source_cols].max(axis=1).astype(np.int8)
    for out_col, source_cols in safe_variants.items():
        valid_eval[out_col] = valid_eval[source_cols].max(axis=1).astype(np.int8)
        test_eval[out_col] = test_eval[source_cols].max(axis=1).astype(np.int8)
    valid_eval["full_strict_plus_safe_attr_brand_veto"] = (
        valid_eval[["full_veto_strict", "safe_attr_brand_veto"]].max(axis=1).astype(np.int8)
    )
    test_eval["full_strict_plus_safe_attr_brand_veto"] = (
        test_eval[["full_veto_strict", "safe_attr_brand_veto"]].max(axis=1).astype(np.int8)
    )
    valid_eval["full_balanced_plus_safe_attr_brand_veto"] = (
        valid_eval[["full_veto_balanced", "safe_attr_brand_veto"]].max(axis=1).astype(np.int8)
    )
    test_eval["full_balanced_plus_safe_attr_brand_veto"] = (
        test_eval[["full_veto_balanced", "safe_attr_brand_veto"]].max(axis=1).astype(np.int8)
    )

    replacement_term_ids = set(
        test_eval.loc[
            test_eval["main_accessory_veto"].eq(1) | test_eval["compatibility_veto"].eq(1),
            "term_id",
        ]
    )
    full_eval = pd.DataFrame()
    if replacement_term_ids:
        full_pool = prepare_full_test_pool(replacement_term_ids)
        extra_item_ids = set(full_pool["item_id"]) - set(item_info)
        if extra_item_ids:
            item_info.update(build_item_meta(extra_item_ids))
        full_eval = evaluate_rows(
            full_pool[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]],
            query_info,
            item_info,
            family_stats,
        ).merge(full_pool[["id", "base_prediction"]], on="id", how="left")
        for out_col, source_cols in derived_variants.items():
            full_eval[out_col] = full_eval[source_cols].max(axis=1).astype(np.int8)

    calib_cols = [
        "gender_score_veto",
        "color_score_veto",
        "gender_color_score_veto",
        "brand_tail_veto",
        "safe_attr_brand_veto",
        "gender_veto",
        "main_accessory_veto",
        "head_conflict_veto",
        "package_variant_veto",
        "compatibility_veto",
        "family_veto_strict",
        "family_veto_balanced",
        "clean_core_veto",
        "head_augmented_veto",
        "full_veto_strict",
        "full_veto_balanced",
        "precision_core_veto",
        "precision_plus_head_veto",
        "precision_plus_family_strict_veto",
        "precision_plus_family_balanced_veto",
        "precision_full_strict_veto",
        "precision_full_balanced_veto",
        "gender_package_veto",
        "safe_clean_core_veto",
        "safe_clean_core_plus_compat_veto",
        "safe_head_augmented_veto",
        "safe_head_plus_compat_veto",
        "safe_full_strict_veto",
        "safe_full_balanced_veto",
        "safe_full_strict_plus_compat_veto",
        "safe_full_balanced_plus_compat_veto",
        "full_strict_plus_safe_attr_brand_veto",
        "full_balanced_plus_safe_attr_brand_veto",
    ]
    calibration = [calibration_table(valid_eval, col) for col in calib_cols]

    test_hits = {col: int(test_eval[col].sum()) for col in calib_cols}
    reason_counts = Counter()
    for text in test_eval["reason_text"].fillna(""):
        for part in str(text).split("|"):
            if part:
                reason_counts[part] += 1

    ids = base["id"]
    base_pred = base["prediction"].to_numpy(np.int8)
    test_mask = test_eval.set_index("id")
    diff_eval = test_eval
    if not full_eval.empty:
        extra_diff = full_eval[~full_eval["id"].isin(set(test_eval["id"]))].copy()
        diff_eval = pd.concat([test_eval, extra_diff], ignore_index=True, sort=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    variant_map = {
        "clean_core": "clean_core_veto",
        "head_augmented": "head_augmented_veto",
        "full_strict": "full_veto_strict",
        "full_balanced": "full_veto_balanced",
        "safe_clean_core": "safe_clean_core_veto",
        "safe_clean_core_plus_compat": "safe_clean_core_plus_compat_veto",
        "safe_head_augmented": "safe_head_augmented_veto",
        "safe_head_plus_compat": "safe_head_plus_compat_veto",
        "safe_full_strict": "safe_full_strict_veto",
        "safe_full_balanced": "safe_full_balanced_veto",
        "safe_full_strict_plus_compat": "safe_full_strict_plus_compat_veto",
        "safe_full_balanced_plus_compat": "safe_full_balanced_plus_compat_veto",
        "full_strict_plus_safe_attr_brand": "full_strict_plus_safe_attr_brand_veto",
        "full_balanced_plus_safe_attr_brand": "full_balanced_plus_safe_attr_brand_veto",
        "precision_core": "precision_core_veto",
        "precision_plus_head": "precision_plus_head_veto",
        "precision_plus_family_strict": "precision_plus_family_strict_veto",
        "precision_plus_family_balanced": "precision_plus_family_balanced_veto",
        "precision_full_strict": "precision_full_strict_veto",
        "precision_full_balanced": "precision_full_balanced_veto",
        "gender_package": "gender_package_veto",
        "safe_attr_brand_only": "safe_attr_brand_veto",
        "gender_score_only": "gender_score_veto",
        "color_score_only": "color_score_veto",
        "gender_color_score_only": "gender_color_score_veto",
        "brand_tail_only": "brand_tail_veto",
        "gender_only": "gender_veto",
        "main_accessory_only": "main_accessory_veto",
        "head_only": "head_conflict_veto",
        "package_only": "package_variant_veto",
        "compatibility_only": "compatibility_veto",
        "family_strict_only": "family_veto_strict",
        "family_balanced_only": "family_veto_balanced",
    }
    replacement_variant_map = {
        "clean_core_replaced": "clean_core_veto",
        "full_strict_replaced": "full_veto_strict",
        "full_balanced_replaced": "full_veto_balanced",
    }

    diff_reports = {}
    replacement_reports = {}
    for name, col in variant_map.items():
        pred = base.copy()
        pred["prediction"] = pred["prediction"].astype(np.int8)
        veto_ids = set(test_mask.index[test_mask[col].eq(1)])
        if veto_ids:
            row_mask = pred["id"].isin(veto_ids) & pred["prediction"].eq(1)
            pred.loc[row_mask, "prediction"] = 0
        save_variant(ids, pred["prediction"].to_numpy(np.int8), name, base_pred, candidates)
        diff_reports[name] = diff_summary(base, pred, diff_eval)

    for name, col in replacement_variant_map.items():
        pred = base.copy()
        pred["prediction"] = pred["prediction"].astype(np.int8)
        veto_ids = set(test_mask.index[test_mask[col].eq(1)])
        if veto_ids:
            row_mask = pred["id"].isin(veto_ids) & pred["prediction"].eq(1)
            pred.loc[row_mask, "prediction"] = 0

        promoted_ids, promo_meta = ([], [])
        if not full_eval.empty:
            promoted_ids, promo_meta = build_replacement_promotions(full_eval, col)
            if promoted_ids:
                promote_mask = pred["id"].isin(set(promoted_ids)) & pred["prediction"].eq(0)
                pred.loc[promote_mask, "prediction"] = 1

        save_variant(ids, pred["prediction"].to_numpy(np.int8), name, base_pred, candidates)
        diff_reports[name] = diff_summary(base, pred, diff_eval)
        route_family = Counter((row["query_intent"], row["query_device_family"]) for row in promo_meta)
        replacement_reports[name] = {
            "promotions": int(len(promoted_ids)),
            "route_family_counts": {f"{route}:{family}": int(cnt) for (route, family), cnt in route_family.items()},
            "sample": promo_meta[:25],
        }

    report = {
        "base_file": str(BASE),
        "base_positives": int(base_pred.sum()),
        "holdout_calibration": calibration,
        "test_positive_hits": test_hits,
        "test_reason_counts": dict(reason_counts.most_common(40)),
        "family_stats_size": int(len(family_stats)),
        "replacement_query_count": int(len(replacement_term_ids)),
        "replacement_variants": replacement_reports,
        "candidates": candidates,
        "final_diff_vs_v95": diff_reports,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
