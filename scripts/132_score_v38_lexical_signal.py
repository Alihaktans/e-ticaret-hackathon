from pathlib import Path
import re
import json
import time
import math
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd


ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports/manual_review"

OUT_PARQUET = OUT_DIR / "v38_lexical_pair_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v38_lexical_score_summary.json"
OUT_DEBUG = REPORT_DIR / "v38_lexical_debug_top_bottom.csv"


TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "için", "bir", "adet", "set", "takim", "takım",
    "model", "uyumlu", "orjinal", "orijinal", "yeni", "renk", "boy",
    "numara", "beden", "cm", "mm", "lt", "kg", "gr", "ml", "x", "no",
    "olan", "veya", "plus", "pro", "max", "mini", "buyuk", "büyük",
    "kucuk", "küçük", "orta", "sade", "renkli", "urun", "ürün",
    "icin", "için", "da", "de", "mi", "mı", "mu", "mü",
}

COLOR_WORDS = {
    "siyah", "beyaz", "kirmizi", "kırmızı", "mavi", "lacivert", "yesil", "yeşil",
    "sari", "sarı", "pembe", "mor", "turuncu", "gri", "antrasit", "kahverengi",
    "bej", "krem", "gold", "gumus", "gümüş", "silver", "bordo", "lila", "ekru",
}

GENDER_AGE_WORDS = {
    "erkek", "kadin", "kadın", "bayan", "kiz", "kız", "cocuk", "çocuk",
    "bebek", "unisex", "genc", "genç", "yetiskin", "yetişkin",
}

# These are product-type tokens. They are still useful for product matching,
# but brand-like / special-token matching ignores many of them.
PRODUCT_TYPE_WORDS = {
    "ayakkabi", "ayakkabı", "bot", "cizme", "çizme", "sneaker", "terlik",
    "sandalet", "babet", "loafer", "hali", "halı", "saha",
    "pantolon", "gomlek", "gömlek", "elbise", "etek", "kazak", "mont",
    "ceket", "tshirt", "tisort", "tişört", "jean", "tayt", "sort", "şort",
    "sweatshirt", "bluz", "hirka", "hırka",
    "canta", "çanta", "valiz", "bavul", "cuzdan", "cüzdan",
    "telefon", "cep", "tablet", "laptop", "bilgisayar", "kulaklik", "kulaklık",
    "kamera", "fotograf", "fotoğraf", "kilif", "kılıf", "kapak", "ekran",
    "koruyucu", "sarj", "şarj", "kablo", "adaptör", "adapter",
    "lastik", "jant", "oto", "arac", "araç", "araba", "motosiklet",
    "paspas", "silecek", "cam", "suyu", "antifriz",
    "krem", "fondoten", "fondöten", "ruj", "sampuan", "şampuan",
    "parfum", "parfüm", "sac", "saç", "cilt", "serum", "wax",
    "masa", "sandalye", "koltuk", "perde", "dolap", "sehpa", "tablo",
    "kitap", "defter", "kalem", "oyuncak", "cikolata", "çikolata",
    "seker", "şeker", "boncuk", "tesbih", "ram", "ssd", "ddr", "ddr4",
    "ddr5", "playstation", "ps2", "ps3", "ps4", "ps5",
    "makinesi", "makina", "cihazi", "cihaz", "aksesuar", "askisi",
    "askısı", "tokasi", "tokası", "toka",
}

CATEGORY_ALIAS = {
    "otomobil": "auto",
    "motosiklet": "auto",
    "lastik": "auto",
    "jant": "auto",
    "ayakkabi": "shoe",
    "giyim": "clothing",
    "aksesuar": "accessory",
    "kozmetik": "cosmetic",
    "kisisel": "cosmetic",
    "elektronik": "electronics",
    "telefon": "phone",
    "bilgisayar": "computer",
    "mobilya": "home",
    "ev": "home",
    "yapi": "hardware",
    "hirdavat": "hardware",
    "supermarket": "supermarket",
    "gida": "supermarket",
    "kitap": "book",
    "kirtasiye": "stationery",
    "ofis": "stationery",
    "spor": "sport",
    "outdoor": "sport",
    "hobi": "hobby",
    "eglence": "hobby",
}


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def simple_stem(t: str) -> str:
    # Small Turkish-ish suffix cleanup after TR normalization.
    # This is intentionally conservative.
    if len(t) <= 4:
        return t

    suffixes = [
        "larimiz", "lerimiz", "lariniz", "leriniz",
        "lari", "leri", "ları", "leri",
        "imiz", "ımız", "umuz", "ümüz",
        "iniz", "ınız", "unuz", "ünüz",
        "nin", "nın", "nun", "nün",
        "dan", "den", "tan", "ten",
        "dir", "dır", "dur", "dür",
        "lik", "lık", "luk", "lük",
        "si", "sı", "su", "sü",
        "yi", "yı", "yu", "yü",
        "i", "ı", "u", "ü",
    ]

    for s in suffixes:
        ns = norm_text(s)
        if len(t) - len(ns) >= 3 and t.endswith(ns):
            return t[:-len(ns)]
    return t


def token_set(x: object, keep_stop: bool = False) -> set[str]:
    s = norm_text(x)
    out = set()
    for raw in s.split():
        if len(raw) < 2:
            continue
        if (not keep_stop) and raw in STOP:
            continue
        out.add(raw)
        st = simple_stem(raw)
        if len(st) >= 2:
            out.add(st)
    return out


def char_ngrams(s: str, ns=(3, 4, 5), max_grams=80) -> tuple[str, ...]:
    s = norm_text(s)
    if not s:
        return tuple()
    s2 = f" {s} "
    grams = []
    for n in ns:
        if len(s2) >= n:
            grams.extend(s2[i:i+n] for i in range(len(s2) - n + 1))
    # Unique but stable order
    seen = set()
    uniq = []
    for g in grams:
        if g not in seen:
            uniq.append(g)
            seen.add(g)
        if len(uniq) >= max_grams:
            break
    return tuple(uniq)


def number_tokens(x: object) -> set[str]:
    return set(re.findall(r"\d+", norm_text(x)))


def model_tokens_from_tokens(tokens: set[str]) -> set[str]:
    out = set()
    for t in tokens:
        has_alpha = bool(re.search(r"[a-z]", t))
        has_digit = bool(re.search(r"\d", t))
        if has_alpha and has_digit:
            out.add(t)
        # Keep numeric model-ish tokens too; 64, 128, 256, 2025, etc.
        if has_digit and len(t) >= 2:
            out.add(t)
    return out


def coverage(q: set[str], target: set[str]) -> float:
    if not q:
        return 0.0
    return len(q & target) / max(1, len(q))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def ratio_intersection(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def exact_phrase_hit(q_norm: str, text_norm: str) -> int:
    if not q_norm or not text_norm:
        return 0
    # Single-word exact phrase is just token match, handled elsewhere.
    if len(q_norm.split()) < 2:
        return 0
    return int(q_norm in text_norm)


def category_family_tokens(cat_norm: str) -> set[str]:
    out = set()
    for k, v in CATEGORY_ALIAS.items():
        if k in cat_norm:
            out.add(v)
    return out


def pick_col(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def safe_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


def build_query_info(terms: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for r in terms[["term_id", "query"]].itertuples(index=False):
        term_id = str(r.term_id)
        q = str(r.query)
        q_norm = norm_text(q)
        q_tokens = token_set(q)
        q_tokens_all = token_set(q, keep_stop=True)

        must = set(q_tokens)
        must -= {norm_text(x) for x in STOP}
        must -= {norm_text(x) for x in COLOR_WORDS}
        must -= {norm_text(x) for x in GENDER_AGE_WORDS}

        special = set(must)
        special -= {norm_text(x) for x in PRODUCT_TYPE_WORDS}

        nums = number_tokens(q)
        models = model_tokens_from_tokens(q_tokens_all)

        out[term_id] = {
            "query": q,
            "q_norm": q_norm,
            "q_tokens": q_tokens,
            "q_tokens_all": q_tokens_all,
            "must": must,
            "special": special,
            "nums": nums,
            "models": models,
            "chargrams": char_ngrams(q_norm),
            "q_len": len(q_tokens),
            "must_len": len(must),
            "special_len": len(special),
            "num_len": len(nums),
            "model_len": len(models),
        }
    return out


def build_item_info(items: pd.DataFrame) -> dict[str, dict]:
    title_col = pick_col(items, ["title", "item_title", "name", "item_name", "product_name"])
    cat_col = pick_col(items, ["category", "category_name", "category_path", "category_tree"])
    brand_col = pick_col(items, ["brand", "brand_name"])
    gender_col = pick_col(items, ["gender", "gender_name"])
    age_col = pick_col(items, ["age_group", "age", "ageGroup"])
    attr_col = pick_col(items, ["attributes", "attribute", "attrs", "description", "desc"])

    print("item columns used:", {
        "title": title_col,
        "category": cat_col,
        "brand": brand_col,
        "gender": gender_col,
        "age_group": age_col,
        "attributes": attr_col,
    })

    title_s = safe_series(items, title_col)
    cat_s = safe_series(items, cat_col)
    brand_s = safe_series(items, brand_col)
    gender_s = safe_series(items, gender_col)
    age_s = safe_series(items, age_col)
    attr_s = safe_series(items, attr_col)

    out = {}

    for item_id, title, cat, brand, gender, age, attr in zip(
        items["item_id"].astype(str).to_numpy(),
        title_s.to_numpy(),
        cat_s.to_numpy(),
        brand_s.to_numpy(),
        gender_s.to_numpy(),
        age_s.to_numpy(),
        attr_s.to_numpy(),
    ):
        title_norm = norm_text(title)
        cat_norm = norm_text(cat)
        brand_norm = norm_text(brand)
        gender_norm = norm_text(gender)
        age_norm = norm_text(age)
        attr_norm = norm_text(attr)

        full_norm = " ".join(x for x in [title_norm, cat_norm, brand_norm, gender_norm, age_norm, attr_norm] if x)

        title_tokens = token_set(title_norm)
        cat_tokens = token_set(cat_norm)
        brand_tokens = token_set(brand_norm)
        full_tokens = token_set(full_norm)

        out[str(item_id)] = {
            "title_norm": title_norm,
            "cat_norm": cat_norm,
            "brand_norm": brand_norm,
            "full_norm": full_norm,
            "title_tokens": title_tokens,
            "cat_tokens": cat_tokens,
            "brand_tokens": brand_tokens,
            "full_tokens": full_tokens,
            "nums": number_tokens(full_norm),
            "models": model_tokens_from_tokens(token_set(full_norm, keep_stop=True)),
            "cat_family": category_family_tokens(cat_norm),
        }

    return out


def chargram_coverage(q_grams: tuple[str, ...], title_norm: str, full_norm: str) -> tuple[float, float]:
    if not q_grams:
        return 0.0, 0.0

    title_hits = 0
    full_hits = 0
    for g in q_grams:
        if g in title_norm:
            title_hits += 1
        if g in full_norm:
            full_hits += 1

    denom = max(1, len(q_grams))
    return title_hits / denom, full_hits / denom


def score_pair(qi: dict, ii: dict) -> dict:
    q_tokens = qi["q_tokens"]
    must = qi["must"]
    special = qi["special"]
    nums = qi["nums"]
    models = qi["models"]

    title_tokens = ii["title_tokens"]
    cat_tokens = ii["cat_tokens"]
    brand_tokens = ii["brand_tokens"]
    full_tokens = ii["full_tokens"]

    q_len = max(1, len(q_tokens))

    title_overlap_cnt = len(q_tokens & title_tokens)
    cat_overlap_cnt = len(q_tokens & cat_tokens)
    full_overlap_cnt = len(q_tokens & full_tokens)

    title_cov = title_overlap_cnt / q_len
    cat_cov = cat_overlap_cnt / q_len
    full_cov = full_overlap_cnt / q_len

    word_jacc_title = jaccard(q_tokens, title_tokens)
    word_jacc_full = jaccard(q_tokens, full_tokens)

    exact_title = exact_phrase_hit(qi["q_norm"], ii["title_norm"])
    exact_full = exact_phrase_hit(qi["q_norm"], ii["full_norm"])

    char_title, char_full = chargram_coverage(qi["chargrams"], ii["title_norm"], ii["full_norm"])
    char_sim = 0.72 * char_title + 0.28 * char_full

    must_title_cov = coverage(must, title_tokens)
    must_full_cov = coverage(must, full_tokens)
    special_title_cov = coverage(special, title_tokens)
    special_full_cov = coverage(special, full_tokens)

    brand_match = 0.0
    brand_query_overlap = 0.0
    if ii["brand_norm"]:
        # Full phrase brand in query, or brand token overlap.
        brand_match = float(ii["brand_norm"] in qi["q_norm"] or len(q_tokens & brand_tokens) > 0)
        brand_query_overlap = ratio_intersection(brand_tokens, q_tokens)

    num_match = coverage(nums, ii["nums"])
    model_match = coverage(models, ii["models"])

    title_hit_any = float(title_overlap_cnt > 0)
    cat_hit_any = float(cat_overlap_cnt > 0)
    full_hit_any = float(full_overlap_cnt > 0)

    category_only_tokens = (q_tokens & cat_tokens) - title_tokens
    category_only_ratio = len(category_only_tokens) / q_len

    title_missing_ratio = 1.0 - title_cov
    must_missing_ratio = 1.0 - must_full_cov if must else 0.0
    special_missing_ratio = 1.0 - special_full_cov if special else 0.0

    cat_title_both = float(title_overlap_cnt > 0 and cat_overlap_cnt > 0)
    category_title_support = min(1.0, 0.70 * title_cov + 0.30 * cat_cov + 0.12 * cat_title_both)

    category_family_match = 0.0
    if ii["cat_family"]:
        # Infer very rough query family from query tokens through category aliases.
        qfam = set()
        for k, v in CATEGORY_ALIAS.items():
            if k in qi["q_norm"]:
                qfam.add(v)
        if qfam:
            category_family_match = float(len(qfam & ii["cat_family"]) > 0)

    # Query word title/category location.
    query_word_in_title_ratio = title_cov
    query_word_only_category_ratio = category_only_ratio
    query_word_only_category_any = float(category_only_ratio > 0 and title_cov == 0)

    # Main lexical score.
    # "category-only" is penalized because title usually carries the real product identity.
    score = (
        0.18 * exact_title +
        0.05 * exact_full +
        0.17 * title_cov +
        0.08 * full_cov +
        0.09 * word_jacc_title +
        0.04 * word_jacc_full +
        0.12 * char_sim +
        0.11 * must_full_cov +
        0.07 * special_full_cov +
        0.06 * brand_match +
        0.05 * num_match +
        0.06 * model_match +
        0.07 * category_title_support +
        0.04 * category_family_match +
        0.03 * title_hit_any -
        0.09 * query_word_only_category_ratio -
        0.08 * must_missing_ratio -
        0.08 * special_missing_ratio
    )

    # Keep score in a stable range but allow negatives for very weak matches.
    score = float(np.clip(score, -0.35, 1.25))

    return {
        "v38_lex_score": score,

        # Requested feature groups
        "v38_exact_phrase_title": float(exact_title),
        "v38_exact_phrase_full": float(exact_full),

        "v38_word_overlap_title": float(title_cov),
        "v38_word_overlap_category": float(cat_cov),
        "v38_word_overlap_full": float(full_cov),
        "v38_word_jaccard_title": float(word_jacc_title),
        "v38_word_jaccard_full": float(word_jacc_full),

        "v38_char_ngram_title": float(char_title),
        "v38_char_ngram_full": float(char_full),
        "v38_char_ngram_sim": float(char_sim),

        "v38_must_token_coverage_title": float(must_title_cov),
        "v38_must_token_coverage_full": float(must_full_cov),
        "v38_special_token_coverage_title": float(special_title_cov),
        "v38_special_token_coverage_full": float(special_full_cov),

        "v38_brand_match": float(brand_match),
        "v38_brand_query_overlap": float(brand_query_overlap),
        "v38_number_match": float(num_match),
        "v38_model_match": float(model_match),

        "v38_category_title_support": float(category_title_support),
        "v38_category_family_match": float(category_family_match),

        "v38_query_word_in_title_ratio": float(query_word_in_title_ratio),
        "v38_query_word_in_title_any": float(title_hit_any),
        "v38_query_word_only_category_ratio": float(query_word_only_category_ratio),
        "v38_query_word_only_category_any": float(query_word_only_category_any),

        # Useful diagnostics
        "v38_title_missing_ratio": float(title_missing_ratio),
        "v38_must_missing_ratio": float(must_missing_ratio),
        "v38_special_missing_ratio": float(special_missing_ratio),
        "v38_title_token_count_hit": int(title_overlap_cnt),
        "v38_category_token_count_hit": int(cat_overlap_cnt),
        "v38_full_token_count_hit": int(full_overlap_cnt),
        "v38_query_token_count": int(len(q_tokens)),
        "v38_must_token_count": int(len(must)),
        "v38_special_token_count": int(len(special)),
    }


def main():
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading sample/pairs...")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("submission_pairs id order does not match sample_submission")

    print("loading terms/items...")
    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)

    items = pd.read_csv(ITEMS, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)

    # Only keep rows that appear in submission_pairs to reduce preprocessing.
    used_terms = set(pairs["term_id"].unique())
    used_items = set(pairs["item_id"].unique())

    terms = terms[terms["term_id"].isin(used_terms)].copy()
    items = items[items["item_id"].isin(used_items)].copy()

    if "query" not in terms.columns:
        raise RuntimeError("terms.csv must contain a 'query' column")

    print("unique terms:", len(terms), "unique items:", len(items))

    print("precomputing query info...")
    q_info = build_query_info(terms)

    print("precomputing item info...")
    item_info = build_item_info(items)

    feature_names = [
        "v38_lex_score",

        "v38_exact_phrase_title",
        "v38_exact_phrase_full",

        "v38_word_overlap_title",
        "v38_word_overlap_category",
        "v38_word_overlap_full",
        "v38_word_jaccard_title",
        "v38_word_jaccard_full",

        "v38_char_ngram_title",
        "v38_char_ngram_full",
        "v38_char_ngram_sim",

        "v38_must_token_coverage_title",
        "v38_must_token_coverage_full",
        "v38_special_token_coverage_title",
        "v38_special_token_coverage_full",

        "v38_brand_match",
        "v38_brand_query_overlap",
        "v38_number_match",
        "v38_model_match",

        "v38_category_title_support",
        "v38_category_family_match",

        "v38_query_word_in_title_ratio",
        "v38_query_word_in_title_any",
        "v38_query_word_only_category_ratio",
        "v38_query_word_only_category_any",

        "v38_title_missing_ratio",
        "v38_must_missing_ratio",
        "v38_special_missing_ratio",
        "v38_title_token_count_hit",
        "v38_category_token_count_hit",
        "v38_full_token_count_hit",
        "v38_query_token_count",
        "v38_must_token_count",
        "v38_special_token_count",
    ]

    n = len(pairs)

    float_features = [c for c in feature_names if not c.endswith("_count") and not c.endswith("_hit")]
    int_features = [c for c in feature_names if c.endswith("_count") or c.endswith("_hit")]

    arrays = {}
    for c in feature_names:
        if c in int_features:
            arrays[c] = np.zeros(n, dtype=np.int16)
        else:
            arrays[c] = np.zeros(n, dtype=np.float32)

    print("scoring pairs:", n)
    missing_q = 0
    missing_item = 0

    term_ids = pairs["term_id"].to_numpy()
    item_ids = pairs["item_id"].to_numpy()

    progress_every = 250_000

    for i, (tid, iid) in enumerate(zip(term_ids, item_ids)):
        qi = q_info.get(str(tid))
        ii = item_info.get(str(iid))

        if qi is None:
            missing_q += 1
            continue
        if ii is None:
            missing_item += 1
            continue

        f = score_pair(qi, ii)

        for c, v in f.items():
            arrays[c][i] = v

        if (i + 1) % progress_every == 0:
            elapsed = (time.time() - t0) / 60
            print(f"scored {i+1:,}/{n:,} elapsed_min={elapsed:.2f}")

    print("building output dataframe...")
    out = pd.DataFrame({
        "id": pairs["id"].to_numpy(),
        "term_id": pairs["term_id"].to_numpy(),
        "item_id": pairs["item_id"].to_numpy(),
    })

    for c in feature_names:
        out[c] = arrays[c]

    print("ranking within term...")
    out["v38_lex_rank"] = (
        out.groupby("term_id")["v38_lex_score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )

    cnt = out.groupby("term_id")["id"].transform("count").astype(np.float32)
    out["v38_lex_pct_rank"] = ((out["v38_lex_rank"].astype(np.float32) - 1.0) / np.maximum(1.0, cnt - 1.0)).astype(np.float32)

    term_mean = out.groupby("term_id")["v38_lex_score"].transform("mean").astype(np.float32)
    term_std = out.groupby("term_id")["v38_lex_score"].transform("std").fillna(0).astype(np.float32)
    out["v38_lex_term_z"] = ((out["v38_lex_score"].astype(np.float32) - term_mean) / np.maximum(1e-6, term_std)).astype(np.float32)

    # Save with id first. Keeping term_id/item_id helps later candidate scripts.
    print("saving:", OUT_PARQUET)
    out.to_parquet(OUT_PARQUET, index=False)

    # Debug sample: top and bottom by lexical score
    print("saving debug sample...")
    debug = out[["id", "term_id", "item_id", "v38_lex_score", "v38_lex_rank", "v38_lex_pct_rank"]].copy()
    debug_top = debug.sort_values("v38_lex_score", ascending=False).head(1000)
    debug_bottom = debug.sort_values("v38_lex_score", ascending=True).head(1000)
    pd.concat([debug_top.assign(bucket="top"), debug_bottom.assign(bucket="bottom")], ignore_index=True).to_csv(OUT_DEBUG, index=False)

    elapsed = (time.time() - t0) / 60

    summary = {
        "output": str(OUT_PARQUET),
        "rows": int(len(out)),
        "unique_terms": int(out["term_id"].nunique()),
        "unique_items": int(out["item_id"].nunique()),
        "missing_query_rows": int(missing_q),
        "missing_item_rows": int(missing_item),
        "score_min": float(out["v38_lex_score"].min()),
        "score_mean": float(out["v38_lex_score"].mean()),
        "score_std": float(out["v38_lex_score"].std()),
        "score_max": float(out["v38_lex_score"].max()),
        "exact_phrase_title_rate": float(out["v38_exact_phrase_title"].mean()),
        "query_word_title_any_rate": float(out["v38_query_word_in_title_any"].mean()),
        "query_word_only_category_any_rate": float(out["v38_query_word_only_category_any"].mean()),
        "elapsed_min": float(elapsed),
    }

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("done")


if __name__ == "__main__":
    main()
