from pathlib import Path
import re
import json
import time
import unicodedata
import gc

import numpy as np
import pandas as pd


ROOT = Path(".")

PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

OUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports/manual_review"
CHUNK_DIR = OUT_DIR / "v38_lexical_fast_chunks"

OUT_PARQUET = OUT_DIR / "v38_lexical_pair_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v38_lexical_score_summary.json"
OUT_DEBUG = REPORT_DIR / "v38_lexical_debug_top_bottom.csv"

CHUNK_SIZE = 200_000


TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "bir", "adet", "set", "takim", "model", "uyumlu",
    "orjinal", "orijinal", "yeni", "renk", "boy", "numara", "beden",
    "cm", "mm", "lt", "kg", "gr", "ml", "x", "no", "olan", "veya",
    "plus", "pro", "max", "mini", "buyuk", "kucuk", "orta", "sade",
    "renkli", "urun", "da", "de", "mi", "mı", "mu", "mü",
}

COLOR_WORDS = {
    "siyah", "beyaz", "kirmizi", "mavi", "lacivert", "yesil", "sari",
    "pembe", "mor", "turuncu", "gri", "antrasit", "kahverengi", "bej",
    "krem", "gold", "gumus", "silver", "bordo", "lila", "ekru",
}

GENDER_AGE_WORDS = {
    "erkek", "kadin", "bayan", "kiz", "cocuk", "bebek", "unisex",
    "genc", "yetiskin",
}

PRODUCT_TYPE_WORDS = {
    "ayakkabi", "bot", "cizme", "sneaker", "terlik", "sandalet", "babet",
    "loafer", "hali", "saha", "pantolon", "gomlek", "elbise", "etek",
    "kazak", "mont", "ceket", "tshirt", "tisort", "jean", "tayt", "sort",
    "sweatshirt", "bluz", "hirka", "canta", "valiz", "bavul", "cuzdan",
    "telefon", "cep", "tablet", "laptop", "bilgisayar", "kulaklik",
    "kamera", "fotograf", "kilif", "kapak", "ekran", "koruyucu", "sarj",
    "kablo", "adaptor", "adapter", "lastik", "jant", "oto", "arac",
    "araba", "motosiklet", "paspas", "silecek", "cam", "suyu", "antifriz",
    "krem", "fondoten", "ruj", "sampuan", "parfum", "sac", "cilt",
    "serum", "wax", "masa", "sandalye", "koltuk", "perde", "dolap",
    "sehpa", "tablo", "kitap", "defter", "kalem", "oyuncak", "cikolata",
    "seker", "boncuk", "tesbih", "ram", "ssd", "ddr", "ddr4", "ddr5",
    "playstation", "ps2", "ps3", "ps4", "ps5", "makinesi", "makina",
    "cihazi", "cihaz", "aksesuar", "askisi", "tokasi", "toka",
}

CATEGORY_ALIAS = {
    "otomobil": "auto", "motosiklet": "auto", "lastik": "auto", "jant": "auto",
    "ayakkabi": "shoe", "giyim": "clothing", "aksesuar": "accessory",
    "kozmetik": "cosmetic", "kisisel": "cosmetic", "elektronik": "electronics",
    "telefon": "phone", "bilgisayar": "computer", "mobilya": "home", "ev": "home",
    "yapi": "hardware", "hirdavat": "hardware", "supermarket": "supermarket",
    "gida": "supermarket", "kitap": "book", "kirtasiye": "stationery",
    "ofis": "stationery", "spor": "sport", "outdoor": "sport", "hobi": "hobby",
    "eglence": "hobby",
}


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def toks_from_norm(s, keep_stop=False):
    if not s:
        return set()
    if keep_stop:
        return {t for t in s.split() if len(t) >= 2}
    return {t for t in s.split() if len(t) >= 2 and t not in STOP}


def nums_from_norm(s):
    return set(re.findall(r"\d+", s))


def model_tokens(tokens):
    out = set()
    for t in tokens:
        if re.search(r"[a-z]", t) and re.search(r"\d", t):
            out.add(t)
        if re.search(r"\d", t) and len(t) >= 2:
            out.add(t)
    return out


def chargrams(s, max_grams=70):
    if not s:
        return tuple()
    s = " " + s + " "
    out = []
    seen = set()
    for n in (3, 4, 5):
        if len(s) >= n:
            for i in range(len(s) - n + 1):
                g = s[i:i+n]
                if g not in seen:
                    out.append(g)
                    seen.add(g)
                    if len(out) >= max_grams:
                        return tuple(out)
    return tuple(out)


def cov(a, b):
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def exact_phrase(q_norm, text_norm):
    if not q_norm or not text_norm:
        return 0.0
    if len(q_norm.split()) < 2:
        return 0.0
    return 1.0 if q_norm in text_norm else 0.0


def cat_families(cat_norm):
    out = set()
    for k, v in CATEGORY_ALIAS.items():
        if k in cat_norm:
            out.add(v)
    return out


def q_families(q_norm):
    out = set()
    for k, v in CATEGORY_ALIAS.items():
        if k in q_norm:
            out.add(v)
    return out


def char_cov(qgrams, target_norm):
    if not qgrams:
        return 0.0
    hit = 0
    for g in qgrams:
        if g in target_norm:
            hit += 1
    return hit / max(1, len(qgrams))


def build_term_features(terms):
    rows = []
    for r in terms[["term_id", "query"]].itertuples(index=False):
        q_norm = norm_text(r.query)
        qt = toks_from_norm(q_norm)
        q_all = toks_from_norm(q_norm, keep_stop=True)

        must = set(qt) - STOP - COLOR_WORDS - GENDER_AGE_WORDS
        special = set(must) - PRODUCT_TYPE_WORDS
        nums = nums_from_norm(q_norm)
        models = model_tokens(q_all)

        rows.append({
            "term_id": str(r.term_id),
            "query": str(r.query),
            "q_norm": q_norm,
            "q_tokens": tuple(sorted(qt)),
            "q_all": tuple(sorted(q_all)),
            "must_tokens": tuple(sorted(must)),
            "special_tokens": tuple(sorted(special)),
            "q_nums": tuple(sorted(nums)),
            "q_models": tuple(sorted(models)),
            "q_grams": chargrams(q_norm),
            "q_family": tuple(sorted(q_families(q_norm))),
        })
    return pd.DataFrame(rows)


def score_rows(df):
    n = len(df)

    cols_float = [
        "v38_lex_score",
        "v38_exact_phrase_title", "v38_exact_phrase_full",
        "v38_word_overlap_title", "v38_word_overlap_category", "v38_word_overlap_full",
        "v38_word_jaccard_title", "v38_word_jaccard_full",
        "v38_char_ngram_title", "v38_char_ngram_full", "v38_char_ngram_sim",
        "v38_must_token_coverage_title", "v38_must_token_coverage_full",
        "v38_special_token_coverage_title", "v38_special_token_coverage_full",
        "v38_brand_match", "v38_brand_query_overlap",
        "v38_number_match", "v38_model_match",
        "v38_category_title_support", "v38_category_family_match",
        "v38_query_word_in_title_ratio", "v38_query_word_in_title_any",
        "v38_query_word_only_category_ratio", "v38_query_word_only_category_any",
        "v38_title_missing_ratio", "v38_must_missing_ratio", "v38_special_missing_ratio",
    ]

    cols_int = [
        "v38_title_token_count_hit", "v38_category_token_count_hit", "v38_full_token_count_hit",
        "v38_query_token_count", "v38_must_token_count", "v38_special_token_count",
    ]

    arr = {c: np.zeros(n, dtype=np.float32) for c in cols_float}
    arr.update({c: np.zeros(n, dtype=np.int16) for c in cols_int})

    for i, r in enumerate(df.itertuples(index=False)):
        q_tokens = set(r.q_tokens)
        must = set(r.must_tokens)
        special = set(r.special_tokens)
        q_nums = set(r.q_nums)
        q_models = set(r.q_models)
        qgrams = tuple(r.q_grams)
        qfam = set(r.q_family)

        title_norm = r.title_norm or ""
        cat_norm = r.category_norm or ""
        brand_norm = r.brand_norm or ""
        full_norm = r.full_norm or ""

        title_tokens = toks_from_norm(title_norm)
        cat_tokens = toks_from_norm(cat_norm)
        brand_tokens = toks_from_norm(brand_norm)
        full_tokens = toks_from_norm(full_norm)
        full_all = toks_from_norm(full_norm, keep_stop=True)

        title_hit = len(q_tokens & title_tokens)
        cat_hit = len(q_tokens & cat_tokens)
        full_hit = len(q_tokens & full_tokens)
        q_len = max(1, len(q_tokens))

        title_cov = title_hit / q_len
        cat_cov = cat_hit / q_len
        full_cov = full_hit / q_len

        exact_t = exact_phrase(r.q_norm, title_norm)
        exact_f = exact_phrase(r.q_norm, full_norm)

        char_t = char_cov(qgrams, title_norm)
        char_f = char_cov(qgrams, full_norm)
        char_sim = 0.72 * char_t + 0.28 * char_f

        must_title = cov(must, title_tokens)
        must_full = cov(must, full_tokens)
        special_title = cov(special, title_tokens)
        special_full = cov(special, full_tokens)

        brand_match = 0.0
        brand_overlap = 0.0
        if brand_norm:
            brand_match = float(brand_norm in r.q_norm or len(q_tokens & brand_tokens) > 0)
            brand_overlap = cov(brand_tokens, q_tokens)

        item_nums = nums_from_norm(full_norm)
        item_models = model_tokens(full_all)
        num_match = cov(q_nums, item_nums)
        model_match = cov(q_models, item_models)

        title_any = float(title_hit > 0)

        category_only = (q_tokens & cat_tokens) - title_tokens
        category_only_ratio = len(category_only) / q_len

        cat_title_both = float(title_hit > 0 and cat_hit > 0)
        cat_title_support = min(1.0, 0.70 * title_cov + 0.30 * cat_cov + 0.12 * cat_title_both)

        cfam = cat_families(cat_norm)
        cat_family_match = float(len(qfam & cfam) > 0) if qfam and cfam else 0.0

        must_missing = 1.0 - must_full if must else 0.0
        special_missing = 1.0 - special_full if special else 0.0

        score = (
            0.18 * exact_t +
            0.05 * exact_f +
            0.17 * title_cov +
            0.08 * full_cov +
            0.09 * jac(q_tokens, title_tokens) +
            0.04 * jac(q_tokens, full_tokens) +
            0.12 * char_sim +
            0.11 * must_full +
            0.07 * special_full +
            0.06 * brand_match +
            0.05 * num_match +
            0.06 * model_match +
            0.07 * cat_title_support +
            0.04 * cat_family_match +
            0.03 * title_any -
            0.09 * category_only_ratio -
            0.08 * must_missing -
            0.08 * special_missing
        )
        score = float(np.clip(score, -0.35, 1.25))

        arr["v38_lex_score"][i] = score
        arr["v38_exact_phrase_title"][i] = exact_t
        arr["v38_exact_phrase_full"][i] = exact_f

        arr["v38_word_overlap_title"][i] = title_cov
        arr["v38_word_overlap_category"][i] = cat_cov
        arr["v38_word_overlap_full"][i] = full_cov
        arr["v38_word_jaccard_title"][i] = jac(q_tokens, title_tokens)
        arr["v38_word_jaccard_full"][i] = jac(q_tokens, full_tokens)

        arr["v38_char_ngram_title"][i] = char_t
        arr["v38_char_ngram_full"][i] = char_f
        arr["v38_char_ngram_sim"][i] = char_sim

        arr["v38_must_token_coverage_title"][i] = must_title
        arr["v38_must_token_coverage_full"][i] = must_full
        arr["v38_special_token_coverage_title"][i] = special_title
        arr["v38_special_token_coverage_full"][i] = special_full

        arr["v38_brand_match"][i] = brand_match
        arr["v38_brand_query_overlap"][i] = brand_overlap
        arr["v38_number_match"][i] = num_match
        arr["v38_model_match"][i] = model_match

        arr["v38_category_title_support"][i] = cat_title_support
        arr["v38_category_family_match"][i] = cat_family_match

        arr["v38_query_word_in_title_ratio"][i] = title_cov
        arr["v38_query_word_in_title_any"][i] = title_any
        arr["v38_query_word_only_category_ratio"][i] = category_only_ratio
        arr["v38_query_word_only_category_any"][i] = float(category_only_ratio > 0 and title_cov == 0)

        arr["v38_title_missing_ratio"][i] = 1.0 - title_cov
        arr["v38_must_missing_ratio"][i] = must_missing
        arr["v38_special_missing_ratio"][i] = special_missing

        arr["v38_title_token_count_hit"][i] = title_hit
        arr["v38_category_token_count_hit"][i] = cat_hit
        arr["v38_full_token_count_hit"][i] = full_hit
        arr["v38_query_token_count"][i] = len(q_tokens)
        arr["v38_must_token_count"][i] = len(must)
        arr["v38_special_token_count"][i] = len(special)

    out = pd.DataFrame({
        "id": df["id"].astype(str).to_numpy(),
        "term_id": df["term_id"].astype(str).to_numpy(),
        "item_id": df["item_id"].astype(str).to_numpy(),
    })
    for c, v in arr.items():
        out[c] = v
    return out


def main():
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if CHUNK_DIR.exists():
        for p in CHUNK_DIR.glob("part_*.parquet"):
            p.unlink()
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    print("loading sample/terms/items...")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs_head = pd.read_csv(PAIRS, usecols=["term_id", "item_id"])
    pairs_head["term_id"] = pairs_head["term_id"].astype(str)
    pairs_head["item_id"] = pairs_head["item_id"].astype(str)
    used_terms = set(pairs_head["term_id"].unique())
    used_items = set(pairs_head["item_id"].unique())
    del pairs_head
    gc.collect()

    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    terms = terms[terms["term_id"].isin(used_terms)].copy()
    term_df = build_term_features(terms)
    print("terms:", len(term_df))

    items = pd.read_csv(ITEMS, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)
    items = items[items["item_id"].isin(used_items)].copy()
    print("items:", len(items))

    for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if c not in items.columns:
            items[c] = ""

    print("normalizing item text...")
    items_small = items[["item_id", "title", "category", "brand", "attributes"]].copy()
    items_small["title_norm"] = items_small["title"].map(norm_text)
    items_small["category_norm"] = items_small["category"].map(norm_text)
    items_small["brand_norm"] = items_small["brand"].map(norm_text)
    attr_norm = items_small["attributes"].fillna("").astype(str).str.slice(0, 400).map(norm_text)
    items_small["full_norm"] = (
        items_small["title_norm"].fillna("") + " " +
        items_small["category_norm"].fillna("") + " " +
        items_small["brand_norm"].fillna("") + " " +
        attr_norm.fillna("")
    ).str.strip()

    items_small = items_small[["item_id", "title_norm", "category_norm", "brand_norm", "full_norm"]]
    del items
    gc.collect()

    print("processing pair chunks...")
    part_paths = []
    total_rows = 0

    for chunk_id, chunk in enumerate(pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"], chunksize=CHUNK_SIZE)):
        chunk["id"] = chunk["id"].astype(str)
        chunk["term_id"] = chunk["term_id"].astype(str)
        chunk["item_id"] = chunk["item_id"].astype(str)

        df = chunk.merge(term_df, on="term_id", how="left", validate="many_to_one")
        df = df.merge(items_small, on="item_id", how="left", validate="many_to_one")

        for c in ["q_norm", "title_norm", "category_norm", "brand_norm", "full_norm"]:
            df[c] = df[c].fillna("").astype(str)

        for c in ["q_tokens", "q_all", "must_tokens", "special_tokens", "q_nums", "q_models", "q_grams", "q_family"]:
            df[c] = df[c].apply(lambda x: x if isinstance(x, tuple) else tuple())

        scored = score_rows(df)

        part = CHUNK_DIR / f"part_{chunk_id:04d}.parquet"
        scored.to_parquet(part, index=False)
        part_paths.append(part)

        total_rows += len(scored)
        elapsed = (time.time() - t0) / 60
        print(f"chunk={chunk_id} rows={total_rows:,} elapsed_min={elapsed:.2f}")

        del chunk, df, scored
        gc.collect()

    print("concatenating parts...")
    out = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)

    if not out["id"].astype(str).reset_index(drop=True).equals(sample["id"].astype(str).reset_index(drop=True)):
        raise RuntimeError("output id order mismatch sample_submission")

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

    print("saving final:", OUT_PARQUET)
    out.to_parquet(OUT_PARQUET, index=False)

    debug = out[["id", "term_id", "item_id", "v38_lex_score", "v38_lex_rank", "v38_lex_pct_rank"]].copy()
    pd.concat([
        debug.sort_values("v38_lex_score", ascending=False).head(1000).assign(bucket="top"),
        debug.sort_values("v38_lex_score", ascending=True).head(1000).assign(bucket="bottom"),
    ], ignore_index=True).to_csv(OUT_DEBUG, index=False)

    elapsed = (time.time() - t0) / 60
    summary = {
        "output": str(OUT_PARQUET),
        "rows": int(len(out)),
        "unique_terms": int(out["term_id"].nunique()),
        "unique_items": int(out["item_id"].nunique()),
        "score_min": float(out["v38_lex_score"].min()),
        "score_mean": float(out["v38_lex_score"].mean()),
        "score_std": float(out["v38_lex_score"].std()),
        "score_max": float(out["v38_lex_score"].max()),
        "exact_phrase_title_rate": float(out["v38_exact_phrase_title"].mean()),
        "query_word_title_any_rate": float(out["v38_query_word_in_title_any"].mean()),
        "query_word_only_category_any_rate": float(out["v38_query_word_only_category_any"].mean()),
        "elapsed_min": float(elapsed),
        "chunk_size": CHUNK_SIZE,
    }

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("done")


if __name__ == "__main__":
    main()
