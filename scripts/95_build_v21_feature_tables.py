from pathlib import Path
from functools import lru_cache
import re
import time
import numpy as np
import pandas as pd

ROOT = Path(".")

TRAIN_CAND = ROOT / "data/processed/v21_train_candidates.parquet"
SUB_PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT_TRAIN = ROOT / "data/processed/v21_train_features.parquet"
OUT_TEST = ROOT / "data/processed/v21_test_features.parquet"
FEATURE_LIST = ROOT / "data/processed/v21_feature_columns.txt"

REPORT = ROOT / "reports/manual_review/v21_feature_build_report.csv"

CHUNK_SIZE = 350000

TR_MAP = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")

STOPWORDS = {
    "ve", "ile", "icin", "için", "bir", "bu", "de", "da",
    "the", "and", "or", "in", "on", "of", "to", "for", "a", "an",
    "adet", "li", "lu", "lü", "lı"
}

COLORS = {
    "siyah", "beyaz", "kirmizi", "kırmızı", "mavi", "lacivert", "yesil", "yeşil",
    "sari", "sarı", "pembe", "mor", "turuncu", "gri", "bej", "kahverengi",
    "bordo", "ekru", "krem", "haki", "füme", "fume", "gold", "gümüş", "gumus",
    "renkli", "transparent", "seffaf", "şeffaf"
}

FEMALE_WORDS = {
    "kadin", "kadın", "bayan", "kiz", "kız", "tesettur", "tesettür"
}

MALE_WORDS = {
    "erkek", "bay", "adam"
}

CHILD_WORDS = {
    "cocuk", "çocuk", "bebek", "baby", "kids", "kid", "genc", "genç"
}

SIZE_WORDS = {
    "xs", "s", "m", "l", "xl", "xxl", "xxxl",
    "small", "medium", "large",
    "36", "37", "38", "39", "40", "41", "42", "43", "44", "45",
    "46", "48", "50", "52", "54", "56"
}

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

@lru_cache(maxsize=400000)
def token_tuple_cached(s):
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    out = []
    for t in s.split():
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return tuple(out)

@lru_cache(maxsize=400000)
def token_set_cached(s):
    return frozenset(token_tuple_cached(s))

@lru_cache(maxsize=400000)
def digit_set_cached(s):
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return frozenset(re.findall(r"\d+", s))

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def root_category(x):
    s = "" if pd.isna(x) else str(x)
    for sep in [">", "/", "|"]:
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()

def contains_phrase(text, phrase):
    if not phrase:
        return False
    return (" " + phrase + " ") in (" " + text + " ")

def detect_flags(qset, qnorm):
    q_female = int(bool(qset & FEMALE_WORDS))
    q_male = int(bool(qset & MALE_WORDS))
    q_child = int(bool(qset & CHILD_WORDS))
    q_color = int(bool(qset & COLORS))
    q_size = int(bool(qset & SIZE_WORDS))
    q_digit = int(bool(digit_set_cached(qnorm)))
    return q_female, q_male, q_child, q_color, q_size, q_digit

def build_brand_detector(items):
    brand_counts = items["brand_norm"].value_counts()
    brands = []
    for b, cnt in brand_counts.items():
        if not isinstance(b, str) or not b:
            continue
        if cnt < 15:
            continue
        ts = token_tuple_cached(b)
        if not ts:
            continue
        if len(b) < 3:
            continue
        brands.append((b, ts, cnt))

    first_map = {}
    for b, ts, cnt in brands:
        first = ts[0]
        first_map.setdefault(first, []).append((b, ts, cnt))

    for k in list(first_map.keys()):
        first_map[k] = sorted(
            first_map[k],
            key=lambda x: (len(x[1]), len(x[0]), x[2]),
            reverse=True
        )[:300]

    return first_map

def detect_query_brand(qnorm, first_map):
    qset = token_set_cached(qnorm)
    best = ""
    best_len = 0

    for t in qset:
        for b, btoks, cnt in first_map.get(t, []):
            ok = True
            for bt in btoks:
                if bt not in qset:
                    ok = False
                    break
            if ok and len(b) > best_len:
                best = b
                best_len = len(b)

    return best

def compute_pair_features(
    qnorm,
    qbrand_detected,
    title_norm,
    category_norm,
    brand_norm,
    gender_norm,
    age_group_norm,
    attr_norm,
    item_train_pos_count,
    brand_train_pos_count,
    root_train_pos_count,
):
    qset = token_set_cached(qnorm)
    tset = token_set_cached(title_norm)
    cset = token_set_cached(category_norm)
    bset = token_set_cached(brand_norm)
    aset = token_set_cached(attr_norm)

    q_len = len(qset)
    t_len = len(tset)
    c_len = len(cset)
    b_len = len(bset)
    a_len = len(aset)

    title_overlap = len(qset & tset)
    cat_overlap = len(qset & cset)
    brand_overlap = len(qset & bset)
    attr_overlap = len(qset & aset)

    product_set = tset | cset | bset | aset
    total_overlap = len(qset & product_set)

    title_cov = safe_div(title_overlap, q_len)
    cat_cov = safe_div(cat_overlap, q_len)
    brand_cov = safe_div(brand_overlap, q_len)
    attr_cov = safe_div(attr_overlap, q_len)
    total_cov = safe_div(total_overlap, q_len)

    title_jacc = safe_div(title_overlap, len(qset | tset))
    cat_jacc = safe_div(cat_overlap, len(qset | cset))
    brand_jacc = safe_div(brand_overlap, len(qset | bset))
    attr_jacc = safe_div(attr_overlap, len(qset | aset))
    total_jacc = safe_div(total_overlap, len(qset | product_set))

    weighted_overlap = (
        3.0 * title_overlap
        + 2.5 * brand_overlap
        + 1.5 * cat_overlap
        + 0.7 * attr_overlap
    )

    q_digits = digit_set_cached(qnorm)
    prod_digits = (
        digit_set_cached(title_norm)
        | digit_set_cached(category_norm)
        | digit_set_cached(brand_norm)
        | digit_set_cached(attr_norm)
    )

    digit_overlap = len(q_digits & prod_digits)
    digit_mismatch = int(len(q_digits) > 0 and digit_overlap == 0)
    digit_cov = safe_div(digit_overlap, len(q_digits))

    q_colors = qset & COLORS
    prod_colors = product_set & COLORS
    color_overlap = len(q_colors & prod_colors)
    color_mismatch = int(len(q_colors) > 0 and color_overlap == 0)
    color_cov = safe_div(color_overlap, len(q_colors))

    q_sizes = qset & SIZE_WORDS
    prod_sizes = product_set & SIZE_WORDS
    size_overlap = len(q_sizes & prod_sizes)
    size_mismatch = int(len(q_sizes) > 0 and size_overlap == 0)
    size_cov = safe_div(size_overlap, len(q_sizes))

    q_female, q_male, q_child, q_has_color, q_has_size, q_has_digit = detect_flags(qset, qnorm)

    item_female = int(("kadin" in gender_norm) or ("kadın" in gender_norm) or ("bayan" in gender_norm))
    item_male = int(("erkek" in gender_norm) or ("bay" in gender_norm))
    item_child = int(
        ("cocuk" in gender_norm)
        or ("çocuk" in gender_norm)
        or ("bebek" in gender_norm)
        or ("cocuk" in age_group_norm)
        or ("çocuk" in age_group_norm)
        or ("bebek" in age_group_norm)
    )

    gender_match = int((q_female and item_female) or (q_male and item_male))
    gender_mismatch = int((q_female and item_male) or (q_male and item_female))

    age_match = int(q_child and item_child)
    age_mismatch = int(q_child and not item_child and (item_female or item_male))

    brand_in_query_pair = int(bool(brand_norm) and contains_phrase(qnorm, brand_norm))
    query_has_known_brand = int(bool(qbrand_detected))
    query_brand_match = int(bool(qbrand_detected) and qbrand_detected == brand_norm)
    query_brand_mismatch = int(bool(qbrand_detected) and qbrand_detected != brand_norm)

    exact_query_in_title = int(bool(qnorm) and contains_phrase(title_norm, qnorm))
    exact_query_in_category = int(bool(qnorm) and contains_phrase(category_norm, qnorm))

    if q_len:
        all_query_in_title = int(qset.issubset(tset))
        all_query_in_title_cat = int(qset.issubset(tset | cset))
    else:
        all_query_in_title = 0
        all_query_in_title_cat = 0

    q_tokens = token_tuple_cached(qnorm)
    first_token_in_title = int(len(q_tokens) > 0 and q_tokens[0] in tset)
    last_token_in_title = int(len(q_tokens) > 0 and q_tokens[-1] in tset)

    item_train_pos_log = np.log1p(float(item_train_pos_count))
    brand_train_pos_log = np.log1p(float(brand_train_pos_count))
    root_train_pos_log = np.log1p(float(root_train_pos_count))

    return (
        q_len,
        len(qnorm),
        t_len,
        c_len,
        b_len,
        a_len,
        title_overlap,
        cat_overlap,
        brand_overlap,
        attr_overlap,
        total_overlap,
        title_cov,
        cat_cov,
        brand_cov,
        attr_cov,
        total_cov,
        title_jacc,
        cat_jacc,
        brand_jacc,
        attr_jacc,
        total_jacc,
        weighted_overlap,
        len(q_digits),
        digit_overlap,
        digit_cov,
        digit_mismatch,
        len(q_colors),
        color_overlap,
        color_cov,
        color_mismatch,
        len(q_sizes),
        size_overlap,
        size_cov,
        size_mismatch,
        q_female,
        q_male,
        q_child,
        q_has_color,
        q_has_size,
        q_has_digit,
        item_female,
        item_male,
        item_child,
        gender_match,
        gender_mismatch,
        age_match,
        age_mismatch,
        brand_in_query_pair,
        query_has_known_brand,
        query_brand_match,
        query_brand_mismatch,
        exact_query_in_title,
        exact_query_in_category,
        all_query_in_title,
        all_query_in_title_cat,
        first_token_in_title,
        last_token_in_title,
        int(brand_norm == ""),
        item_train_pos_log,
        brand_train_pos_log,
        root_train_pos_log,
    )

PAIR_FEATURE_COLUMNS = [
    "q_token_count",
    "q_char_len",
    "title_token_count",
    "category_token_count",
    "brand_token_count",
    "attr_token_count",
    "title_overlap",
    "category_overlap",
    "brand_overlap",
    "attr_overlap",
    "total_overlap",
    "title_cov",
    "category_cov",
    "brand_cov",
    "attr_cov",
    "total_cov",
    "title_jacc",
    "category_jacc",
    "brand_jacc",
    "attr_jacc",
    "total_jacc",
    "weighted_overlap",
    "query_digit_count",
    "digit_overlap",
    "digit_cov",
    "digit_mismatch",
    "query_color_count",
    "color_overlap",
    "color_cov",
    "color_mismatch",
    "query_size_count",
    "size_overlap",
    "size_cov",
    "size_mismatch",
    "query_female",
    "query_male",
    "query_child",
    "query_has_color",
    "query_has_size",
    "query_has_digit",
    "item_female",
    "item_male",
    "item_child",
    "gender_match",
    "gender_mismatch",
    "age_match",
    "age_mismatch",
    "brand_in_query_pair",
    "query_has_known_brand",
    "query_brand_match",
    "query_brand_mismatch",
    "exact_query_in_title",
    "exact_query_in_category",
    "all_query_in_title",
    "all_query_in_title_cat",
    "first_token_in_title",
    "last_token_in_title",
    "item_brand_empty",
    "item_train_pos_log",
    "brand_train_pos_log",
    "root_train_pos_log",
]

RANK_BASE_COLS = [
    "weighted_overlap",
    "total_cov",
    "title_cov",
    "brand_cov",
    "category_cov",
    "total_jacc",
    "digit_cov",
    "color_cov",
    "size_cov",
    "item_train_pos_log",
    "brand_train_pos_log",
    "root_train_pos_log",
]

def downcast_numeric(df):
    for c in df.columns:
        if c in {"id", "term_id", "item_id", "candidate_source", "root_key", "brand_key", "category_key"}:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            mn = df[c].min()
            mx = df[c].max()
            if mn >= 0 and mx <= 255:
                df[c] = df[c].astype("uint8")
            elif mn >= -32768 and mx <= 32767:
                df[c] = df[c].astype("int16")
            else:
                df[c] = df[c].astype("int32")
    return df

def add_group_features(feat):
    print("adding group features...")

    feat["candidate_count"] = feat.groupby("term_id")["item_id"].transform("size").astype("int32")

    feat["brand_key"] = feat["brand_key"].fillna("")
    feat["root_key"] = feat["root_key"].fillna("")
    feat["category_key"] = feat["category_key"].fillna("")

    feat["brand_count_in_term"] = feat.groupby(["term_id", "brand_key"])["item_id"].transform("size").astype("int32")
    feat["root_count_in_term"] = feat.groupby(["term_id", "root_key"])["item_id"].transform("size").astype("int32")
    feat["category_count_in_term"] = feat.groupby(["term_id", "category_key"])["item_id"].transform("size").astype("int32")

    feat["brand_count_ratio"] = (feat["brand_count_in_term"] / feat["candidate_count"]).astype("float32")
    feat["root_count_ratio"] = (feat["root_count_in_term"] / feat["candidate_count"]).astype("float32")
    feat["category_count_ratio"] = (feat["category_count_in_term"] / feat["candidate_count"]).astype("float32")

    for col in RANK_BASE_COLS:
        rank_col = f"{col}_rank"
        pct_col = f"{col}_pct_rank"
        max_col = f"{col}_term_max"
        delta_col = f"{col}_delta_top"

        feat[rank_col] = feat.groupby("term_id")[col].rank(method="first", ascending=False).astype("float32")
        feat[pct_col] = (feat[rank_col] / feat["candidate_count"]).astype("float32")
        feat[max_col] = feat.groupby("term_id")[col].transform("max").astype("float32")
        feat[delta_col] = (feat[max_col] - feat[col]).astype("float32")

    return feat

def build_one(candidates, mode, terms_feat, items_feat):
    total = len(candidates)
    parts = []

    print(f"\nBUILD {mode}: rows={total:,}")

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk = candidates.iloc[start:end].copy()

        chunk["term_id"] = chunk["term_id"].astype(str)
        chunk["item_id"] = chunk["item_id"].astype(str)

        chunk = chunk.merge(terms_feat, on="term_id", how="left", validate="many_to_one")
        chunk = chunk.merge(items_feat, on="item_id", how="left", validate="many_to_one")

        fill_cols = [
            "query_norm", "query_brand_detected",
            "title_norm", "category_norm", "brand_norm",
            "gender_norm", "age_group_norm", "attr_norm",
            "root_key", "brand_key", "category_key",
        ]

        for c in fill_cols:
            if c not in chunk.columns:
                chunk[c] = ""
            chunk[c] = chunk[c].fillna("").astype(str)

        for c in ["item_train_pos_count", "brand_train_pos_count", "root_train_pos_count"]:
            if c not in chunk.columns:
                chunk[c] = 0
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").fillna(0)

        rows = [
            compute_pair_features(
                qnorm,
                qbrand,
                title,
                cat,
                brand,
                gender,
                age,
                attr,
                item_pos,
                brand_pos,
                root_pos,
            )
            for qnorm, qbrand, title, cat, brand, gender, age, attr, item_pos, brand_pos, root_pos
            in zip(
                chunk["query_norm"],
                chunk["query_brand_detected"],
                chunk["title_norm"],
                chunk["category_norm"],
                chunk["brand_norm"],
                chunk["gender_norm"],
                chunk["age_group_norm"],
                chunk["attr_norm"],
                chunk["item_train_pos_count"],
                chunk["brand_train_pos_count"],
                chunk["root_train_pos_count"],
            )
        ]

        f = pd.DataFrame(rows, columns=PAIR_FEATURE_COLUMNS)

        keep = []
        for c in ["id", "term_id", "item_id", "label", "fold", "candidate_source"]:
            if c in chunk.columns:
                keep.append(c)

        meta = chunk[keep + ["root_key", "brand_key", "category_key"]].reset_index(drop=True)
        out = pd.concat([meta, f], axis=1)

        parts.append(out)

        print(f"{mode}: {end:,}/{total:,}")

    feat = pd.concat(parts, ignore_index=True)
    feat = add_group_features(feat)

    temp_cols = ["root_key", "brand_key", "category_key"]
    feat = feat.drop(columns=temp_cols)

    feat = downcast_numeric(feat)

    return feat

def main():
    t0 = time.time()

    print("loading raw tables...")

    train_pairs = pd.read_csv(TRAIN_PAIRS)
    train_pairs["term_id"] = train_pairs["term_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)
    if "label" in train_pairs.columns:
        train_pairs = train_pairs[train_pairs["label"].astype(int) == 1].copy()

    item_pos = train_pairs["item_id"].value_counts().rename("item_train_pos_count").reset_index()
    item_pos.columns = ["item_id", "item_train_pos_count"]
    item_pos["item_id"] = item_pos["item_id"].astype(str)

    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    terms["query_norm"] = terms["query"].map(norm)

    item_cols = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    items = pd.read_csv(ITEMS, usecols=lambda c: c in item_cols, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)

    for c in item_cols:
        if c not in items.columns:
            items[c] = ""

    print("normalizing item text...")
    items["title_norm"] = items["title"].map(norm)
    items["category_norm"] = items["category"].map(norm)
    items["brand_norm"] = items["brand"].map(norm)
    items["gender_norm"] = items["gender"].map(norm)
    items["age_group_norm"] = items["age_group"].map(norm)
    items["attr_norm"] = items["attributes"].fillna("").astype(str).str.slice(0, 600).map(norm)
    items["root_key"] = items["category"].map(root_category).map(norm)
    items["brand_key"] = items["brand_norm"]
    items["category_key"] = items["category_norm"].str.slice(0, 120)

    items = items.merge(item_pos, on="item_id", how="left")
    items["item_train_pos_count"] = items["item_train_pos_count"].fillna(0).astype("int32")

    brand_pos = (
        items.groupby("brand_norm")["item_train_pos_count"]
        .sum()
        .rename("brand_train_pos_count")
        .reset_index()
    )

    root_pos = (
        items.groupby("root_key")["item_train_pos_count"]
        .sum()
        .rename("root_train_pos_count")
        .reset_index()
    )

    items = items.merge(brand_pos, on="brand_norm", how="left")
    items = items.merge(root_pos, on="root_key", how="left")

    items["brand_train_pos_count"] = items["brand_train_pos_count"].fillna(0).astype("int32")
    items["root_train_pos_count"] = items["root_train_pos_count"].fillna(0).astype("int32")

    print("building brand detector...")
    brand_first_map = build_brand_detector(items)

    print("detecting query brands...")
    terms["query_brand_detected"] = terms["query_norm"].map(lambda x: detect_query_brand(x, brand_first_map))

    terms_feat = terms[["term_id", "query_norm", "query_brand_detected"]].copy()

    items_feat = items[
        [
            "item_id",
            "title_norm",
            "category_norm",
            "brand_norm",
            "gender_norm",
            "age_group_norm",
            "attr_norm",
            "root_key",
            "brand_key",
            "category_key",
            "item_train_pos_count",
            "brand_train_pos_count",
            "root_train_pos_count",
        ]
    ].copy()

    print("loading candidates...")
    train_cand = pd.read_parquet(TRAIN_CAND)
    train_cand["term_id"] = train_cand["term_id"].astype(str)
    train_cand["item_id"] = train_cand["item_id"].astype(str)

    test_cand = pd.read_csv(SUB_PAIRS)
    test_cand["id"] = test_cand["id"].astype(str)
    test_cand["term_id"] = test_cand["term_id"].astype(str)
    test_cand["item_id"] = test_cand["item_id"].astype(str)

    train_feat = build_one(train_cand, "train", terms_feat, items_feat)
    test_feat = build_one(test_cand, "test", terms_feat, items_feat)

    non_features = {"id", "term_id", "item_id", "label", "fold", "candidate_source"}
    feature_cols = [c for c in train_feat.columns if c not in non_features]

    missing_in_test = [c for c in feature_cols if c not in test_feat.columns]
    if missing_in_test:
        raise RuntimeError(f"Missing test feature columns: {missing_in_test}")

    # Ensure same feature column order exists in both.
    train_out_cols = [c for c in ["term_id", "item_id", "label", "fold", "candidate_source"] if c in train_feat.columns] + feature_cols
    test_out_cols = [c for c in ["id", "term_id", "item_id"] if c in test_feat.columns] + feature_cols

    train_feat = train_feat[train_out_cols]
    test_feat = test_feat[test_out_cols]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    print("saving train features...")
    train_feat.to_parquet(OUT_TRAIN, index=False)

    print("saving test features...")
    test_feat.to_parquet(OUT_TEST, index=False)

    FEATURE_LIST.write_text("\n".join(feature_cols), encoding="utf-8")

    report = pd.DataFrame([
        {
            "table": "train",
            "rows": len(train_feat),
            "cols": train_feat.shape[1],
            "feature_cols": len(feature_cols),
            "positive_rows": int(train_feat["label"].sum()),
            "pos_ratio": float(train_feat["label"].mean()),
            "unique_terms": int(train_feat["term_id"].nunique()),
            "unique_items": int(train_feat["item_id"].nunique()),
        },
        {
            "table": "test",
            "rows": len(test_feat),
            "cols": test_feat.shape[1],
            "feature_cols": len(feature_cols),
            "positive_rows": "",
            "pos_ratio": "",
            "unique_terms": int(test_feat["term_id"].nunique()),
            "unique_items": int(test_feat["item_id"].nunique()),
        },
    ])

    report["runtime_min"] = round((time.time() - t0) / 60, 3)
    report.to_csv(REPORT, index=False)

    print("\nREPORT")
    print(report.to_string(index=False))

    print("feature cols:", len(feature_cols))
    print("saved:", OUT_TRAIN)
    print("saved:", OUT_TEST)
    print("saved:", FEATURE_LIST)
    print("report:", REPORT)

if __name__ == "__main__":
    main()
