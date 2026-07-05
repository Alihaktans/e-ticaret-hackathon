from pathlib import Path
import re
import unicodedata
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
ITEMS = ROOT / "data/raw/items.csv"
TERMS = ROOT / "data/raw/terms.csv"

V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"
V38 = ROOT / "data/processed/v38_lexical_pair_scores.parquet"
V39 = ROOT / "data/processed/v39_item_history_pair_scores.parquet"
V41 = ROOT / "data/processed/v41_learned_meta_prob.parquet"
V42 = ROOT / "data/processed/v42_query_graph_pair_scores.parquet"
V43 = ROOT / "data/processed/v43_pop_brand_category_meta_prob.parquet"
V43_PRIOR = ROOT / "data/processed/v43_pop_brand_category_pair_scores.parquet"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

SAVED_V43 = ROOT / "reports/manual_review/v43_pop_brand_category_saved_candidates.csv"
SAVED_V45 = ROOT / "reports/manual_review/v45_semguard_v43_saved_candidates.csv"

REPORT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_REVIEW = REPORT_DIR / "review_v46_consensus_semantic_swaps.csv"
OUT_SUMMARY = REPORT_DIR / "v46_consensus_semantic_candidate_summary.csv"
OUT_EVAL = REPORT_DIR / "v46_consensus_semantic_candidate_eval.csv"
OUT_SAVED = REPORT_DIR / "v46_consensus_semantic_saved_candidates.csv"

TARGET_VARIANTS = [
    "v43_v41_meta_same_quota_v43_meta_prob_impact_cap7000",
    "v43_v41_meta_same_quota_v43_meta_prob_impact_cap4200",
    "v43_v41_meta_same_quota_v43_meta_prob_balanced_cap4200",
    "v43_v41_meta_same_quota_v43_meta_prob_strict_cap2400",
    "v43_v41_meta_same_quota_v43_pop_graph_score_strict_cap2400",
    "v43_v42_qgraph_v41_balanced3600_v43_meta_prob_impact_cap2400",
    "v43_v42_qgraph_v41_balanced3600_v43_meta_prob_strict_cap2400",
    "v43_v42_qgraph_v41_balanced3600_v43_pop_graph_score_strict_cap2400",
]

BASELINE_NAMES = [
    "anchor_v33_qprob2000",
    "raw_v35_b5000",
    "v41_meta_same_quota",
    "v42_qgraph_v41_balanced3600",
    "v45_noveto_b2acd38f_cap2200",
    "v45_guarded_b2acd38f_cap600",
]

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
    "v26_sparse": ROOT / "reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv",
}

TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "bir", "adet", "set", "takim", "model", "uyumlu",
    "orjinal", "orijinal", "yeni", "renk", "boy", "numara", "beden",
    "cm", "mm", "lt", "kg", "gr", "ml", "x", "no", "olan", "veya",
    "plus", "pro", "max", "mini", "urun", "da", "de", "mi", "mı", "mu", "mü",
}

GENDER_MALE = {"erkek", "bay", "adam"}
GENDER_FEMALE = {"kadin", "bayan", "kiz"}
CHILD = {"cocuk", "bebek", "junior", "jr", "ps", "gs", "kids", "kid"}

FOOTWEAR_TYPES = {"ayakkabi", "sneaker", "bot", "terlik", "sandalet", "krampon"}
APPAREL_TYPES = {"kaban", "mont", "ceket", "pantolon", "etek", "elbise", "trenckot", "palto", "hirka", "kazak", "sweatshirt", "esofman"}
PRODUCT_INTENT_WORDS = FOOTWEAR_TYPES | APPAREL_TYPES | {
    "parfum", "ruj", "maskara", "kapatici", "serum", "krem", "boya",
    "saat", "canta", "telefon", "tablet", "monitor", "laptop", "klima",
    "lastik", "termos", "tencere", "kase", "tepsi", "besik", "bebek",
    "oyuncak", "kitap", "atlas", "sut", "kahve", "kapsul", "ısıtıcı", "isitici",
}

GENERIC_BAD_BRANDS = {
    "", "nan", "none", "null", "trendyol", "diğer", "diger", "markasiz", "markasız",
    "trend", "shop", "store", "home", "fashion", "aksesuar", "no brand"
}

GENERIC_QUERY_BRAND_TOKENS = {
    "new", "mini", "model", "takim", "set", "kadin", "erkek", "bayan", "cocuk", "bebek",
    "anne", "canta", "forma", "yas", "celik", "dijital", "runner", "natural", "elektrik",
    "uydu", "tak", "sis", "alez", "dugme", "boya", "parfum", "saat", "gold",
    "beyaz", "siyah", "yesil", "mavi", "kirmizi", "kahverengi", "buyuk", "kucuk",
    "orta", "spor", "ayakkabi", "sneaker", "bot", "terlik", "kaban", "mont", "ceket",
    "hirka", "kazak", "sweatshirt", "tencere", "kase", "tepsi", "kap",
    "dekoratif", "cizgi", "doktor", "polar", "polo", "sut", "kahve", "kapsul",
}

REAL_SINGLE_BRAND_WHITELIST = {
    "nike", "puma", "adidas", "skechers", "hummel", "apple", "samsung", "lenovo", "acer",
    "philips", "casio", "stanley", "dior", "versace", "gap", "ugg", "dgn", "dove",
    "oriflame", "garnier", "parex", "baymak", "korkmaz", "fakir", "arzum",
    "bosch", "siemens", "tefal", "motul", "komili", "ikea", "mango", "zara",
    "defacto", "koton", "mad", "bargello", "flormar", "maybelline", "nivea", "vichy",
    "nutraxin", "huawei", "xiaomi", "lego", "rich", "kado",
}


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def toks(x):
    return [t for t in norm_text(x).split() if len(t) >= 2 and t not in STOP]


def tokset(x):
    return set(toks(x))


def number_tokens(x):
    return set(re.findall(r"\b\d+(?:[.,]\d+)?[a-z]?\b", norm_text(x)))


def modelish_tokens(x):
    out = set()
    for t in toks(x):
        if re.search(r"\d", t):
            out.add(t)
        elif len(t) >= 7:
            out.add(t)
    return out


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


def detect_col(cols, exacts=(), contains=()):
    lower = {c.lower(): c for c in cols}
    for e in exacts:
        if e.lower() in lower:
            return lower[e.lower()]
    for pat in contains:
        for c in cols:
            if pat.lower() in c.lower():
                return c
    return None


def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def read_saved_table(path):
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    if "variant" in d.columns and "file" in d.columns:
        d["variant"] = d["variant"].astype(str)
        d["file"] = d["file"].astype(str)
    return d


def find_variant_file(variant, tables):
    for d in tables:
        if len(d) == 0 or "variant" not in d.columns or "file" not in d.columns:
            continue
        hit = d[d["variant"].eq(variant)]
        if len(hit):
            p = Path(str(hit.iloc[0]["file"]))
            if p.exists():
                return p
    return None


def load_items_terms():
    items = pd.read_csv(ITEMS)
    cols = list(items.columns)
    item_col = detect_col(cols, exacts=("item_id", "product_id", "id"), contains=("item_id",))
    title_col = detect_col(cols, exacts=("title", "name", "product_name", "item_name", "urun_adi", "ürün_adı"), contains=("title", "name", "urun", "ürün"))
    brand_col = detect_col(cols, exacts=("brand", "marka"), contains=("brand", "marka"))
    cat_col = detect_col(cols, exacts=("category", "kategori", "category_name", "leaf_category", "cat"), contains=("category", "kategori", "cat"))

    if item_col is None:
        raise RuntimeError("item_id column not found")

    item_meta = pd.DataFrame({"item_id": items[item_col].astype(str)})
    item_meta["title"] = items[title_col].astype(str) if title_col else ""
    item_meta["brand"] = items[brand_col].astype(str) if brand_col else ""
    item_meta["category"] = items[cat_col].astype(str) if cat_col else ""
    item_meta["brand_norm"] = item_meta["brand"].map(norm_text)
    item_meta["title_norm"] = item_meta["title"].map(norm_text)
    item_meta["category_norm"] = item_meta["category"].map(norm_text)

    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    if "query" not in terms.columns:
        cand = [c for c in terms.columns if c != "term_id" and terms[c].dtype == "object"]
        if not cand:
            raise RuntimeError("query column not found")
        terms = terms.rename(columns={cand[0]: "query"})
    terms["query_norm"] = terms["query"].map(norm_text)

    brand_counts = item_meta["brand_norm"].value_counts()
    brand_list = []
    for b, cnt in brand_counts.items():
        if b in GENERIC_BAD_BRANDS or len(b) < 3:
            continue
        if b in {"home", "baby", "kids", "sport", "moda", "trend", "classic", "premium"}:
            continue
        brand_list.append((b, int(cnt), len(b)))
    brand_list.sort(key=lambda x: (x[2], x[1]), reverse=True)

    return item_meta, terms[["term_id", "query", "query_norm"]], brand_list


def find_query_brands(query_norm, brand_list, max_hits=4):
    hits = []
    q = f" {query_norm} "
    q_tokens = set(query_norm.split())

    for b, cnt, ln in brand_list:
        if " " not in b and "-" not in b:
            continue
        if b in GENERIC_BAD_BRANDS:
            continue
        if f" {b} " in q or query_norm.startswith(b + " ") or query_norm.endswith(" " + b):
            hits.append(b)
            if len(hits) >= max_hits:
                return hits

    for b, cnt, ln in brand_list:
        if " " in b or "-" in b:
            continue
        if b in GENERIC_BAD_BRANDS or b in GENERIC_QUERY_BRAND_TOKENS or b in PRODUCT_INTENT_WORDS:
            continue
        if b not in q_tokens:
            continue
        strong_single = b in REAL_SINGLE_BRAND_WHITELIST or (len(b) >= 5 and cnt >= 80) or (len(b) >= 7 and cnt >= 20)
        if not strong_single:
            continue
        hits.append(b)
        if len(hits) >= max_hits:
            break

    return hits


def brand_matches(brand_norm, title_norm, query_brands):
    if not query_brands:
        return True
    for b in query_brands:
        if brand_norm == b or b in brand_norm or b in title_norm:
            return True
    return False


def gender_flags(text):
    s = tokset(text)
    return {
        "male": bool(s & GENDER_MALE),
        "female": bool(s & GENDER_FEMALE),
        "child": bool(s & CHILD),
    }


def overlap_ratio(query, text):
    q = toks(query)
    if not q:
        return 0.0
    st = tokset(text)
    return len([t for t in q if t in st]) / len(q)


def coverage(query_tokens, text):
    if not query_tokens:
        return 1.0
    st = tokset(text)
    return len(set(query_tokens) & st) / len(set(query_tokens))


def semantic_veto(row):
    query = row["query_norm"]
    add_text = row["add_title_norm"] + " " + row["add_brand_norm"] + " " + row["add_category_norm"]
    drop_text = row["drop_title_norm"] + " " + row["drop_brand_norm"] + " " + row["drop_category_norm"]

    qset = tokset(query)
    addset = tokset(add_text)

    reasons = []

    qbrands = row["query_brands"]
    if qbrands:
        add_brand_ok = brand_matches(row["add_brand_norm"], row["add_title_norm"], qbrands)
        drop_brand_ok = brand_matches(row["drop_brand_norm"], row["drop_title_norm"], qbrands)
        if not add_brand_ok:
            reasons.append("brand_mismatch")
        if drop_brand_ok and not add_brand_ok:
            reasons.append("drop_brand_better")

    qg = gender_flags(query)
    ag = gender_flags(add_text)
    dg = gender_flags(drop_text)

    if qg["male"] and ag["female"] and not ag["male"]:
        reasons.append("male_query_add_female")
    if qg["female"] and ag["male"] and not ag["female"]:
        reasons.append("female_query_add_male")
    if qg["child"] and not ag["child"] and dg["child"]:
        reasons.append("child_query_drop_child_better")
    if not qg["child"] and ag["child"] and (qset & PRODUCT_INTENT_WORDS):
        reasons.append("adult_query_add_child")

    if "erkek" in qset and ("kadin" in addset or "kiz" in addset):
        reasons.append("male_query_add_female")
    if ("kadin" in qset or "bayan" in qset) and ("erkek" in addset or "jr" in addset or "cocuk" in addset):
        reasons.append("female_query_add_male_or_child")

    qnums = number_tokens(query)
    if qnums:
        add_nums = number_tokens(add_text)
        drop_nums = number_tokens(drop_text)
        if len(qnums & add_nums) < len(qnums):
            reasons.append("number_mismatch")
        if len(qnums & drop_nums) > len(qnums & add_nums):
            reasons.append("drop_number_better")

    qmodel = modelish_tokens(query)
    if qmodel:
        add_model = modelish_tokens(add_text)
        drop_model = modelish_tokens(drop_text)
        add_cov = len(qmodel & add_model) / len(qmodel)
        drop_cov = len(qmodel & drop_model) / len(qmodel)
        if add_cov < 0.5 and drop_cov >= add_cov:
            reasons.append("model_token_loss")

    add_title_ov = row["add_title_overlap"]
    drop_title_ov = row["drop_title_overlap"]
    add_cat_ov = row["add_category_overlap"]

    if drop_title_ov >= add_title_ov + 0.20 and drop_title_ov >= 0.25:
        reasons.append("drop_title_better")
    if add_title_ov <= 0.10 and add_cat_ov >= 0.30 and not qbrands:
        reasons.append("category_only_add")
    if add_title_ov <= 0.08 and row["v43_query_pop_support_add"] <= 0.20:
        reasons.append("weak_title_and_pop_support")

    if "apple" in qset and "saat" in qset and not ("apple" in addset or "watch" in addset):
        reasons.append("apple_watch_brand_mismatch")

    tire_size_tokens = {t for t in qset if re.match(r"r\d{2}$", t) or re.match(r"\d{3}$", t) or re.match(r"\d{2}$", t)}
    if "lastik" in qset and tire_size_tokens:
        missing = [t for t in tire_size_tokens if t not in addset]
        if missing:
            reasons.append("tire_size_mismatch")

    if "parfum" in qset:
        if "erkek" in qset and "kadin" in addset and "erkek" not in addset:
            reasons.append("male_perfume_add_female")
        if "kadin" in qset and "erkek" in addset and "kadin" not in addset:
            reasons.append("female_perfume_add_male")

    if "4k" in qset and ("monitor" in qset or "monitör" in qset):
        if "4k" not in addset and "uhd" not in addset:
            reasons.append("4k_missing")

    return sorted(set(reasons))


def semantic_keep(row):
    query = row["query_norm"]
    qbrands = row["query_brands"]
    brand_ok = brand_matches(row["add_brand_norm"], row["add_title_norm"], qbrands) if qbrands else True
    if qbrands and not brand_ok:
        return False

    title_gain = row["add_title_overlap"] - row["drop_title_overlap"]
    title_abs_ok = row["add_title_overlap"] >= 0.20
    title_gain_ok = title_gain >= 0.10
    support_ok = row["v43_query_pop_support_add"] >= 0.22
    prob_ok = row["v43_gain"] >= 0.45
    number_ok = row["add_number_cov"] >= row["drop_number_cov"]

    if title_gain_ok and support_ok and number_ok:
        return True
    if title_abs_ok and prob_ok and number_ok:
        return True
    if qbrands and brand_ok and number_ok and row["add_title_overlap"] >= 0.12:
        return True

    q_len = len(toks(query))
    if q_len <= 2 and row["add_category_overlap"] >= 0.50 and row["add_title_overlap"] >= 0.10 and row["v43_gain"] >= 0.70:
        return True

    if row["v43_query_pop_support_add"] >= 0.38 and row["v43_gain"] >= 0.75 and number_ok and row["add_title_overlap"] >= 0.10:
        return True

    return False


def safe_parquet(path, sample, cols):
    out = pd.DataFrame({"id": sample["id"].astype(str)})
    if not path.exists():
        for c in cols:
            out[c] = 0.0
        return out
    try:
        d = pd.read_parquet(path)
    except Exception as e:
        print("could not read", path, repr(e))
        for c in cols:
            out[c] = 0.0
        return out

    d["id"] = d["id"].astype(str)
    keep = ["id"] + [c for c in cols if c in d.columns]
    d = d[keep].copy()
    if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        out = d.copy()
    else:
        out = out.merge(d, on="id", how="left", validate="one_to_one")
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.float32)
    return out[["id"] + cols]


def build_feature_frame(sample, pairs):
    df = pairs[["id", "term_id", "item_id"]].copy()

    specs = [
        (V33, ["v33_ft_score"]),
        (V34, ["v34_ce_score"]),
        (V38, ["v38_lex_score", "v38_word_overlap_title", "v38_query_word_only_category_ratio"]),
        (V39, ["v39_hist_score", "v39_hist_has_item", "v39_hist_weighted_cov"]),
        (V41, ["v41_meta_prob"]),
        (V42, ["v42_graph_score", "v42_graph_norm_score", "v42_graph_max_sim"]),
        (V43, ["v43_meta_prob"]),
        (V43_PRIOR, ["v43_pop_prior_score", "v43_query_pop_support", "v43_popularity_trap"]),
    ]

    for path, cols in specs:
        x = safe_parquet(path, sample, cols)
        for c in cols:
            df[c] = x[c].astype(np.float32).to_numpy()

    # Fill robust aliases.
    for c in [
        "v33_ft_score", "v34_ce_score", "v38_lex_score", "v39_hist_score",
        "v41_meta_prob", "v42_graph_score", "v43_meta_prob",
        "v43_query_pop_support", "v43_popularity_trap"
    ]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

    return df


def build_swaps_from_anchor(sample, anchor, cand, pairs):
    add_mask = (anchor == 0) & (cand == 1)
    drop_mask = (anchor == 1) & (cand == 0)

    add = pairs.loc[add_mask, ["id", "term_id", "item_id"]].copy()
    drop = pairs.loc[drop_mask, ["id", "term_id", "item_id"]].copy()

    add = add.rename(columns={"id": "id_add", "item_id": "item_id_add"})
    drop = drop.rename(columns={"id": "id_drop", "item_id": "item_id_drop"})

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    return add.merge(drop, on=["term_id", "pair_rank"], how="inner")


def add_scores(sw, df, sample):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    add_idx = sw["id_add"].astype(str).map(id_to_idx)
    drop_idx = sw["id_drop"].astype(str).map(id_to_idx)
    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("id mapping failed")
    ai = add_idx.astype(int).to_numpy()
    di = drop_idx.astype(int).to_numpy()

    score_cols = [
        "v33_ft_score", "v34_ce_score", "v38_lex_score", "v39_hist_score",
        "v39_hist_has_item", "v39_hist_weighted_cov", "v41_meta_prob",
        "v42_graph_score", "v42_graph_norm_score", "v42_graph_max_sim",
        "v43_meta_prob", "v43_pop_prior_score", "v43_query_pop_support", "v43_popularity_trap",
        "v38_word_overlap_title", "v38_query_word_only_category_ratio",
    ]

    for c in score_cols:
        arr = df[c].astype(np.float32).to_numpy() if c in df.columns else np.zeros(len(df), dtype=np.float32)
        sw[c + "_add"] = arr[ai]
        sw[c + "_drop"] = arr[di]

    for c in ["v33_ft_score", "v34_ce_score", "v38_lex_score", "v39_hist_score", "v41_meta_prob", "v42_graph_score", "v43_meta_prob"]:
        sw[c.replace("_score", "").replace("_prob", "") + "_gain"] = sw[c + "_add"] - sw[c + "_drop"]

    # Friendly canonical gains.
    sw["v33_gain"] = sw["v33_ft_score_add"] - sw["v33_ft_score_drop"]
    sw["v34_gain"] = sw["v34_ce_score_add"] - sw["v34_ce_score_drop"]
    sw["v38_gain"] = sw["v38_lex_score_add"] - sw["v38_lex_score_drop"]
    sw["v39_gain"] = sw["v39_hist_score_add"] - sw["v39_hist_score_drop"]
    sw["v41_gain"] = sw["v41_meta_prob_add"] - sw["v41_meta_prob_drop"]
    sw["v42_gain"] = sw["v42_graph_score_add"] - sw["v42_graph_score_drop"]
    sw["v43_gain"] = sw["v43_meta_prob_add"] - sw["v43_meta_prob_drop"]

    return sw


def enrich_semantics(sw, item_meta, terms, brand_list):
    add = item_meta.rename(columns={
        "item_id": "item_id_add",
        "title": "add_title",
        "brand": "add_brand",
        "category": "add_category",
        "title_norm": "add_title_norm",
        "brand_norm": "add_brand_norm",
        "category_norm": "add_category_norm",
    })
    drop = item_meta.rename(columns={
        "item_id": "item_id_drop",
        "title": "drop_title",
        "brand": "drop_brand",
        "category": "drop_category",
        "title_norm": "drop_title_norm",
        "brand_norm": "drop_brand_norm",
        "category_norm": "drop_category_norm",
    })

    out = sw.merge(terms, on="term_id", how="left")
    out = out.merge(add[["item_id_add", "add_title", "add_brand", "add_category", "add_title_norm", "add_brand_norm", "add_category_norm"]], on="item_id_add", how="left")
    out = out.merge(drop[["item_id_drop", "drop_title", "drop_brand", "drop_category", "drop_title_norm", "drop_brand_norm", "drop_category_norm"]], on="item_id_drop", how="left")

    for c in [
        "query", "query_norm", "add_title", "add_brand", "add_category",
        "drop_title", "drop_brand", "drop_category",
        "add_title_norm", "add_brand_norm", "add_category_norm",
        "drop_title_norm", "drop_brand_norm", "drop_category_norm",
    ]:
        out[c] = out[c].fillna("").astype(str)

    out["query_brands"] = out["query_norm"].map(lambda q: find_query_brands(q, brand_list))
    out["query_brand_text"] = out["query_brands"].map(lambda xs: "|".join(xs))

    out["add_title_overlap"] = [overlap_ratio(q, t) for q, t in zip(out["query_norm"], out["add_title_norm"])]
    out["drop_title_overlap"] = [overlap_ratio(q, t) for q, t in zip(out["query_norm"], out["drop_title_norm"])]
    out["add_category_overlap"] = [overlap_ratio(q, t) for q, t in zip(out["query_norm"], out["add_category_norm"])]
    out["drop_category_overlap"] = [overlap_ratio(q, t) for q, t in zip(out["query_norm"], out["drop_category_norm"])]

    out["add_number_cov"] = [
        coverage(number_tokens(q), t + " " + b + " " + c)
        for q, t, b, c in zip(out["query_norm"], out["add_title_norm"], out["add_brand_norm"], out["add_category_norm"])
    ]
    out["drop_number_cov"] = [
        coverage(number_tokens(q), t + " " + b + " " + c)
        for q, t, b, c in zip(out["query_norm"], out["drop_title_norm"], out["drop_brand_norm"], out["drop_category_norm"])
    ]

    out["v46_veto_reasons"] = out.apply(semantic_veto, axis=1)
    out["v46_veto_reason_text"] = out["v46_veto_reasons"].map(lambda xs: "|".join(xs))
    out["v46_veto"] = out["v46_veto_reasons"].map(lambda xs: len(xs) > 0).astype(int)
    out["v46_semantic_keep"] = out.apply(semantic_keep, axis=1).astype(int)

    signal_cols = ["v33_gain", "v34_gain", "v38_gain", "v41_gain", "v42_gain", "v43_gain"]
    out["v46_signal_win_count"] = sum((out[c] > 0).astype(int) for c in signal_cols)
    out["v46_signal_strong_win_count"] = (
        (out["v33_gain"] > 0.02).astype(int) +
        (out["v34_gain"] > 0.02).astype(int) +
        (out["v38_gain"] > 0.02).astype(int) +
        (out["v41_gain"] > 0.03).astype(int) +
        (out["v42_gain"] > 0.00).astype(int) +
        (out["v43_gain"] > 0.25).astype(int)
    )

    out["title_gain"] = out["add_title_overlap"] - out["drop_title_overlap"]
    out["category_gain"] = out["add_category_overlap"] - out["drop_category_overlap"]

    out["v46_consensus_score"] = (
        0.40 * out["v43_gain"].astype(float)
        + 0.16 * out["v34_gain"].astype(float)
        + 0.12 * out["v33_gain"].astype(float)
        + 0.10 * out["v38_gain"].astype(float)
        + 0.08 * out["v41_gain"].astype(float)
        + 0.04 * np.tanh(out["v42_gain"].astype(float))
        + 0.18 * out["v46_semantic_keep"].astype(float)
        - 0.35 * out["v46_veto"].astype(float)
        + 0.12 * out["title_gain"].astype(float)
        + 0.05 * out["category_gain"].astype(float)
        + 0.03 * out["v46_signal_win_count"].astype(float)
    )

    return out.sort_values(["v46_consensus_score", "v43_gain"], ascending=False).reset_index(drop=True)


def filter_swaps(sw, mode):
    no_veto = sw["v46_veto"].eq(0)
    not_title_bad = sw["drop_title_overlap"] <= sw["add_title_overlap"] + 0.18
    not_low_title = sw["add_title_overlap"] >= 0.08

    if mode == "strict":
        return (
            no_veto
            & sw["v46_semantic_keep"].eq(1)
            & (sw["v46_signal_win_count"] >= 4)
            & (sw["v46_signal_strong_win_count"] >= 2)
            & (sw["v43_gain"] >= 0.45)
            & (sw["v34_gain"] >= -0.03)
            & (sw["v38_gain"] >= -0.04)
            & not_title_bad
        )

    if mode == "balanced":
        return (
            no_veto
            & ((sw["v46_semantic_keep"].eq(1)) | (sw["v46_signal_win_count"] >= 5))
            & (sw["v46_signal_win_count"] >= 4)
            & (sw["v43_gain"] >= 0.30)
            & (sw["v34_gain"] >= -0.06)
            & (sw["v38_gain"] >= -0.07)
            & not_title_bad
            & not_low_title
        )

    if mode == "impact":
        return (
            no_veto
            & (sw["v46_signal_win_count"] >= 4)
            & (sw["v46_signal_strong_win_count"] >= 2)
            & (sw["v43_gain"] >= 0.22)
            & (sw["v34_gain"] >= -0.10)
            & (sw["v38_gain"] >= -0.10)
            & not_title_bad
            & not_low_title
            & (sw["v46_consensus_score"] >= 0.12)
        )

    if mode == "ultra":
        return (
            no_veto
            & sw["v46_semantic_keep"].eq(1)
            & (sw["v46_signal_win_count"] >= 5)
            & (sw["v43_gain"] >= 0.60)
            & (sw["v34_gain"] >= 0.00)
            & (sw["v38_gain"] >= -0.02)
            & (sw["add_title_overlap"] >= 0.15)
        )

    raise ValueError(mode)


def make_pred(sample, anchor, sw, cap):
    take = sw.sort_values(["v46_consensus_score", "v43_gain"], ascending=False).head(cap).copy()
    pred = anchor.copy()
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    add_idx = take["id_add"].astype(str).map(id_to_idx)
    drop_idx = take["id_drop"].astype(str).map(id_to_idx)
    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("swap id map failed")
    pred[add_idx.astype(int).to_numpy()] = 1
    pred[drop_idx.astype(int).to_numpy()] = 0
    return pred, take


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1], zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_variants(variants, sample):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    rows = []
    for label_set, path in LABEL_FILES.items():
        if not path.exists():
            continue
        lab = pd.read_csv(path)
        if "id" not in lab.columns or "assistant_label" not in lab.columns:
            continue
        lab["id"] = lab["id"].astype(str)
        lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
        if len(lab) < 30:
            continue
        lab["assistant_label"] = lab["assistant_label"].astype(int)

        subsets = {"all": lab}
        if "needs_recheck" in lab.columns:
            nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
            subsets["clean"] = lab[nr == 0].copy()
        if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
            conf = lab["assistant_confidence"].astype(str).str.lower()
            nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
            subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()
            subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()
        if label_set in ["v20_active", "v21_active", "v26_sparse"] and "review_bucket" in lab.columns:
            for b, g in lab.groupby("review_bucket"):
                if len(g) >= 20 and g["assistant_label"].nunique() >= 2:
                    subsets[f"bucket_{b}"] = g.copy()

        for subset_name, part in subsets.items():
            if len(part) < 30 or part["assistant_label"].nunique() < 2:
                continue
            idx = part["id"].map(id_to_idx)
            ok = idx.notna()
            if ok.sum() < 30:
                continue
            idx = idx[ok].astype(int).to_numpy()
            y = part.loc[ok, "assistant_label"].astype(int).to_numpy()
            for name, pred in variants.items():
                row = {"label_set": label_set, "subset": subset_name, "eval_key": f"{label_set}_{subset_name}", "variant": name, "n": int(len(y))}
                row.update(metrics(y, pred[idx]))
                rows.append(row)
    return pd.DataFrame(rows)


def weighted_score(eval_df):
    weights = {
        "random_clean_v2_all": 0.26,
        "random_clean_v2_clean": 0.26,
        "manual_v1_clean": 0.25,
        "manual_v1_high_clean": 0.12,
        "manual_v1_high_medium_clean": 0.10,
        "review_v15_vs_v13_clean": 0.13,
        "review_v15_vs_v13_high_medium_clean": 0.11,
        "review_v13_vs_v5_clean": 0.05,
        "v20_active_clean": 0.04,
        "v21_active_clean": 0.09,
        "v21_active_high_medium_clean": 0.07,
        "v26_sparse_clean": 0.07,
        "v26_sparse_high_medium_clean": 0.05,
    }
    rows = []
    if len(eval_df) == 0:
        return pd.DataFrame()
    for name, g in eval_df.groupby("variant"):
        val = 0.0
        wsum = 0.0
        used = []
        for _, r in g.iterrows():
            w = weights.get(r["eval_key"], 0.0)
            if w:
                val += w * float(r["macro_f1"])
                wsum += w
                used.append(float(r["macro_f1"]))
        if wsum == 0:
            continue
        rows.append({
            "variant": name,
            "weighted_macro": float(val / wsum),
            "used_min_macro": float(np.min(used)) if used else np.nan,
            "eval_count": int(len(g)),
            "mean_precision": float(g["precision"].mean()),
            "mean_recall": float(g["recall"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print("loading sample/pairs...")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not sample["id"].reset_index(drop=True).equals(pairs["id"].reset_index(drop=True)):
        raise RuntimeError("sample/pairs order mismatch")

    anchor_path = first_existing(ANCHOR_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor not found")
    anchor = load_pred(anchor_path, sample)

    saved_v43 = read_saved_table(SAVED_V43)
    saved_v45 = read_saved_table(SAVED_V45)
    tables = [saved_v43, saved_v45]

    variants = {"anchor_v33_qprob2000": anchor.copy()}
    for name in BASELINE_NAMES:
        if name == "anchor_v33_qprob2000":
            continue
        p = find_variant_file(name, tables)
        if p is not None:
            variants[name] = load_pred(p, sample)
            print("baseline", name, "diff", int((variants[name] != anchor).sum()))

    print("loading features...")
    feat = build_feature_frame(sample, pairs)

    print("loading item/term metadata...")
    item_meta, terms, brand_list = load_items_terms()

    meta_rows = []
    review_parts = []

    caps = [400, 700, 1000, 1400, 1900, 2600, 3600]
    modes = ["ultra", "strict", "balanced", "impact"]

    for target in TARGET_VARIANTS:
        p = find_variant_file(target, tables)
        if p is None:
            print("missing target", target)
            continue

        cand = load_pred(p, sample)
        base_sw = build_swaps_from_anchor(sample, anchor, cand, pairs)
        if len(base_sw) == 0:
            continue

        print("target", target, "swaps", len(base_sw))
        sw = add_scores(base_sw, feat, sample)
        sw = enrich_semantics(sw, item_meta, terms, brand_list)
        sw["source_variant"] = target

        review_parts.append(sw.head(250))
        review_parts.append(sw[sw["v46_veto"].eq(1)].sort_values("v46_consensus_score", ascending=False).head(200))
        review_parts.append(sw[(sw["v46_veto"].eq(0)) & (sw["v46_semantic_keep"].eq(1))].head(250))

        for mode in modes:
            acc = sw[filter_swaps(sw, mode)].copy()
            if len(acc) == 0:
                continue
            acc = acc.sort_values(["v46_consensus_score", "v43_gain"], ascending=False)

            for cap in caps:
                used = min(cap, len(acc))
                pred, take = make_pred(sample, anchor, acc, used)
                name = f"v46_{mode}_{short_hash(target)}_cap{cap}"
                variants[name] = pred
                meta_rows.append({
                    "variant": name,
                    "source": "v46_consensus_semantic",
                    "source_variant": target,
                    "mode": mode,
                    "cap": cap,
                    "used_swaps": int(len(take)),
                    "source_swaps": int(len(sw)),
                    "accepted_pool": int(len(acc)),
                    "veto_rate_source": float(sw["v46_veto"].mean()),
                    "semantic_keep_rate_source": float(sw["v46_semantic_keep"].mean()),
                    "signal_win_mean": float(take["v46_signal_win_count"].mean()),
                    "strong_win_mean": float(take["v46_signal_strong_win_count"].mean()),
                    "consensus_score_mean": float(take["v46_consensus_score"].mean()),
                    "v43_gain_mean": float(take["v43_gain"].mean()),
                    "v34_gain_mean": float(take["v34_gain"].mean()),
                    "v38_gain_mean": float(take["v38_gain"].mean()),
                    "title_gain_mean": float(take["title_gain"].mean()),
                    "add_title_overlap_mean": float(take["add_title_overlap"].mean()),
                    "query_pop_support_mean": float(take["v43_query_pop_support_add"].mean()),
                })
                print(" candidate", name, "used", len(take), "accepted", len(acc), "diff", int((pred != anchor).sum()))

    if review_parts:
        rev = pd.concat(review_parts, ignore_index=True)
        rev = rev.drop_duplicates(["source_variant", "id_add", "id_drop"], keep="first")
        front = [
            "source_variant", "term_id", "pair_rank", "query", "query_brand_text",
            "id_add", "item_id_add", "add_title", "add_brand", "add_category",
            "id_drop", "item_id_drop", "drop_title", "drop_brand", "drop_category",
            "v46_veto", "v46_semantic_keep", "v46_veto_reason_text",
            "v46_consensus_score", "v46_signal_win_count", "v46_signal_strong_win_count",
            "v43_gain", "v34_gain", "v33_gain", "v38_gain", "v41_gain", "v42_gain",
            "add_title_overlap", "drop_title_overlap", "title_gain",
            "add_category_overlap", "drop_category_overlap", "v43_query_pop_support_add",
        ]
        rest = [c for c in rev.columns if c not in front]
        rev[front + rest].to_csv(OUT_REVIEW, index=False)

    print("evaluating...")
    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)
    wdf = weighted_score(eval_df)

    summary = pd.DataFrame(meta_rows)
    known = set(summary["variant"].astype(str)) if len(summary) else set()
    for name, pred in variants.items():
        if name not in known:
            summary = pd.concat([summary, pd.DataFrame([{
                "variant": name,
                "source": "baseline",
                "source_variant": "",
                "mode": "",
                "cap": "",
                "used_swaps": "",
                "source_swaps": "",
                "accepted_pool": "",
                "veto_rate_source": "",
                "semantic_keep_rate_source": "",
                "signal_win_mean": np.nan,
                "strong_win_mean": np.nan,
                "consensus_score_mean": np.nan,
                "v43_gain_mean": np.nan,
                "v34_gain_mean": np.nan,
                "v38_gain_mean": np.nan,
                "title_gain_mean": np.nan,
                "add_title_overlap_mean": np.nan,
                "query_pop_support_mean": np.nan,
            }])], ignore_index=True)

    aux = []
    for name, pred in variants.items():
        aux.append({
            "variant": name,
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != variants["raw_v35_b5000"]).sum()) if "raw_v35_b5000" in variants else -1,
            "diff_vs_v41_same": int((pred != variants["v41_meta_same_quota"]).sum()) if "v41_meta_same_quota" in variants else -1,
            "diff_vs_v42_qgraph": int((pred != variants["v42_qgraph_v41_balanced3600"]).sum()) if "v42_qgraph_v41_balanced3600" in variants else -1,
        })

    summary = summary.merge(pd.DataFrame(aux), on="variant", how="right")
    if len(wdf):
        summary = summary.merge(wdf, on="variant", how="left")

    diff = summary["diff_vs_anchor"].fillna(0).astype(float)
    move_bonus = np.minimum(0.018, np.log1p(diff) / np.log1p(25000) * 0.018)
    consensus_bonus = np.minimum(0.008, summary["consensus_score_mean"].fillna(0).astype(float) * 0.006)
    signal_bonus = np.minimum(0.004, summary["signal_win_mean"].fillna(0).astype(float) / 6 * 0.004)
    too_big_penalty = np.maximum(0, diff - 22000) / 400000.0

    summary["v46_decision_score"] = (
        summary["weighted_macro"].fillna(0)
        + move_bonus
        + consensus_bonus
        + signal_bonus
        - too_big_penalty
    )

    summary = summary.sort_values(["v46_decision_score", "weighted_macro"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    save_names = []
    top = summary[
        (summary["source"].eq("v46_consensus_semantic"))
        & (summary["diff_vs_anchor"] >= 800)
        & (summary["diff_vs_anchor"] <= 16000)
    ].head(45)
    for nm in top["variant"].tolist():
        if nm not in save_names:
            save_names.append(nm)

    for nm in ["v42_qgraph_v41_balanced3600", "raw_v35_b5000", "v41_meta_same_quota", "anchor_v33_qprob2000"]:
        if nm in variants and nm not in save_names:
            save_names.append(nm)

    saved_rows = []
    for i, nm in enumerate(save_names[:50], 1):
        pred = variants[nm]
        out = SUB_DIR / f"FINAL_CANDIDATE_v46_consensus_{i:03d}_{short_hash(nm)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)
        row = summary[summary["variant"] == nm].iloc[0].to_dict()
        row["file"] = str(out)
        saved_rows.append(row)
        print("saved", out, "<-", nm)

    pd.DataFrame(saved_rows).to_csv(OUT_SAVED, index=False)

    print("\nTOP V46")
    cols = [
        "variant", "v46_decision_score", "weighted_macro", "source", "source_variant", "mode", "cap",
        "used_swaps", "accepted_pool", "veto_rate_source", "semantic_keep_rate_source",
        "signal_win_mean", "strong_win_mean", "consensus_score_mean", "v43_gain_mean", "v34_gain_mean", "v38_gain_mean",
        "title_gain_mean", "add_title_overlap_mean", "query_pop_support_mean",
        "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v41_same", "diff_vs_v42_qgraph",
        "mean_precision", "mean_recall",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(100).to_string(index=False))

    print("\nSAVED")
    sdf = pd.DataFrame(saved_rows)
    if len(sdf):
        print(sdf[["variant", "file", "v46_decision_score", "weighted_macro", "diff_vs_anchor"]].to_string(index=False))

    print("\noutputs:")
    print(OUT_REVIEW)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_SAVED)


if __name__ == "__main__":
    main()
