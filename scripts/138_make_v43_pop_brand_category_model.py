from pathlib import Path
import re
import json
import math
import hashlib
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN = ROOT / "data/raw/training_pairs.csv"
ITEMS = ROOT / "data/raw/items.csv"
TERMS = ROOT / "data/raw/terms.csv"

V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"
V38 = ROOT / "data/processed/v38_lexical_pair_scores.parquet"
V39 = ROOT / "data/processed/v39_item_history_pair_scores.parquet"
V41 = ROOT / "data/processed/v41_learned_meta_prob.parquet"
V42 = ROOT / "data/processed/v42_query_graph_pair_scores.parquet"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

DIRECT_BASES = {
    "raw_v35_b5000": [
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv",
    ],
    "v36p2_balanced": [
        ROOT / "submissions/FINAL_MAIN_v36p2_balanced.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v36p2_balanced.csv",
    ],
    "v36p2_strict": [
        ROOT / "submissions/FINAL_MAIN_v36p2_strict.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v36p2_strict.csv",
    ],
}

SAVED_CSVS = [
    ROOT / "reports/manual_review/v38_lexical_saved_candidates.csv",
    ROOT / "reports/manual_review/v39_item_history_saved_candidates.csv",
    ROOT / "reports/manual_review/v40_boundary_stability_saved_candidates.csv",
    ROOT / "reports/manual_review/v41_learned_meta_saved_candidates.csv",
    ROOT / "reports/manual_review/v42_query_graph_saved_candidates.csv",
]

SAVED_VARIANTS = {
    "v38_precision_loose_2800": "v38_raw_v35_b5000_v38_precision_score_loose_cap2800",
    "v39_raw35_history_lex_veto_bad": "v39_raw_v35_b5000_v39_history_lex_score_history_veto_contradiction_and_bad",
    "v40_history_impact2200": "v40_raw_v35_b5000_v40_history_score_impact_cap2200",
    "v41_meta_same_quota": "v41_meta_same_quota",
    "v42_qgraph_v41_balanced3600": "v42_v41_meta_same_quota_v42_graph_only_score_balanced_cap3600",
}

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
    "v26_sparse": ROOT / "reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv",
}

PROC_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_PRIOR = PROC_DIR / "v43_pop_brand_category_pair_scores.parquet"
OUT_PROB = PROC_DIR / "v43_pop_brand_category_meta_prob.parquet"
OUT_INFO = REPORT_DIR / "v43_pop_brand_category_info.json"
OUT_MODEL = REPORT_DIR / "v43_pop_brand_category_model_info.csv"
OUT_OOF = REPORT_DIR / "v43_pop_brand_category_label_oof.csv"
OUT_SUMMARY = REPORT_DIR / "v43_pop_brand_category_candidate_summary.csv"
OUT_EVAL = REPORT_DIR / "v43_pop_brand_category_candidate_eval.csv"
OUT_SAVED = REPORT_DIR / "v43_pop_brand_category_saved_candidates.csv"
OUT_REVIEW = REPORT_DIR / "review_v43_pop_brand_category_swaps.csv"

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


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def toks(x):
    return [t for t in norm_text(x).split() if len(t) >= 2 and t not in STOP]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def find_saved_variant(variant_name):
    for csv_path in SAVED_CSVS:
        if not csv_path.exists():
            continue
        d = pd.read_csv(csv_path)
        if "variant" not in d.columns or "file" not in d.columns:
            continue
        hit = d[d["variant"].astype(str).eq(variant_name)]
        if len(hit):
            p = Path(str(hit.iloc[0]["file"]))
            if p.exists():
                return p
    return None


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def safe_parquet(path, sample, cols):
    out = pd.DataFrame({"id": sample["id"].astype(str)})
    if not path.exists():
        for c in cols:
            out[c] = 0.0
        return out
    try:
        d = pd.read_parquet(path)
    except Exception as e:
        print("could not read parquet:", path, repr(e))
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


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


def load_item_meta():
    print("loading items metadata...")
    items = pd.read_csv(ITEMS)
    cols = list(items.columns)

    item_col = detect_col(cols, exacts=("item_id", "product_id", "id"), contains=("item_id",))
    if item_col is None:
        raise RuntimeError(f"Could not detect item id column in items.csv columns={cols[:30]}")

    title_col = detect_col(
        cols,
        exacts=("title", "name", "product_name", "item_name", "urun_adi", "ürün_adı"),
        contains=("title", "name", "urun", "ürün"),
    )
    brand_col = detect_col(cols, exacts=("brand", "marka"), contains=("brand", "marka"))
    category_col = detect_col(
        cols,
        exacts=("category", "kategori", "category_name", "leaf_category", "cat"),
        contains=("category", "kategori", "cat"),
    )

    out = pd.DataFrame({"item_id": items[item_col].astype(str)})
    out["title_text"] = items[title_col].astype(str) if title_col else ""
    out["brand_raw"] = items[brand_col].astype(str) if brand_col else ""
    out["category_raw"] = items[category_col].astype(str) if category_col else ""

    out["brand_norm"] = out["brand_raw"].map(norm_text)
    out["category_norm"] = out["category_raw"].map(norm_text)
    out["title_norm"] = out["title_text"].map(norm_text)

    out["brand_norm"] = out["brand_norm"].replace({"nan": "", "none": "", "null": ""})
    out["category_norm"] = out["category_norm"].replace({"nan": "", "none": "", "null": ""})

    info = {
        "item_id_col": item_col,
        "title_col": title_col,
        "brand_col": brand_col,
        "category_col": category_col,
        "items": int(len(out)),
        "unique_brand": int(out["brand_norm"].nunique()),
        "unique_category": int(out["category_norm"].nunique()),
    }
    print("metadata columns:", info)
    return out, info


def load_terms():
    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    if "query" not in terms.columns:
        cand = [c for c in terms.columns if c != "term_id" and terms[c].dtype == "object"]
        if not cand:
            raise RuntimeError("terms.csv query column not found")
        terms = terms.rename(columns={cand[0]: "query"})
    terms["query_norm"] = terms["query"].map(norm_text)
    terms["query_tokens"] = terms["query_norm"].map(toks)
    return terms[["term_id", "query", "query_norm", "query_tokens"]]


def weighted_affinity(token_maps, total_maps, tokens, key):
    if not tokens or not key:
        return 0.0
    hit = 0.0
    denom = 0.0
    for t in tokens:
        total = total_maps.get(t, 0)
        if total <= 0:
            continue
        val = token_maps.get(t, {}).get(key, 0)
        # Shrunk token likelihood. log makes frequent brands/categories not dominate too much.
        hit += math.log1p(val)
        denom += math.log1p(total)
    return float(hit / denom) if denom > 0 else 0.0


def build_v43_priors(sample, pairs):
    if OUT_PRIOR.exists():
        print("using existing V43 prior:", OUT_PRIOR)
        out = pd.read_parquet(OUT_PRIOR)
        out["id"] = out["id"].astype(str)
        if not out["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            raise RuntimeError("V43 prior id order mismatch")
        return out

    item_meta, meta_info = load_item_meta()
    terms = load_terms()

    print("loading train positives...")
    train = pd.read_csv(TRAIN, usecols=["term_id", "item_id"])
    train["term_id"] = train["term_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    train = train.merge(item_meta[["item_id", "brand_norm", "category_norm"]], on="item_id", how="left")
    train = train.merge(terms[["term_id", "query_tokens"]], on="term_id", how="left")

    item_count = train["item_id"].value_counts()
    brand_count = train["brand_norm"].fillna("").value_counts()
    cat_count = train["category_norm"].fillna("").value_counts()

    max_item = math.log1p(float(item_count.max())) if len(item_count) else 1.0
    max_brand = math.log1p(float(brand_count.max())) if len(brand_count) else 1.0
    max_cat = math.log1p(float(cat_count.max())) if len(cat_count) else 1.0

    print("building token -> brand/category priors...")
    token_brand = defaultdict(Counter)
    token_cat = defaultdict(Counter)
    token_total_brand = Counter()
    token_total_cat = Counter()

    for row in train[["query_tokens", "brand_norm", "category_norm"]].itertuples(index=False):
        qtokens = row.query_tokens if isinstance(row.query_tokens, list) else []
        brand = str(row.brand_norm) if not pd.isna(row.brand_norm) else ""
        cat = str(row.category_norm) if not pd.isna(row.category_norm) else ""
        for t in set(qtokens):
            if brand:
                token_brand[t][brand] += 1
                token_total_brand[t] += 1
            if cat:
                token_cat[t][cat] += 1
                token_total_cat[t] += 1

    print("joining pair metadata...")
    df = pairs[["id", "term_id", "item_id"]].copy()
    df = df.merge(item_meta[["item_id", "brand_norm", "category_norm", "title_norm"]], on="item_id", how="left")
    df = df.merge(terms[["term_id", "query_norm", "query_tokens"]], on="term_id", how="left")

    df["brand_norm"] = df["brand_norm"].fillna("")
    df["category_norm"] = df["category_norm"].fillna("")
    df["title_norm"] = df["title_norm"].fillna("")
    df["query_norm"] = df["query_norm"].fillna("")

    print("mapping global popularity...")
    df["v43_item_pos_count"] = df["item_id"].map(item_count).fillna(0).astype(np.float32)
    df["v43_brand_pos_count"] = df["brand_norm"].map(brand_count).fillna(0).astype(np.float32)
    df["v43_category_pos_count"] = df["category_norm"].map(cat_count).fillna(0).astype(np.float32)

    df["v43_item_pop"] = (np.log1p(df["v43_item_pos_count"]) / max_item).astype(np.float32)
    df["v43_brand_pop"] = (np.log1p(df["v43_brand_pos_count"]) / max_brand).astype(np.float32)
    df["v43_category_pop"] = (np.log1p(df["v43_category_pos_count"]) / max_cat).astype(np.float32)

    print("scoring query-conditioned brand/category affinities...")
    brand_aff = np.zeros(len(df), dtype=np.float32)
    cat_aff = np.zeros(len(df), dtype=np.float32)
    brand_hit = np.zeros(len(df), dtype=np.int8)
    cat_hit = np.zeros(len(df), dtype=np.int8)
    title_brand_hit = np.zeros(len(df), dtype=np.int8)

    # group-wise cache avoids recomputing tokens for every row.
    for i, row in enumerate(df[["query_tokens", "brand_norm", "category_norm", "query_norm", "title_norm"]].itertuples(index=False)):
        qtokens = row.query_tokens if isinstance(row.query_tokens, list) else []
        brand = row.brand_norm or ""
        cat = row.category_norm or ""
        qnorm = row.query_norm or ""
        title = row.title_norm or ""

        brand_aff[i] = weighted_affinity(token_brand, token_total_brand, qtokens, brand)
        cat_aff[i] = weighted_affinity(token_cat, token_total_cat, qtokens, cat)

        if brand and brand in qnorm:
            brand_hit[i] = 1
        if cat and any(t in qnorm for t in cat.split() if len(t) >= 3):
            cat_hit[i] = 1
        if brand and brand in title:
            title_brand_hit[i] = 1

        if (i + 1) % 500_000 == 0:
            print("affinity rows:", i + 1, "/", len(df))

    df["v43_qbrand_affinity"] = brand_aff
    df["v43_qcategory_affinity"] = cat_aff
    df["v43_brand_query_hit"] = brand_hit
    df["v43_category_query_hit"] = cat_hit
    df["v43_title_brand_hit"] = title_brand_hit

    # Popularity trap: globally frequent item/brand/category but query-specific brand/category support weak.
    df["v43_global_pop"] = (
        0.50 * df["v43_item_pop"] +
        0.25 * df["v43_brand_pop"] +
        0.25 * df["v43_category_pop"]
    ).astype(np.float32)

    df["v43_query_pop_support"] = (
        0.38 * df["v43_qbrand_affinity"] +
        0.38 * df["v43_qcategory_affinity"] +
        0.10 * df["v43_brand_query_hit"] +
        0.08 * df["v43_category_query_hit"] +
        0.06 * df["v43_title_brand_hit"]
    ).astype(np.float32)

    df["v43_popularity_trap"] = (
        df["v43_global_pop"] * (1.0 - np.clip(df["v43_query_pop_support"], 0, 1))
    ).astype(np.float32)

    df["v43_pop_prior_score"] = (
        0.38 * df["v43_query_pop_support"] +
        0.22 * df["v43_item_pop"] +
        0.16 * df["v43_brand_pop"] +
        0.13 * df["v43_category_pop"] -
        0.11 * df["v43_popularity_trap"]
    ).astype(np.float32)

    # Within-term rank.
    df["v43_pop_rank"] = (
        df.groupby("term_id")["v43_pop_prior_score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    cnt = df.groupby("term_id")["id"].transform("count").astype(np.float32)
    df["v43_pop_pct_rank"] = ((df["v43_pop_rank"].astype(np.float32) - 1.0) / np.maximum(1.0, cnt - 1.0)).astype(np.float32)

    term_mean = df.groupby("term_id")["v43_pop_prior_score"].transform("mean").astype(np.float32)
    term_std = df.groupby("term_id")["v43_pop_prior_score"].transform("std").fillna(0).astype(np.float32)
    df["v43_pop_term_z"] = ((df["v43_pop_prior_score"].astype(np.float32) - term_mean) / np.maximum(1e-6, term_std)).astype(np.float32)

    out_cols = [
        "id", "term_id", "item_id",
        "v43_item_pos_count", "v43_brand_pos_count", "v43_category_pos_count",
        "v43_item_pop", "v43_brand_pop", "v43_category_pop", "v43_global_pop",
        "v43_qbrand_affinity", "v43_qcategory_affinity",
        "v43_brand_query_hit", "v43_category_query_hit", "v43_title_brand_hit",
        "v43_query_pop_support", "v43_popularity_trap",
        "v43_pop_prior_score", "v43_pop_rank", "v43_pop_pct_rank", "v43_pop_term_z",
    ]
    out = df[out_cols].copy()

    if not out["id"].astype(str).reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("V43 output id order mismatch")

    out.to_parquet(OUT_PRIOR, index=False)

    info = {
        **meta_info,
        "rows": int(len(out)),
        "score_mean": float(out["v43_pop_prior_score"].mean()),
        "score_std": float(out["v43_pop_prior_score"].std()),
        "score_max": float(out["v43_pop_prior_score"].max()),
        "item_nonzero_rate": float((out["v43_item_pos_count"] > 0).mean()),
        "brand_nonzero_rate": float((out["v43_brand_pos_count"] > 0).mean()),
        "category_nonzero_rate": float((out["v43_category_pos_count"] > 0).mean()),
        "qbrand_affinity_mean": float(out["v43_qbrand_affinity"].mean()),
        "qcategory_affinity_mean": float(out["v43_qcategory_affinity"].mean()),
        "brand_query_hit_rate": float(out["v43_brand_query_hit"].mean()),
        "category_query_hit_rate": float(out["v43_category_query_hit"].mean()),
    }
    OUT_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    return out


def load_labels():
    parts = []
    for name, path in LABEL_FILES.items():
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if "id" not in d.columns or "assistant_label" not in d.columns:
            continue
        d["id"] = d["id"].astype(str)
        d = d[d["assistant_label"].isin([0, 1, "0", "1"])].copy()
        if len(d) == 0:
            continue
        d["label"] = d["assistant_label"].astype(int)

        w = np.ones(len(d), dtype=np.float32)
        if "needs_recheck" in d.columns:
            nr = pd.to_numeric(d["needs_recheck"], errors="coerce").fillna(0).astype(int).to_numpy()
            w *= np.where(nr == 0, 1.0, 0.35)
        if "assistant_confidence" in d.columns:
            conf = d["assistant_confidence"].astype(str).str.lower().to_numpy()
            w *= np.where(conf == "high", 1.25, np.where(conf == "medium", 0.85, 0.55))
        if name.startswith("random"):
            w *= 1.30
        if name.startswith("manual"):
            w *= 1.20
        if name in ["v20_active", "v21_active", "v26_sparse"]:
            w *= 0.90

        d["sample_weight"] = w
        d["label_source"] = name
        parts.append(d[["id", "label", "sample_weight", "label_source"]])

    if not parts:
        raise RuntimeError("No label files available")

    lab = pd.concat(parts, ignore_index=True)

    agg = []
    for id_, g in lab.groupby("id"):
        pos = float(g.loc[g["label"] == 1, "sample_weight"].sum())
        neg = float(g.loc[g["label"] == 0, "sample_weight"].sum())
        total = pos + neg
        agg.append({
            "id": id_,
            "label": int(pos >= neg),
            "sample_weight": max(0.25, total),
            "label_margin": abs(pos - neg) / max(1e-9, total),
            "label_sources": "|".join(sorted(g["label_source"].unique())),
        })

    out = pd.DataFrame(agg)
    print("labels:", len(out), "pos_rate:", out["label"].mean())
    return out


def build_feature_frame(sample, pairs, v43, baselines):
    df = pairs[["id", "term_id", "item_id"]].copy()

    for c in v43.columns:
        if c not in ["id", "term_id", "item_id"]:
            df[c] = pd.to_numeric(v43[c], errors="coerce").fillna(0).astype(np.float32)

    external = [
        (V33, ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"]),
        (V34, ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"]),
        (V38, [
            "v38_lex_score", "v38_lex_pct_rank", "v38_lex_term_z",
            "v38_word_overlap_title", "v38_must_token_coverage_full",
            "v38_query_word_only_category_ratio",
        ]),
        (V39, ["v39_hist_score", "v39_hist_pct_rank", "v39_hist_has_item", "v39_hist_weighted_cov"]),
        (V41, ["v41_meta_prob"]),
        (V42, [
            "v42_graph_score", "v42_graph_norm_score", "v42_graph_max_sim",
            "v42_graph_hit_count", "v42_graph_pct_rank", "v42_graph_term_z",
        ]),
    ]

    for path, cols in external:
        x = safe_parquet(path, sample, cols)
        for c in cols:
            df[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(np.float32)

    df["v33_rs"] = (1.0 - df["v33_ft_pct_rank"]).clip(0, 1)
    df["v34_rs"] = (1.0 - df["v34_ce_pct_rank"]).clip(0, 1)
    df["v38_rs"] = (1.0 - df["v38_lex_pct_rank"]).clip(0, 1)
    df["v39_rs"] = (1.0 - df["v39_hist_pct_rank"]).clip(0, 1)
    df["v42_rs"] = (1.0 - df["v42_graph_pct_rank"]).clip(0, 1)
    df["v43_rs"] = (1.0 - df["v43_pop_pct_rank"]).clip(0, 1)

    df["v43_pop_sem_score"] = (
        0.30 * df["v43_rs"] +
        0.22 * df["v43_pop_prior_score"] +
        0.17 * df["v41_meta_prob"] +
        0.12 * df["v34_rs"] +
        0.09 * df["v33_rs"] +
        0.06 * df["v38_rs"] +
        0.04 * df["v42_rs"]
    ).astype(np.float32)

    df["v43_pop_graph_score"] = (
        0.28 * df["v43_rs"] +
        0.20 * df["v43_query_pop_support"] +
        0.18 * df["v42_rs"] +
        0.17 * df["v41_meta_prob"] +
        0.09 * df["v34_rs"] +
        0.05 * df["v38_rs"] +
        0.03 * df["v39_rs"]
    ).astype(np.float32)

    df["v43_pop_safe_score"] = (
        0.23 * df["v43_rs"] +
        0.20 * df["v43_query_pop_support"] +
        0.18 * df["v41_meta_prob"] +
        0.14 * df["v38_rs"] +
        0.12 * df["v34_rs"] +
        0.08 * df["v33_rs"] +
        0.05 * df["v42_rs"] -
        0.12 * df["v43_popularity_trap"]
    ).astype(np.float32)

    for name, pred in baselines.items():
        df[f"pred_{name}"] = pred.astype(np.int8)

    pred_cols = [f"pred_{name}" for name in baselines]
    df["pred_vote_mean"] = df[pred_cols].mean(axis=1).astype(np.float32)
    df["pred_vote_std"] = df[pred_cols].std(axis=1).fillna(0).astype(np.float32)

    return df


def feature_cols(df):
    drop = {"id", "term_id", "item_id"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def make_model(seed=42):
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            loss_function="Logloss",
            iterations=1700,
            depth=5,
            learning_rate=0.032,
            l2_leaf_reg=9.0,
            random_seed=seed,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
        ), "catboost"
    except Exception as e:
        print("CatBoost unavailable, sklearn fallback:", repr(e))
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.032,
            max_leaf_nodes=31,
            l2_regularization=0.10,
            random_state=seed,
        ), "hist_gradient_boosting"


def pred_proba(model, kind, X):
    return model.predict_proba(X)[:, 1]


def train_and_score(df, labels, fcols, sample):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    lab = labels.copy()
    lab["row_idx"] = lab["id"].map(id_to_idx)
    lab = lab[lab["row_idx"].notna()].copy()
    lab["row_idx"] = lab["row_idx"].astype(int)

    X = df.loc[lab["row_idx"].to_numpy(), fcols].replace([np.inf, -np.inf], 0).fillna(0)
    y = lab["label"].astype(int).to_numpy()
    w = lab["sample_weight"].astype(float).to_numpy()
    groups = df.loc[lab["row_idx"].to_numpy(), "term_id"].astype(str).to_numpy()

    n_splits = 5 if len(y) >= 500 and len(np.unique(groups)) >= 5 else 3
    splitter = GroupKFold(n_splits=n_splits) if len(np.unique(groups)) >= n_splits else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    split_iter = splitter.split(X, y, groups) if isinstance(splitter, GroupKFold) else splitter.split(X, y)

    oof = np.zeros(len(y), dtype=np.float32)
    rows = []

    for fold, (tr, va) in enumerate(split_iter, 1):
        model, kind = make_model(400 + fold)
        model.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        p = pred_proba(model, kind, X.iloc[va])
        oof[va] = p.astype(np.float32)

        auc = roc_auc_score(y[va], p) if len(np.unique(y[va])) == 2 else np.nan
        ap = average_precision_score(y[va], p) if len(np.unique(y[va])) == 2 else np.nan
        rows.append({
            "fold": fold,
            "model_kind": kind,
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "auc": float(auc) if not np.isnan(auc) else np.nan,
            "ap": float(ap) if not np.isnan(ap) else np.nan,
            "valid_pos_rate": float(y[va].mean()),
        })
        print("fold", fold, "auc", auc, "ap", ap)

    overall_auc = roc_auc_score(y, oof) if len(np.unique(y)) == 2 else np.nan
    overall_ap = average_precision_score(y, oof) if len(np.unique(y)) == 2 else np.nan

    best_f1 = -1
    best_t = 0.5
    for t in np.linspace(0.10, 0.90, 81):
        f = f1_score(y, (oof >= t).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)

    model_info = {
        "oof_auc": float(overall_auc) if not np.isnan(overall_auc) else np.nan,
        "oof_ap": float(overall_ap) if not np.isnan(overall_ap) else np.nan,
        "oof_best_macro_f1": float(best_f1),
        "oof_best_threshold": float(best_t),
        "n_features": int(len(fcols)),
        "n_labels": int(len(y)),
        "model_kind": rows[0]["model_kind"] if rows else "",
    }

    pd.DataFrame(rows + [{**{"fold": "OOF", "n_train": len(y), "n_valid": len(y), "auc": model_info["oof_auc"], "ap": model_info["oof_ap"], "valid_pos_rate": float(y.mean())}, **model_info}]).to_csv(OUT_MODEL, index=False)

    oof_df = lab[["id", "label", "sample_weight", "label_sources", "label_margin"]].copy()
    oof_df["v43_oof_prob"] = oof
    oof_df.to_csv(OUT_OOF, index=False)

    print("OOF:", model_info)

    final, kind = make_model(999)
    final.fit(X, y, sample_weight=w)

    print("scoring full pairs with V43 meta...")
    probs = np.zeros(len(df), dtype=np.float32)
    chunk = 250_000
    for start in range(0, len(df), chunk):
        end = min(len(df), start + chunk)
        Xc = df.iloc[start:end][fcols].replace([np.inf, -np.inf], 0).fillna(0)
        probs[start:end] = pred_proba(final, kind, Xc).astype(np.float32)
        print("scored", end, "/", len(df))

    pd.DataFrame({"id": sample["id"], "v43_meta_prob": probs}).to_parquet(OUT_PROB, index=False)
    return probs, model_info


def build_same_quota(df, score, anchor):
    tmp = pd.DataFrame({"term_id": df["term_id"].to_numpy(), "score": score, "anchor": anchor})
    quota = tmp.groupby("term_id")["anchor"].sum()
    tmp["quota"] = tmp["term_id"].map(quota).fillna(0).astype(np.int32)
    tmp["rank"] = tmp.groupby("term_id")["score"].rank(method="first", ascending=False).astype(np.int32)
    return (tmp["rank"].to_numpy() <= tmp["quota"].to_numpy()).astype(np.int8)


def pair_swaps(df, anchor, base, score_col):
    add_mask = (anchor == 0) & (base == 1)
    drop_mask = (anchor == 1) & (base == 0)

    cols = [
        "id", "term_id", "item_id", score_col, "v43_meta_prob",
        "v43_pop_prior_score", "v43_query_pop_support", "v43_popularity_trap",
        "v43_item_pop", "v43_brand_pop", "v43_category_pop",
        "v43_qbrand_affinity", "v43_qcategory_affinity",
        "v41_meta_prob", "v42_graph_score", "v42_graph_max_sim",
        "v33_rs", "v34_rs", "v38_rs", "v39_rs", "v42_rs",
        "v38_lex_score", "v38_word_overlap_title", "v38_query_word_only_category_ratio",
        "v39_hist_score", "v39_hist_has_item", "v39_hist_weighted_cov",
    ]

    cols = list(dict.fromkeys(cols))

    add = df.loc[add_mask, cols].copy()
    drop = df.loc[drop_mask, cols].copy()
    add = add.sort_values(["term_id", score_col], ascending=[True, False])
    drop = drop.sort_values(["term_id", score_col], ascending=[True, True])
    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    sw = add.merge(drop, on=["term_id", "pair_rank"], suffixes=("_add", "_drop"), how="inner")

    sw["score_gain"] = sw[f"{score_col}_add"] - sw[f"{score_col}_drop"]
    sw["prob_gain"] = sw["v43_meta_prob_add"] - sw["v43_meta_prob_drop"]
    sw["pop_gain"] = sw["v43_pop_prior_score_add"] - sw["v43_pop_prior_score_drop"]
    sw["support_gain"] = sw["v43_query_pop_support_add"] - sw["v43_query_pop_support_drop"]
    sw["sem_gain"] = ((sw["v33_rs_add"] + sw["v34_rs_add"]) / 2) - ((sw["v33_rs_drop"] + sw["v34_rs_drop"]) / 2)
    sw["lex_gain"] = sw["v38_lex_score_add"] - sw["v38_lex_score_drop"]
    sw["graph_gain"] = sw["v42_graph_score_add"] - sw["v42_graph_score_drop"]

    sw["pop_trap_add"] = (
        (sw["v43_popularity_trap_add"] >= 0.50)
        & (sw["v43_query_pop_support_add"] <= 0.12)
        & (sw["v38_word_overlap_title_add"] <= 0.10)
    )

    sw["category_only_risk"] = (
        (sw["v38_query_word_only_category_ratio_add"] >= 0.50)
        & (sw["v38_word_overlap_title_add"] <= 0.10)
        & (sw["v43_query_pop_support_add"] <= 0.12)
    )

    sw["history_contra"] = (
        (sw["v39_hist_has_item_add"].astype(int) == 1)
        & (sw["v39_hist_has_item_drop"].astype(int) == 1)
        & (sw["v39_hist_score_drop"] >= sw["v39_hist_score_add"] + 0.12)
        & (sw["v39_hist_weighted_cov_drop"] >= sw["v39_hist_weighted_cov_add"] + 0.18)
    )

    sw["v43_swap_support_score"] = (
        0.54 * sw["prob_gain"]
        + 0.18 * sw["score_gain"]
        + 0.16 * sw["support_gain"]
        + 0.10 * sw["sem_gain"]
        + 0.06 * sw["lex_gain"]
        + 0.04 * np.tanh(sw["graph_gain"])
        - 0.16 * sw["pop_trap_add"].astype(float)
        - 0.12 * sw["category_only_risk"].astype(float)
        - 0.12 * sw["history_contra"].astype(float)
    ).astype(np.float32)

    return sw.sort_values(["v43_swap_support_score", "prob_gain"], ascending=False).reset_index(drop=True)


def v43_mask(sw, mode):
    bad = sw["pop_trap_add"] | sw["category_only_risk"] | sw["history_contra"]

    if mode == "ultra":
        return (
            (sw["v43_meta_prob_add"] >= 0.82)
            & (sw["v43_meta_prob_drop"] <= 0.48)
            & (sw["prob_gain"] >= 0.28)
            & (sw["v43_query_pop_support_add"] >= 0.12)
            & (~bad)
        )

    if mode == "strict":
        return (
            (sw["v43_meta_prob_add"] >= 0.74)
            & (sw["v43_meta_prob_drop"] <= 0.54)
            & (sw["prob_gain"] >= 0.18)
            & (sw["v43_query_pop_support_add"] >= 0.08)
            & (~bad)
        )

    if mode == "balanced":
        return (
            (sw["v43_meta_prob_add"] >= 0.66)
            & (sw["prob_gain"] >= 0.10)
            & (sw["v43_swap_support_score"] >= 0.10)
            & (~(sw["pop_trap_add"] | sw["category_only_risk"]))
        )

    if mode == "impact":
        return (
            (sw["v43_meta_prob_add"] >= 0.58)
            & (sw["prob_gain"] >= 0.05)
            & (sw["v43_swap_support_score"] >= 0.05)
            & (~sw["category_only_risk"])
        )

    raise ValueError(mode)


def apply_swaps(sample, anchor, sw, cap):
    take = sw.sort_values(["v43_swap_support_score", "prob_gain"], ascending=False).head(cap).copy()
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
                row = {
                    "label_set": label_set,
                    "subset": subset_name,
                    "eval_key": f"{label_set}_{subset_name}",
                    "variant": name,
                    "n": int(len(y)),
                }
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


def evaluate_and_weight(variants, sample):
    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)
    return eval_df, weighted_score(eval_df)


def build_review(chosen):
    parts = []
    for name, sw in chosen.items():
        if len(sw) == 0:
            continue
        x = sw.copy()
        x["variant"] = name
        parts.append(x.head(120))
        parts.append(x.tail(80))
    if not parts:
        return
    r = pd.concat(parts, ignore_index=True).drop_duplicates(["variant", "id_add", "id_drop"], keep="first")
    keep = [
        "variant", "term_id", "pair_rank", "id_add", "item_id_add", "id_drop", "item_id_drop",
        "v43_swap_support_score", "prob_gain", "score_gain", "pop_gain", "support_gain", "sem_gain", "lex_gain", "graph_gain",
        "v43_meta_prob_add", "v43_meta_prob_drop",
        "v43_pop_prior_score_add", "v43_pop_prior_score_drop",
        "v43_query_pop_support_add", "v43_query_pop_support_drop",
        "v43_popularity_trap_add", "v43_popularity_trap_drop",
        "v43_item_pop_add", "v43_item_pop_drop",
        "v43_brand_pop_add", "v43_brand_pop_drop",
        "v43_category_pop_add", "v43_category_pop_drop",
        "pop_trap_add", "category_only_risk", "history_contra",
    ]
    r[[c for c in keep if c in r.columns]].to_csv(OUT_REVIEW, index=False)


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print("loading sample/pairs...")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("pairs order mismatch sample")

    anchor_path = first_existing(ANCHOR_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor missing")

    anchor = load_pred(anchor_path, sample)
    baselines = {"anchor_v33_qprob2000": anchor.copy()}

    for name, paths in DIRECT_BASES.items():
        p = first_existing(paths)
        if p is not None:
            baselines[name] = load_pred(p, sample)
            print("baseline:", name, p, "diff", int((baselines[name] != anchor).sum()))

    for short_name, variant in SAVED_VARIANTS.items():
        p = find_saved_variant(variant)
        if p is not None:
            baselines[short_name] = load_pred(p, sample)
            print("saved baseline:", short_name, p, "diff", int((baselines[short_name] != anchor).sum()))

    v43 = build_v43_priors(sample, pairs)
    df = build_feature_frame(sample, pairs, v43, baselines)
    fcols = feature_cols(df)
    labels = load_labels()

    if OUT_PROB.exists() and OUT_MODEL.exists():
        print("using existing V43 meta probability:", OUT_PROB)
        _p = pd.read_parquet(OUT_PROB)
        _p["id"] = _p["id"].astype(str)

        if not _p["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            raise RuntimeError("existing V43 meta prob id order mismatch")

        probs = _p["v43_meta_prob"].astype(np.float32).to_numpy()

        _m = pd.read_csv(OUT_MODEL)
        _oof = _m[_m["fold"].astype(str).eq("OOF")]

        if len(_oof):
            _r = _oof.iloc[-1].to_dict()
            model_info = {
                "oof_auc": float(_r.get("oof_auc", _r.get("auc", np.nan))),
                "oof_ap": float(_r.get("oof_ap", _r.get("ap", np.nan))),
                "oof_best_macro_f1": float(_r.get("oof_best_macro_f1", np.nan)),
                "oof_best_threshold": float(_r.get("oof_best_threshold", np.nan)),
                "n_features": int(float(_r.get("n_features", len(fcols)))),
                "n_labels": int(float(_r.get("n_labels", len(labels)))),
                "model_kind": str(_r.get("model_kind", "")),
            }
        else:
            model_info = {
                "oof_auc": np.nan,
                "oof_ap": np.nan,
                "oof_best_macro_f1": np.nan,
                "oof_best_threshold": np.nan,
                "n_features": len(fcols),
                "n_labels": len(labels),
                "model_kind": "",
            }
    else:
        probs, model_info = train_and_score(df, labels, fcols, sample)

    df["v43_meta_prob"] = probs

    variants = dict(baselines)
    meta_rows = []
    chosen = {}

    # Same quota learned popularity model outputs.
    for name, score in {
        "v43_meta_same_quota": probs,
        "v43_pop_sem_score_same_quota": df["v43_pop_sem_score"].to_numpy(np.float32),
        "v43_pop_graph_score_same_quota": df["v43_pop_graph_score"].to_numpy(np.float32),
        "v43_pop_safe_score_same_quota": df["v43_pop_safe_score"].to_numpy(np.float32),
    }.items():
        pred = build_same_quota(df, score, anchor)
        variants[name] = pred
        meta_rows.append({
            "variant": name,
            "source": "pop_brand_category_same_quota",
            "base": "anchor",
            "score_col": name.replace("_same_quota", ""),
            "mode": "",
            "cap": "",
            "used_swaps": int((pred != anchor).sum() // 2),
            "accepted_pool": "",
            "prob_gain_mean": np.nan,
            "query_pop_support_mean": np.nan,
            "pop_trap_rate": np.nan,
        })
        print("same quota:", name, "diff", int((pred != anchor).sum()))

    bases = [
        "v41_meta_same_quota",
        "v42_qgraph_v41_balanced3600",
        "raw_v35_b5000",
        "v40_history_impact2200",
        "v38_precision_loose_2800",
        "v39_raw35_history_lex_veto_bad",
        "v36p2_balanced",
        "v36p2_strict",
    ]
    bases = [b for b in bases if b in baselines]

    score_cols = ["v43_meta_prob", "v43_pop_safe_score", "v43_pop_graph_score", "v43_pop_sem_score"]
    caps = {
        "ultra": [500, 900, 1300],
        "strict": [900, 1500, 2400],
        "balanced": [1500, 2600, 4200],
        "impact": [2400, 4200, 7000],
    }

    for base in bases:
        for sc in score_cols:
            sw = pair_swaps(df, anchor, baselines[base], sc)
            print("swap pool:", base, sc, len(sw))
            for mode, cap_list in caps.items():
                mask = v43_mask(sw, mode)
                acc = sw[mask].copy()
                if len(acc) == 0:
                    continue
                for cap in cap_list:
                    pred, take = apply_swaps(sample, anchor, acc, min(cap, len(acc)))
                    vname = f"v43_{base}_{sc}_{mode}_cap{cap}"
                    variants[vname] = pred
                    chosen[vname] = take
                    meta_rows.append({
                        "variant": vname,
                        "source": "pop_brand_category_swap_filter",
                        "base": base,
                        "score_col": sc,
                        "mode": mode,
                        "cap": cap,
                        "accepted_pool": int(len(acc)),
                        "used_swaps": int(len(take)),
                        "prob_gain_mean": float(take["prob_gain"].mean()) if len(take) else np.nan,
                        "query_pop_support_mean": float(take["v43_query_pop_support_add"].mean()) if len(take) else np.nan,
                        "pop_trap_rate": float(take["pop_trap_add"].mean()) if len(take) else np.nan,
                        "support_score_mean": float(take["v43_swap_support_score"].mean()) if len(take) else np.nan,
                    })
                    print("candidate:", vname, "used", len(take), "diff", int((pred != anchor).sum()))

    print("evaluating...")
    eval_df, wdf = evaluate_and_weight(variants, sample)

    summary = pd.DataFrame(meta_rows)

    for name, pred in baselines.items():
        summary = pd.concat([summary, pd.DataFrame([{
            "variant": name,
            "source": "baseline",
            "base": "",
            "score_col": "",
            "mode": "",
            "cap": "",
            "used_swaps": "",
            "accepted_pool": "",
            "prob_gain_mean": np.nan,
            "query_pop_support_mean": np.nan,
            "pop_trap_rate": np.nan,
            "support_score_mean": np.nan,
        }])], ignore_index=True)

    aux = []
    for name, pred in variants.items():
        aux.append({
            "variant": name,
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != baselines["raw_v35_b5000"]).sum()) if "raw_v35_b5000" in baselines else -1,
            "diff_vs_v41_same": int((pred != baselines["v41_meta_same_quota"]).sum()) if "v41_meta_same_quota" in baselines else -1,
            "diff_vs_v42_qgraph": int((pred != baselines["v42_qgraph_v41_balanced3600"]).sum()) if "v42_qgraph_v41_balanced3600" in baselines else -1,
        })
    summary = summary.merge(pd.DataFrame(aux), on="variant", how="right")

    if len(wdf):
        summary = summary.merge(wdf, on="variant", how="left")

    for k, v in model_info.items():
        summary[k] = v

    diff = summary["diff_vs_anchor"].fillna(0).astype(float)
    move_bonus = np.minimum(0.020, np.log1p(diff) / np.log1p(50000) * 0.020)
    prob_bonus = np.minimum(0.004, summary["prob_gain_mean"].fillna(0).astype(float) * 0.010)
    pop_bonus = np.minimum(0.004, summary["query_pop_support_mean"].fillna(0).astype(float) * 0.010)
    too_big_penalty = np.maximum(0, diff - 110000) / 600000.0
    overfit_penalty = max(0.0, 0.88 - float(model_info["oof_auc"])) * 0.020 if not pd.isna(model_info["oof_auc"]) else 0.010

    summary["v43_decision_score"] = (
        summary["weighted_macro"].fillna(0)
        + move_bonus
        + prob_bonus
        + pop_bonus
        - too_big_penalty
        - overfit_penalty
    )

    summary = summary.sort_values(["v43_decision_score", "weighted_macro"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    build_review(chosen)

    save_names = []
    top = summary[
        (summary["source"].isin(["pop_brand_category_same_quota", "pop_brand_category_swap_filter"]))
        & (summary["diff_vs_anchor"] >= 1500)
        & (summary["diff_vs_anchor"] <= 130000)
    ].head(50)

    for nm in top["variant"].tolist():
        if nm not in save_names:
            save_names.append(nm)

    for nm in [
        "v43_meta_same_quota",
        "v43_pop_safe_score_same_quota",
        "v41_meta_same_quota",
        "v42_qgraph_v41_balanced3600",
        "raw_v35_b5000",
        "v40_history_impact2200",
        "v38_precision_loose_2800",
        "anchor_v33_qprob2000",
    ]:
        if nm in variants and nm not in save_names:
            save_names.append(nm)

    saved = []
    for i, nm in enumerate(save_names[:60], 1):
        pred = variants[nm]
        out = SUB_DIR / f"FINAL_CANDIDATE_v43_popbc_{i:03d}_{short_hash(nm)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)
        row = summary[summary["variant"] == nm].iloc[0].to_dict()
        row["file"] = str(out)
        saved.append(row)
        print("saved:", out, "<-", nm)

    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)

    print("\nMODEL INFO")
    print(model_info)

    print("\nTOP V43")
    cols = [
        "variant", "v43_decision_score", "weighted_macro", "source", "base", "score_col", "mode", "cap",
        "used_swaps", "accepted_pool", "prob_gain_mean", "query_pop_support_mean", "pop_trap_rate",
        "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v41_same", "diff_vs_v42_qgraph",
        "oof_auc", "oof_ap", "oof_best_macro_f1", "mean_precision", "mean_recall",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(100).to_string(index=False))

    print("\nSAVED")
    sdf = pd.DataFrame(saved)
    if len(sdf):
        print(sdf[["variant", "file", "v43_decision_score", "weighted_macro", "diff_vs_anchor"]].to_string(index=False))

    print("\noutputs:")
    print(OUT_PRIOR)
    print(OUT_PROB)
    print(OUT_INFO)
    print(OUT_MODEL)
    print(OUT_OOF)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_SAVED)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
