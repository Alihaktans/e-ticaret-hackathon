from pathlib import Path
import re
import unicodedata
import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
SUB_PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_MEM = ROOT / "data/processed/v48_train_memory_bridge_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v48_train_memory_bridge_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v48_train_memory_bridge_saved_candidates.csv"
OUT_REVIEW = OUT_DIR / "review_v48_train_memory_bridge_swaps.csv"
OUT_QMATCH = OUT_DIR / "review_v48_train_memory_query_matches.csv"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

BASELINE_CSV_CANDIDATES = {
    "raw_v35_b5000": [
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    ],
    "v41_meta_same_quota": [
        ROOT / "submissions/FINAL_CANDIDATE_v41_meta_001_22655235.csv",
    ],
    "v42_qgraph_v41_balanced3600": [
        ROOT / "submissions/final_candidate_last/A_v42_qgraph_v41_balanced3600.csv",
    ],
}

SCORE_SPECS = [
    (ROOT / "data/processed/v33_ft_pair_scores.parquet", "v33", ["v33_ft_score", "ft_score", "score"]),
    (ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet", "v34", ["v34_ce_score", "ce_score", "score"]),
    (ROOT / "data/processed/v38_lexical_pair_scores.parquet", "v38", ["v38_lex_score", "lex_score", "score"]),
    (ROOT / "data/processed/v41_learned_meta_prob.parquet", "v41", ["v41_meta_prob", "meta_prob", "prob"]),
    (ROOT / "data/processed/v42_query_graph_pair_scores.parquet", "v42", ["v42_graph_score", "v42_graph_norm_score", "graph_score", "score"]),
    (ROOT / "data/processed/v43_pop_brand_category_meta_prob.parquet", "v43", ["v43_meta_prob", "meta_prob", "prob"]),
]

TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def first_existing(paths):
    for p in paths:
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


def pick_score_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def add_score(df, sample, path, out_col, candidates):
    if not path.exists():
        print("missing:", path)
        df[out_col] = np.float32(0)
        return df

    s = pd.read_parquet(path)
    if "id" not in s.columns:
        df[out_col] = np.float32(0)
        return df

    col = pick_score_col(s, candidates)
    if col is None:
        df[out_col] = np.float32(0)
        return df

    tmp = s[["id", col]].copy()
    tmp["id"] = tmp["id"].astype(str)
    tmp = tmp.rename(columns={col: out_col})

    if tmp["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        df[out_col] = pd.to_numeric(tmp[out_col], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    else:
        before = len(df)
        df = df.merge(tmp, on="id", how="left", validate="one_to_one")
        assert len(df) == before
        df[out_col] = pd.to_numeric(df[out_col], errors="coerce").fillna(0).astype(np.float32)

    print("score", out_col, "from", path, "col", col, "mean", float(df[out_col].mean()))
    return df


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
    items = pd.read_csv(ITEMS)
    cols = list(items.columns)
    item_col = detect_col(cols, exacts=("item_id", "product_id", "id"), contains=("item_id",))
    title_col = detect_col(cols, exacts=("title", "name", "product_name", "item_name", "urun_adi", "ürün_adı"), contains=("title", "name", "urun", "ürün"))
    brand_col = detect_col(cols, exacts=("brand", "marka"), contains=("brand", "marka"))
    cat_col = detect_col(cols, exacts=("category", "kategori", "category_name", "leaf_category", "cat"), contains=("category", "kategori", "cat"))

    out = pd.DataFrame({"item_id": items[item_col].astype(str)})
    out["title"] = items[title_col].astype(str) if title_col else ""
    out["brand"] = items[brand_col].astype(str) if brand_col else ""
    out["category"] = items[cat_col].astype(str) if cat_col else ""
    return out


def load_terms():
    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    if "query" not in terms.columns:
        cand = [c for c in terms.columns if c != "term_id" and terms[c].dtype == "object"]
        if not cand:
            raise RuntimeError("query column not found in terms")
        terms = terms.rename(columns={cand[0]: "query"})
    terms["query_norm"] = terms["query"].map(norm_text)
    return terms[["term_id", "query", "query_norm"]].copy()


def build_query_fuzzy_map(train_queries, test_queries, min_sim=0.90, topn=3):
    """
    Char n-gram nearest neighbor between normalized queries.
    Returns long DataFrame: test_query_norm, train_query_norm, sim.
    """
    train_list = sorted(pd.Series(train_queries).dropna().astype(str).unique())
    test_list = sorted(pd.Series(test_queries).dropna().astype(str).unique())

    if not train_list or not test_list:
        return pd.DataFrame(columns=["test_query_norm", "train_query_norm", "query_sim"])

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, dtype=np.float32)
    X_train = vectorizer.fit_transform(train_list)
    X_test = vectorizer.transform(test_list)

    nn = NearestNeighbors(n_neighbors=min(topn, len(train_list)), metric="cosine", algorithm="brute")
    nn.fit(X_train)
    dist, ind = nn.kneighbors(X_test, return_distance=True)

    rows = []
    for i, tq in enumerate(test_list):
        for d, j in zip(dist[i], ind[i]):
            sim = 1.0 - float(d)
            if sim >= min_sim:
                rows.append({
                    "test_query_norm": tq,
                    "train_query_norm": train_list[int(j)],
                    "query_sim": sim,
                })
    return pd.DataFrame(rows)


def build_memory_scores(sample, sub_pairs):
    if OUT_MEM.exists():
        print("using existing:", OUT_MEM)
        d = pd.read_parquet(OUT_MEM)
        d["id"] = d["id"].astype(str)
        if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            return d
        print("existing memory score id order mismatch, rebuilding")

    print("loading terms/train...")
    terms = load_terms()
    train = pd.read_csv(TRAIN_PAIRS)
    train["term_id"] = train["term_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    train = train.merge(terms, on="term_id", how="left")
    sub = sub_pairs.merge(terms, on="term_id", how="left")
    sub["query_norm"] = sub["query_norm"].fillna("").astype(str)

    # exact query -> item positive memory
    train_pos = train[["query_norm", "item_id"]].dropna().drop_duplicates()
    train_pos["mem_exact_item_hit"] = 1

    df = sub.merge(train_pos, on=["query_norm", "item_id"], how="left")
    df["mem_exact_item_hit"] = df["mem_exact_item_hit"].fillna(0).astype(np.int8)

    # query-level item frequency from train positives
    item_freq = train.groupby("item_id").size().rename("train_item_pos_count").reset_index()
    q_freq = train.groupby("query_norm").size().rename("train_query_pos_count").reset_index()
    df = df.merge(item_freq, on="item_id", how="left")
    df = df.merge(q_freq, on="query_norm", how="left")
    df["train_item_pos_count"] = df["train_item_pos_count"].fillna(0).astype(np.int32)
    df["train_query_pos_count"] = df["train_query_pos_count"].fillna(0).astype(np.int32)

    # fuzzy query transfer: only count if same item was positive for a very close train query
    print("building fuzzy query memory...")
    test_queries = df["query_norm"].unique()
    train_queries = train["query_norm"].dropna().unique()
    qmap = build_query_fuzzy_map(train_queries, test_queries, min_sim=0.91, topn=3)
    qmap = qmap[qmap["test_query_norm"] != qmap["train_query_norm"]].copy()
    qmap.to_csv(OUT_QMATCH, index=False)

    if len(qmap):
        fuzzy_pos = qmap.merge(
            train_pos.rename(columns={"query_norm": "train_query_norm"}).drop(columns=["mem_exact_item_hit"]),
            on="train_query_norm",
            how="inner",
        )
        fuzzy_pos = fuzzy_pos.sort_values("query_sim", ascending=False)
        fuzzy_best = fuzzy_pos.groupby(["test_query_norm", "item_id"], as_index=False)["query_sim"].max()
        fuzzy_best = fuzzy_best.rename(columns={"test_query_norm": "query_norm", "query_sim": "mem_fuzzy_item_sim"})
        df = df.merge(fuzzy_best, on=["query_norm", "item_id"], how="left")
    else:
        df["mem_fuzzy_item_sim"] = 0.0

    df["mem_fuzzy_item_sim"] = pd.to_numeric(df["mem_fuzzy_item_sim"], errors="coerce").fillna(0).astype(np.float32)
    df["mem_fuzzy_item_hit"] = (df["mem_fuzzy_item_sim"] >= 0.91).astype(np.int8)

    # add existing model scores
    for path, out_col, candidates in SCORE_SPECS:
        df = add_score(df, sample, path, out_col, candidates)

    # consensus signal for memory hits
    for c in ["v33", "v34", "v38", "v41", "v42", "v43"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

    # query-relative ranks for dropping bad anchor positives
    for c in ["v33", "v34", "v38", "v41", "v43"]:
        r = df.groupby("term_id", sort=False)[c].rank(method="first", ascending=False).astype(np.float32)
        n = df.groupby("term_id", sort=False)[c].transform("size").astype(np.float32)
        df[c + "_query_pct"] = np.where(n > 1, 1.0 - ((r - 1.0) / (n - 1.0)), 1.0).astype(np.float32)

    df["mem_hit_strength"] = (
        1.00 * df["mem_exact_item_hit"].astype(float)
        + 0.80 * df["mem_fuzzy_item_hit"].astype(float) * df["mem_fuzzy_item_sim"].astype(float)
        + 0.08 * np.log1p(df["train_item_pos_count"].astype(float))
    ).astype(np.float32)

    df["model_support"] = (
        0.28 * df["v34_query_pct"]
        + 0.24 * df["v33_query_pct"]
        + 0.18 * df["v38_query_pct"]
        + 0.18 * df["v41_query_pct"]
        + 0.12 * df["v43_query_pct"]
    ).astype(np.float32)

    df["v48_memory_score"] = (
        1.30 * df["mem_hit_strength"]
        + 0.45 * df["model_support"]
        + 0.06 * np.log1p(df["train_item_pos_count"].astype(float))
    ).astype(np.float32)

    keep = [
        "id", "term_id", "item_id", "query", "query_norm",
        "mem_exact_item_hit", "mem_fuzzy_item_hit", "mem_fuzzy_item_sim",
        "train_item_pos_count", "train_query_pos_count",
        "mem_hit_strength", "model_support", "v48_memory_score",
        "v33", "v34", "v38", "v41", "v42", "v43",
        "v33_query_pct", "v34_query_pct", "v38_query_pct", "v41_query_pct", "v43_query_pct",
    ]
    OUT_MEM.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_parquet(OUT_MEM, index=False)
    print("wrote", OUT_MEM)
    print("exact hits:", int(df["mem_exact_item_hit"].sum()), "fuzzy hits:", int(df["mem_fuzzy_item_hit"].sum()))
    return df[keep]


def pair_swaps(df, anchor, add_mask, score_col="v48_memory_score"):
    drop_mask = anchor == 1

    add_cols = [
        "id", "term_id", "item_id", score_col, "mem_exact_item_hit", "mem_fuzzy_item_hit",
        "mem_fuzzy_item_sim", "train_item_pos_count", "model_support",
        "v33", "v34", "v38", "v41", "v43",
    ]
    drop_cols = [
        "id", "term_id", "item_id", score_col, "model_support",
        "v33", "v34", "v38", "v41", "v43",
    ]
    add_cols = list(dict.fromkeys(add_cols))
    drop_cols = list(dict.fromkeys(drop_cols))

    add = df.loc[add_mask, add_cols].copy().rename(columns={"id": "id_add", "item_id": "item_id_add"})
    drop = df.loc[drop_mask, drop_cols].copy().rename(columns={"id": "id_drop", "item_id": "item_id_drop"})

    add = add.sort_values(["term_id", score_col], ascending=[True, False])
    drop = drop.sort_values(["term_id", score_col], ascending=[True, True])

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    sw = add.merge(drop, on=["term_id", "pair_rank"], how="inner", suffixes=("_add", "_drop"))

    # add-only memory columns do not get suffix from pandas merge
    for c in ["mem_exact_item_hit", "mem_fuzzy_item_hit", "mem_fuzzy_item_sim", "train_item_pos_count"]:
        if c in sw.columns and f"{c}_add" not in sw.columns:
            sw = sw.rename(columns={c: f"{c}_add"})

    for c in ["mem_exact_item_hit_add", "mem_fuzzy_item_hit_add", "mem_fuzzy_item_sim_add", "train_item_pos_count_add"]:
        if c not in sw.columns:
            sw[c] = 0

    sw["score_gain"] = sw[f"{score_col}_add"] - sw[f"{score_col}_drop"]
    sw["model_support_gain"] = sw["model_support_add"] - sw["model_support_drop"]

    sw["core_win_count"] = (
        (sw["v33_add"] > sw["v33_drop"]).astype(int)
        + (sw["v34_add"] > sw["v34_drop"]).astype(int)
        + (sw["v38_add"] > sw["v38_drop"]).astype(int)
        + (sw["v41_add"] > sw["v41_drop"]).astype(int)
        + (sw["v43_add"] > sw["v43_drop"]).astype(int)
    )

    sw["v48_swap_score"] = (
        1.00 * sw["score_gain"]
        + 0.20 * sw["model_support_gain"]
        + 0.05 * sw["core_win_count"]
        + 0.15 * sw["mem_exact_item_hit_add"]
        + 0.08 * sw["mem_fuzzy_item_sim_add"]
    )
    return sw.sort_values("v48_swap_score", ascending=False).reset_index(drop=True)


def apply_swaps(sample, anchor, sw, cap):
    take = sw.sort_values("v48_swap_score", ascending=False).head(cap).copy()
    pred = anchor.copy()
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    add_idx = take["id_add"].map(id_to_idx)
    drop_idx = take["id_drop"].map(id_to_idx)
    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("swap id map failed")
    pred[add_idx.astype(int).to_numpy()] = 1
    pred[drop_idx.astype(int).to_numpy()] = 0
    return pred, take


def enrich_review(sw):
    if len(sw) == 0:
        return sw

    if "variant" not in sw.columns:
        sw["variant"] = ""
    item_meta = load_item_meta()
    terms = load_terms()[["term_id", "query"]].copy()

    add = item_meta.rename(columns={"item_id": "item_id_add", "title": "add_title", "brand": "add_brand", "category": "add_category"})
    drop = item_meta.rename(columns={"item_id": "item_id_drop", "title": "drop_title", "brand": "drop_brand", "category": "drop_category"})

    out = sw.merge(terms, on="term_id", how="left")
    out = out.merge(add, on="item_id_add", how="left")
    out = out.merge(drop, on="item_id_drop", how="left")

    front = [
        "variant", "mode", "term_id", "query",
        "id_add", "item_id_add", "add_title", "add_brand", "add_category",
        "id_drop", "item_id_drop", "drop_title", "drop_brand", "drop_category",
        "v48_swap_score", "score_gain", "model_support_gain",
        "mem_exact_item_hit_add", "mem_fuzzy_item_hit_add", "mem_fuzzy_item_sim_add",
        "train_item_pos_count_add", "model_support_add", "model_support_drop", "core_win_count",
    ]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    sub_pairs = pd.read_csv(SUB_PAIRS, usecols=["id", "term_id", "item_id"])
    sub_pairs["id"] = sub_pairs["id"].astype(str)
    sub_pairs["term_id"] = sub_pairs["term_id"].astype(str)
    sub_pairs["item_id"] = sub_pairs["item_id"].astype(str)

    if not sample["id"].reset_index(drop=True).equals(sub_pairs["id"].reset_index(drop=True)):
        raise RuntimeError("sample/submission_pairs order mismatch")

    anchor_path = first_existing(ANCHOR_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor missing")
    anchor = load_pred(anchor_path, sample)
    print("anchor ones:", int(anchor.sum()), "pos:", float(anchor.mean()))

    df = build_memory_scores(sample, sub_pairs)

    variants = {"anchor_v33_qprob2000": anchor.copy()}
    for name, paths in BASELINE_CSV_CANDIDATES.items():
        p = first_existing(paths)
        if p is not None:
            variants[name] = load_pred(p, sample)
            print("baseline", name, "diff", int((variants[name] != anchor).sum()))

    add_exact = (anchor == 0) & (df["mem_exact_item_hit"].to_numpy() == 1)
    add_fuzzy = (anchor == 0) & ((df["mem_exact_item_hit"].to_numpy() == 1) | ((df["mem_fuzzy_item_hit"].to_numpy() == 1) & (df["model_support"].to_numpy() >= 0.55)))
    add_any = (anchor == 0) & ((df["mem_exact_item_hit"].to_numpy() == 1) | ((df["mem_fuzzy_item_hit"].to_numpy() == 1) & (df["model_support"].to_numpy() >= 0.45)))

    modes = [
        ("exact", add_exact),
        ("exact_fuzzy_strict", add_fuzzy),
        ("exact_fuzzy_balanced", add_any),
    ]

    caps = [100, 250, 500, 900, 1400, 2200, 3500]
    summaries = []
    reviews = []

    for mode, add_mask in modes:
        sw0 = pair_swaps(df, anchor, add_mask)
        sw0["mode"] = mode
        print(mode, "raw add candidates:", int(add_mask.sum()), "paired swaps:", len(sw0))

        if len(sw0):
            reviews.append(sw0.head(500))

        for cap in caps:
            used = min(cap, len(sw0))
            if used <= 0:
                continue
            pred, take = apply_swaps(sample, anchor, sw0, used)
            vname = f"v48_memory_{mode}_cap{cap}"
            variants[vname] = pred

            summaries.append({
                "variant": vname,
                "source": "train_memory_bridge",
                "mode": mode,
                "cap": cap,
                "used_swaps": int(len(take)),
                "accepted_pool": int(len(sw0)),
                "exact_rate": float(take["mem_exact_item_hit_add"].mean()),
                "fuzzy_rate": float(take["mem_fuzzy_item_hit_add"].mean()),
                "fuzzy_sim_mean": float(take["mem_fuzzy_item_sim_add"].mean()),
                "train_item_pos_mean": float(take["train_item_pos_count_add"].mean()),
                "model_support_mean": float(take["model_support_add"].mean()),
                "model_support_gain_mean": float(take["model_support_gain"].mean()),
                "core_win_mean": float(take["core_win_count"].mean()),
                "score_gain_mean": float(take["score_gain"].mean()),
                "v48_swap_score_mean": float(take["v48_swap_score"].mean()),
            })

    summary = pd.DataFrame(summaries)

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
    aux = pd.DataFrame(aux)

    if len(summary):
        summary = summary.merge(aux, on="variant", how="right")
    else:
        summary = aux.copy()
        summary["source"] = "baseline"

    # Diagnostic score: prioritize exact high-precision memory, not huge movement.
    for c in ["exact_rate", "model_support_mean", "core_win_mean", "score_gain_mean", "diff_vs_anchor"]:
        if c not in summary.columns:
            summary[c] = np.nan
    diff = pd.to_numeric(summary["diff_vs_anchor"], errors="coerce").fillna(0)
    summary["v48_decision_score"] = (
        0.040 * pd.to_numeric(summary["exact_rate"], errors="coerce").fillna(0)
        + 0.018 * pd.to_numeric(summary["model_support_mean"], errors="coerce").fillna(0)
        + 0.008 * (pd.to_numeric(summary["core_win_mean"], errors="coerce").fillna(0) / 5.0)
        + 0.010 * np.tanh(pd.to_numeric(summary["score_gain_mean"], errors="coerce").fillna(0))
        + 0.010 * np.minimum(1, np.log1p(diff) / np.log1p(9000))
        - np.maximum(0, diff - 9000) / 400000
    )

    summary = summary.sort_values(["v48_decision_score", "diff_vs_anchor"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if reviews:
        rev = enrich_review(pd.concat(reviews, ignore_index=True).drop_duplicates(["id_add", "id_drop"], keep="first"))
        rev.to_csv(OUT_REVIEW, index=False)
    else:
        pd.DataFrame().to_csv(OUT_REVIEW, index=False)

    save = summary[
        (summary["variant"].astype(str).str.startswith("v48_memory"))
        & (summary["diff_vs_anchor"] >= 100)
        & (summary["diff_vs_anchor"] <= 7000)
    ].head(30)

    extra = ["anchor_v33_qprob2000", "raw_v35_b5000"]
    for e in extra:
        if e in variants and e not in set(save["variant"]):
            row = summary[summary["variant"].eq(e)]
            if len(row):
                save = pd.concat([save, row], ignore_index=True)

    saved_rows = []
    for i, row in save.head(35).reset_index(drop=True).iterrows():
        name = row["variant"]
        pred = variants[name]
        out = SUB_DIR / f"FINAL_CANDIDATE_v48_memory_{i+1:03d}_{short_hash(name)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)
        d = row.to_dict()
        d["file"] = str(out)
        saved_rows.append(d)
        print("saved", out, "<-", name)

    pd.DataFrame(saved_rows).to_csv(OUT_SAVED, index=False)

    print("\nTOP V48")
    cols = [
        "variant", "v48_decision_score", "source", "mode", "cap", "used_swaps", "accepted_pool",
        "exact_rate", "fuzzy_rate", "fuzzy_sim_mean", "train_item_pos_mean",
        "model_support_mean", "model_support_gain_mean", "core_win_mean", "score_gain_mean",
        "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v41_same", "diff_vs_v42_qgraph", "ones", "pos_ratio",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))

    print("\nSAVED")
    sdf = pd.DataFrame(saved_rows)
    if len(sdf):
        print(sdf[[c for c in ["variant", "file", "v48_decision_score", "diff_vs_anchor", "used_swaps", "exact_rate", "fuzzy_rate", "model_support_mean"] if c in sdf.columns]].to_string(index=False))

    print("\noutputs:")
    print(OUT_MEM)
    print(OUT_QMATCH)
    print(OUT_SUMMARY)
    print(OUT_SAVED)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
