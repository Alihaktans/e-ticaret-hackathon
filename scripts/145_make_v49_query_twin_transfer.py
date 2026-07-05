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
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_QPAIRS = OUT_DIR / "v49_query_twin_pairs.csv"
OUT_REVIEW = OUT_DIR / "review_v49_query_twin_transfer_swaps.csv"
OUT_SUMMARY = OUT_DIR / "v49_query_twin_transfer_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v49_query_twin_transfer_saved_candidates.csv"
OUT_FEATURES = ROOT / "data/processed/v49_query_twin_transfer_features.parquet"

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


def load_terms():
    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    if "query" not in terms.columns:
        cand = [c for c in terms.columns if c != "term_id" and terms[c].dtype == "object"]
        if not cand:
            raise RuntimeError("query column not found")
        terms = terms.rename(columns={cand[0]: "query"})
    terms["query"] = terms["query"].astype(str)
    terms["query_norm"] = terms["query"].map(norm_text)
    return terms[["term_id", "query", "query_norm"]].copy()


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


def pick_score_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def add_score(df, sample, path, out_col, candidates):
    if not path.exists():
        print("missing score:", path)
        df[out_col] = np.float32(0)
        return df, False

    s = pd.read_parquet(path)
    if "id" not in s.columns:
        df[out_col] = np.float32(0)
        return df, False

    col = pick_score_col(s, candidates)
    if col is None:
        df[out_col] = np.float32(0)
        return df, False

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

    ok = float(df[out_col].std()) > 1e-8
    print("score", out_col, "from", path, "col", col, "mean", float(df[out_col].mean()), "ok", ok)
    return df, ok


def query_pct_rank(df, group_col, val_col, out_col, available=True):
    if not available or float(df[val_col].std()) < 1e-8:
        df[out_col] = np.float32(0.5)
        return df
    r = df.groupby(group_col, sort=False)[val_col].rank(method="first", ascending=False).astype(np.float32)
    n = df.groupby(group_col, sort=False)[val_col].transform("size").astype(np.float32)
    df[out_col] = np.where(n > 1, 1.0 - ((r - 1.0) / (n - 1.0)), 1.0).astype(np.float32)
    return df


def build_features(sample, pairs, anchor):
    if OUT_FEATURES.exists():
        print("using existing features:", OUT_FEATURES)
        d = pd.read_parquet(OUT_FEATURES)
        d["id"] = d["id"].astype(str)
        if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            return d
        print("existing feature id order mismatch, rebuilding")

    terms = load_terms()
    df = pairs.merge(terms, on="term_id", how="left")
    df["anchor"] = anchor.astype(np.int8)

    avail = {}
    for path, name, candidates in SCORE_SPECS:
        df, ok = add_score(df, sample, path, name, candidates)
        avail[name] = ok

    for c in ["v33", "v34", "v38", "v41", "v42", "v43"]:
        if c not in df.columns:
            df[c] = np.float32(0)
        df = query_pct_rank(df, "term_id", c, c + "_pct", avail.get(c, False))

    df["support"] = (
        0.28 * df["v34_pct"]
        + 0.25 * df["v33_pct"]
        + 0.18 * df["v41_pct"]
        + 0.15 * df["v38_pct"]
        + 0.09 * df["v43_pct"]
        + 0.05 * df["v42_pct"]
    ).astype(np.float32)

    keep = [
        "id", "term_id", "item_id", "query", "query_norm", "anchor",
        "support", "v33", "v34", "v38", "v41", "v42", "v43",
        "v33_pct", "v34_pct", "v38_pct", "v41_pct", "v42_pct", "v43_pct",
    ]
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_parquet(OUT_FEATURES, index=False)
    print("wrote", OUT_FEATURES)
    return df[keep]


def build_query_twin_pairs(terms, min_sim=0.88, topn=6):
    if OUT_QPAIRS.exists():
        q = pd.read_csv(OUT_QPAIRS)
        q["src_term_id"] = q["src_term_id"].astype(str)
        q["dst_term_id"] = q["dst_term_id"].astype(str)
        return q

    print("building query twin pairs...")
    t = terms.drop_duplicates("term_id").copy()
    t = t[t["query_norm"].astype(str).str.len() >= 3].reset_index(drop=True)

    texts = t["query_norm"].tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, dtype=np.float32)
    X = vectorizer.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=min(topn + 1, len(t)), metric="cosine", algorithm="brute")
    nn.fit(X)
    dist, ind = nn.kneighbors(X, return_distance=True)

    rows = []
    term_ids = t["term_id"].tolist()
    queries = t["query"].tolist()
    norms = t["query_norm"].tolist()

    for i, src in enumerate(term_ids):
        for d, j in zip(dist[i], ind[i]):
            j = int(j)
            if j == i:
                continue
            sim = 1.0 - float(d)
            if sim < min_sim:
                continue
            rows.append({
                "src_term_id": src,
                "dst_term_id": term_ids[j],
                "query_sim": sim,
                "src_query": queries[i],
                "dst_query": queries[j],
                "src_query_norm": norms[i],
                "dst_query_norm": norms[j],
            })

    q = pd.DataFrame(rows)
    q = q[q["src_term_id"] != q["dst_term_id"]].copy()
    q = q.sort_values(["query_sim", "src_term_id", "dst_term_id"], ascending=[False, True, True])
    q.to_csv(OUT_QPAIRS, index=False)
    print("wrote", OUT_QPAIRS, "rows", len(q))
    return q


def build_add_candidates(df, qpairs):
    print("building add candidates from query twins...")
    pos = df[df["anchor"].eq(1)][[
        "term_id", "item_id", "id", "support", "v33", "v34", "v38", "v41", "v42", "v43",
    ]].rename(columns={
        "term_id": "src_term_id",
        "id": "src_id",
        "support": "src_support",
        "v33": "src_v33",
        "v34": "src_v34",
        "v38": "src_v38",
        "v41": "src_v41",
        "v42": "src_v42",
        "v43": "src_v43",
    })

    dst = df[[
        "term_id", "item_id", "id", "anchor", "support", "v33", "v34", "v38", "v41", "v42", "v43",
    ]].rename(columns={
        "term_id": "dst_term_id",
        "id": "id_add",
        "anchor": "add_anchor",
        "support": "add_support",
        "v33": "add_v33",
        "v34": "add_v34",
        "v38": "add_v38",
        "v41": "add_v41",
        "v42": "add_v42",
        "v43": "add_v43",
    })

    # Keep qpairs tight enough before expanding.
    qpairs = qpairs[qpairs["query_sim"] >= 0.88].copy()
    print(" qpairs:", len(qpairs), "source positives:", len(pos))

    m = qpairs.merge(pos, on="src_term_id", how="inner")
    print(" after qpair x src_pos:", len(m))
    m = m.merge(dst, on=["dst_term_id", "item_id"], how="inner")
    print(" after dst candidate join:", len(m))

    m = m[m["add_anchor"].eq(0)].copy()
    if len(m) == 0:
        return m

    # Item has to be supported in destination too. If source is exact duplicate query, allow lower destination support.
    m["same_query_norm"] = (m["src_query_norm"].astype(str) == m["dst_query_norm"].astype(str)).astype(np.int8)

    m["core_win_count_src_vs_add"] = (
        (m["add_v33"] >= m["src_v33"] * 0.70).astype(int)
        + (m["add_v34"] >= m["src_v34"] * 0.70).astype(int)
        + (m["add_v38"] >= m["src_v38"] * 0.70).astype(int)
        + (m["add_v41"] >= m["src_v41"] * 0.70).astype(int)
        + (m["add_v43"] >= m["src_v43"] * 0.70).astype(int)
    )

    m["transfer_score"] = (
        1.20 * m["query_sim"].astype(float)
        + 0.55 * m["src_support"].astype(float)
        + 0.70 * m["add_support"].astype(float)
        + 0.25 * m["same_query_norm"].astype(float)
        + 0.05 * m["core_win_count_src_vs_add"].astype(float)
    )

    # One add id may be supported by multiple twin queries; keep strongest explanation.
    m = m.sort_values(["transfer_score", "query_sim", "add_support"], ascending=False)
    m = m.drop_duplicates("id_add", keep="first").reset_index(drop=True)
    print(" unique add candidates:", len(m))
    return m


def make_swaps(df, add_cands, mode):
    if len(add_cands) == 0:
        return pd.DataFrame()

    # Mode filters before pairing.
    if mode == "dupe_safe":
        a = add_cands[
            (add_cands["query_sim"] >= 0.985)
            & (add_cands["add_support"] >= 0.55)
            & (add_cands["src_support"] >= 0.55)
        ].copy()
        min_gain = 0.08
        min_core = 3
    elif mode == "safe":
        a = add_cands[
            (add_cands["query_sim"] >= 0.94)
            & (add_cands["add_support"] >= 0.60)
            & (add_cands["src_support"] >= 0.60)
            & (add_cands["core_win_count_src_vs_add"] >= 2)
        ].copy()
        min_gain = 0.10
        min_core = 3
    elif mode == "balanced":
        a = add_cands[
            (add_cands["query_sim"] >= 0.91)
            & (add_cands["add_support"] >= 0.52)
            & (add_cands["src_support"] >= 0.55)
            & (add_cands["core_win_count_src_vs_add"] >= 2)
        ].copy()
        min_gain = 0.06
        min_core = 2
    elif mode == "impact":
        a = add_cands[
            (add_cands["query_sim"] >= 0.88)
            & (add_cands["add_support"] >= 0.45)
            & (add_cands["src_support"] >= 0.50)
            & (add_cands["core_win_count_src_vs_add"] >= 1)
        ].copy()
        min_gain = 0.03
        min_core = 2
    else:
        raise ValueError(mode)

    if len(a) == 0:
        return pd.DataFrame()

    drops = df[df["anchor"].eq(1)][[
        "term_id", "id", "item_id", "support", "v33", "v34", "v38", "v41", "v42", "v43",
    ]].rename(columns={
        "term_id": "dst_term_id",
        "id": "id_drop",
        "item_id": "item_id_drop",
        "support": "drop_support",
        "v33": "drop_v33",
        "v34": "drop_v34",
        "v38": "drop_v38",
        "v41": "drop_v41",
        "v42": "drop_v42",
        "v43": "drop_v43",
    })

    a = a.sort_values(["dst_term_id", "transfer_score"], ascending=[True, False]).copy()
    d = drops.sort_values(["dst_term_id", "drop_support"], ascending=[True, True]).copy()

    a["pair_rank"] = a.groupby("dst_term_id").cumcount()
    d["pair_rank"] = d.groupby("dst_term_id").cumcount()

    sw = a.merge(d, on=["dst_term_id", "pair_rank"], how="inner")
    if len(sw) == 0:
        return sw

    sw["support_gain"] = sw["add_support"] - sw["drop_support"]
    sw["model_win_count"] = (
        (sw["add_v33"] > sw["drop_v33"]).astype(int)
        + (sw["add_v34"] > sw["drop_v34"]).astype(int)
        + (sw["add_v38"] > sw["drop_v38"]).astype(int)
        + (sw["add_v41"] > sw["drop_v41"]).astype(int)
        + (sw["add_v43"] > sw["drop_v43"]).astype(int)
    )

    sw["v49_swap_score"] = (
        1.00 * sw["transfer_score"]
        + 1.40 * sw["support_gain"]
        + 0.20 * sw["model_win_count"]
        + 0.18 * sw["same_query_norm"]
        + 0.10 * sw["query_sim"]
    )

    sw = sw[
        (sw["support_gain"] >= min_gain)
        & (sw["model_win_count"] >= min_core)
    ].copy()

    sw["mode"] = mode
    return sw.sort_values("v49_swap_score", ascending=False).reset_index(drop=True)


def apply_swaps(sample, anchor, sw, cap):
    take = sw.sort_values("v49_swap_score", ascending=False).head(cap).copy()
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
    item_meta = load_item_meta()
    terms = load_terms()[["term_id", "query"]].rename(columns={"term_id": "dst_term_id", "query": "dst_query_real"})

    add = item_meta.rename(columns={
        "item_id": "item_id",
        "title": "add_title",
        "brand": "add_brand",
        "category": "add_category",
    })

    drop = item_meta.rename(columns={
        "item_id": "item_id_drop",
        "title": "drop_title",
        "brand": "drop_brand",
        "category": "drop_category",
    })

    out = sw.merge(terms, on="dst_term_id", how="left")
    out = out.merge(add, on="item_id", how="left")
    out = out.merge(drop, on="item_id_drop", how="left")

    front = [
        "variant", "mode", "dst_term_id", "dst_query_real", "src_query", "dst_query", "query_sim",
        "id_add", "item_id", "add_title", "add_brand", "add_category",
        "id_drop", "item_id_drop", "drop_title", "drop_brand", "drop_category",
        "v49_swap_score", "transfer_score", "support_gain", "add_support", "drop_support",
        "src_support", "same_query_norm", "model_win_count", "core_win_count_src_vs_add",
    ]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

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
    print("anchor ones", int(anchor.sum()), "pos", float(anchor.mean()))

    terms = load_terms()
    df = build_features(sample, pairs, anchor)
    qpairs = build_query_twin_pairs(terms, min_sim=0.88, topn=6)
    print("query twin pairs:", len(qpairs), "sim mean", float(qpairs["query_sim"].mean()) if len(qpairs) else None)

    add_cands = build_add_candidates(df, qpairs)

    variants = {"anchor_v33_qprob2000": anchor.copy()}
    for name, paths in BASELINE_CSV_CANDIDATES.items():
        p = first_existing(paths)
        if p is not None:
            variants[name] = load_pred(p, sample)
            print("baseline", name, "diff", int((variants[name] != anchor).sum()))

    caps = [100, 250, 500, 900, 1400, 2200, 3500]
    modes = ["dupe_safe", "safe", "balanced", "impact"]

    summaries = []
    reviews = []

    for mode in modes:
        sw0 = make_swaps(df, add_cands, mode)
        print(mode, "accepted swaps", len(sw0))
        if len(sw0):
            reviews.append(sw0.head(400))

        for cap in caps:
            if len(sw0) == 0:
                continue
            used = min(cap, len(sw0))
            pred, take = apply_swaps(sample, anchor, sw0, used)
            vname = f"v49_twin_{mode}_cap{cap}"
            variants[vname] = pred
            take = take.copy()
            take["variant"] = vname

            summaries.append({
                "variant": vname,
                "source": "query_twin_transfer",
                "mode": mode,
                "cap": cap,
                "used_swaps": int(len(take)),
                "accepted_pool": int(len(sw0)),
                "query_sim_mean": float(take["query_sim"].mean()),
                "same_query_rate": float(take["same_query_norm"].mean()),
                "transfer_score_mean": float(take["transfer_score"].mean()),
                "support_gain_mean": float(take["support_gain"].mean()),
                "add_support_mean": float(take["add_support"].mean()),
                "drop_support_mean": float(take["drop_support"].mean()),
                "src_support_mean": float(take["src_support"].mean()),
                "model_win_mean": float(take["model_win_count"].mean()),
                "core_src_add_mean": float(take["core_win_count_src_vs_add"].mean()),
                "v49_swap_score_mean": float(take["v49_swap_score"].mean()),
            })

    if reviews:
        rev = pd.concat(reviews, ignore_index=True)
        rev["variant"] = ""
        rev = rev.drop_duplicates(["id_add", "id_drop"], keep="first")
        rev = enrich_review(rev)
        rev.to_csv(OUT_REVIEW, index=False)
    else:
        pd.DataFrame().to_csv(OUT_REVIEW, index=False)

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

    for c in [
        "query_sim_mean", "same_query_rate", "support_gain_mean", "add_support_mean",
        "model_win_mean", "v49_swap_score_mean", "diff_vs_anchor",
    ]:
        if c not in summary.columns:
            summary[c] = np.nan

    diff = pd.to_numeric(summary["diff_vs_anchor"], errors="coerce").fillna(0)
    summary["v49_decision_score"] = (
        0.018 * pd.to_numeric(summary["query_sim_mean"], errors="coerce").fillna(0)
        + 0.010 * pd.to_numeric(summary["same_query_rate"], errors="coerce").fillna(0)
        + 0.013 * pd.to_numeric(summary["support_gain_mean"], errors="coerce").fillna(0)
        + 0.006 * (pd.to_numeric(summary["model_win_mean"], errors="coerce").fillna(0) / 5.0)
        + 0.010 * np.minimum(1, np.log1p(diff) / np.log1p(7000))
        - np.maximum(0, diff - 9000) / 500000
    )

    summary = summary.sort_values(["v49_decision_score", "diff_vs_anchor"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    save = summary[
        (summary["variant"].astype(str).str.startswith("v49_twin"))
        & (summary["diff_vs_anchor"] >= 100)
        & (summary["diff_vs_anchor"] <= 9000)
    ].head(35)

    for e in ["anchor_v33_qprob2000", "raw_v35_b5000"]:
        if e in variants and e not in set(save["variant"]):
            row = summary[summary["variant"].eq(e)]
            if len(row):
                save = pd.concat([save, row], ignore_index=True)

    saved_rows = []
    for i, row in save.head(40).reset_index(drop=True).iterrows():
        name = row["variant"]
        pred = variants[name]
        out = SUB_DIR / f"FINAL_CANDIDATE_v49_twin_{i+1:03d}_{short_hash(name)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)
        d = row.to_dict()
        d["file"] = str(out)
        saved_rows.append(d)
        print("saved", out, "<-", name)

    pd.DataFrame(saved_rows).to_csv(OUT_SAVED, index=False)

    print("\nTOP V49")
    cols = [
        "variant", "v49_decision_score", "source", "mode", "cap", "used_swaps", "accepted_pool",
        "query_sim_mean", "same_query_rate", "support_gain_mean", "add_support_mean", "drop_support_mean",
        "model_win_mean", "core_src_add_mean", "v49_swap_score_mean",
        "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v41_same", "diff_vs_v42_qgraph", "ones", "pos_ratio",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))

    print("\nSAVED")
    sdf = pd.DataFrame(saved_rows)
    if len(sdf):
        print(sdf[[c for c in ["variant", "file", "v49_decision_score", "diff_vs_anchor", "used_swaps", "query_sim_mean", "same_query_rate", "support_gain_mean"] if c in sdf.columns]].to_string(index=False))

    print("\noutputs:")
    print(OUT_QPAIRS)
    print(OUT_REVIEW)
    print(OUT_SUMMARY)
    print(OUT_SAVED)
    print(OUT_FEATURES)


if __name__ == "__main__":
    main()
