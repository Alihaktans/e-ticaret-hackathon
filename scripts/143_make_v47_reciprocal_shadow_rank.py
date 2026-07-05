from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
ITEMS = ROOT / "data/raw/items.csv"
TERMS = ROOT / "data/raw/terms.csv"

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_SCORE = ROOT / "data/processed/v47_reciprocal_shadow_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v47_reciprocal_shadow_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v47_reciprocal_shadow_saved_candidates.csv"
OUT_REVIEW = OUT_DIR / "review_v47_reciprocal_shadow_swaps.csv"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

SCORE_SPECS = [
    (ROOT / "data/processed/v33_ft_pair_scores.parquet", "v33", ["v33_ft_score", "ft_score", "score"]),
    (ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet", "v34", ["v34_ce_score", "ce_score", "score"]),
    (ROOT / "data/processed/v38_lexical_pair_scores.parquet", "v38", ["v38_lex_score", "lex_score", "score"]),
    (ROOT / "data/processed/v39_item_history_pair_scores.parquet", "v39", ["v39_hist_score", "hist_score", "score"]),
    (ROOT / "data/processed/v41_learned_meta_prob.parquet", "v41", ["v41_meta_prob", "meta_prob", "prob"]),
    (ROOT / "data/processed/v42_query_graph_pair_scores.parquet", "v42", ["v42_graph_score", "v42_graph_norm_score", "graph_score", "score"]),
    (ROOT / "data/processed/v43_pop_brand_category_meta_prob.parquet", "v43", ["v43_meta_prob", "meta_prob", "prob"]),
    (ROOT / "data/processed/v43_pop_brand_category_pair_scores.parquet", "v43pop", ["v43_query_pop_support", "v43_pop_prior_score", "v43_pop_safe_score", "score"]),
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
    if numeric:
        return numeric[0]
    return None


def add_score(df, sample, path, out_col, candidates):
    if not path.exists():
        print("missing:", path)
        df[out_col] = np.float32(0)
        return df

    print("reading score:", path)
    s = pd.read_parquet(path)
    if "id" not in s.columns:
        print("  no id col, skip")
        df[out_col] = np.float32(0)
        return df

    col = pick_score_col(s, candidates)
    if col is None:
        print("  no score col, skip")
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

    print("  using", col, "as", out_col, "mean", float(df[out_col].mean()))
    return df


def group_pct_rank(df, group_col, value_col, out_col):
    r = df.groupby(group_col, sort=False)[value_col].rank(method="first", ascending=False)
    n = df.groupby(group_col, sort=False)[value_col].transform("size").astype(np.float32)
    pct = np.where(n > 1, 1.0 - ((r.astype(np.float32) - 1.0) / (n - 1.0)), 1.0)
    df[out_col] = pct.astype(np.float32)
    return df


def global_z(x):
    arr = pd.to_numeric(x, errors="coerce").fillna(0).astype(np.float32)
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < 1e-8:
        return np.zeros(len(arr), dtype=np.float32)
    return np.clip((arr.to_numpy(np.float32) - mu) / sd, -5, 5).astype(np.float32)


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


def load_text_meta():
    items = pd.read_csv(ITEMS)
    cols = list(items.columns)
    item_col = detect_col(cols, exacts=("item_id", "product_id", "id"), contains=("item_id",))
    title_col = detect_col(cols, exacts=("title", "name", "product_name", "item_name", "urun_adi", "ürün_adı"), contains=("title", "name", "urun", "ürün"))
    brand_col = detect_col(cols, exacts=("brand", "marka"), contains=("brand", "marka"))
    cat_col = detect_col(cols, exacts=("category", "kategori", "category_name", "leaf_category", "cat"), contains=("category", "kategori", "cat"))

    item_meta = pd.DataFrame({"item_id": items[item_col].astype(str)})
    item_meta["title"] = items[title_col].astype(str) if title_col else ""
    item_meta["brand"] = items[brand_col].astype(str) if brand_col else ""
    item_meta["category"] = items[cat_col].astype(str) if cat_col else ""

    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    if "query" not in terms.columns:
        cand = [c for c in terms.columns if c != "term_id" and terms[c].dtype == "object"]
        terms = terms.rename(columns={cand[0]: "query"})
    return item_meta, terms[["term_id", "query"]].copy()


def build_scores(sample, pairs):
    if OUT_SCORE.exists():
        print("using existing score file:", OUT_SCORE)
        d = pd.read_parquet(OUT_SCORE)
        d["id"] = d["id"].astype(str)
        if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            return d
        print("existing score file id order mismatch, rebuilding")

    df = pairs[["id", "term_id", "item_id"]].copy()

    for path, out_col, candidates in SCORE_SPECS:
        df = add_score(df, sample, path, out_col, candidates)

    raw_cols = ["v33", "v34", "v38", "v39", "v41", "v42", "v43", "v43pop"]
    for c in raw_cols:
        if c not in df.columns:
            df[c] = np.float32(0)
        df[c + "_z"] = global_z(df[c])

    # Forward score: query -> item desirability.
    df["v47_forward_base"] = (
        0.22 * df["v34_z"]
        + 0.18 * df["v33_z"]
        + 0.15 * df["v41_z"]
        + 0.13 * df["v38_z"]
        + 0.12 * df["v43_z"]
        + 0.08 * df["v42_z"]
        + 0.06 * df["v39_z"]
        + 0.06 * df["v43pop_z"]
    ).astype(np.float32)

    # Reverse score: item -> query ownership. Do not overuse popularity here.
    df["v47_reverse_base"] = (
        0.28 * df["v38_z"]
        + 0.24 * df["v34_z"]
        + 0.20 * df["v33_z"]
        + 0.14 * df["v41_z"]
        + 0.08 * df["v42_z"]
        + 0.06 * df["v39_z"]
    ).astype(np.float32)

    print("ranking forward by term...")
    df = group_pct_rank(df, "term_id", "v47_forward_base", "v47_forward_pct")
    df = group_pct_rank(df, "term_id", "v43", "v47_v43_forward_pct")
    df = group_pct_rank(df, "term_id", "v34", "v47_v34_forward_pct")
    df = group_pct_rank(df, "term_id", "v38", "v47_v38_forward_pct")

    print("ranking reverse by item...")
    item_count = df.groupby("item_id", sort=False)["id"].transform("size").astype(np.float32)
    r = df.groupby("item_id", sort=False)["v47_reverse_base"].rank(method="first", ascending=False).astype(np.float32)
    df["v47_item_query_count"] = item_count.astype(np.int32)
    df["v47_reverse_pct"] = np.where(item_count > 1, 1.0 - ((r - 1.0) / (item_count - 1.0)), 0.62).astype(np.float32)

    # Specific trap detector: forward high, reverse low = item belongs to some other query more.
    df["v47_shadow_conflict"] = (
        (df["v47_forward_pct"] >= 0.80)
        & (df["v47_item_query_count"] >= 3)
        & (df["v47_reverse_pct"] <= 0.35)
    ).astype(np.int8)

    df["v47_mutual_score"] = (
        0.56 * df["v47_forward_base"]
        + 0.34 * df["v47_reverse_pct"]
        + 0.16 * (df["v47_forward_pct"] * df["v47_reverse_pct"])
        + 0.05 * df["v47_v34_forward_pct"]
        + 0.04 * df["v47_v38_forward_pct"]
        - 0.22 * df["v47_shadow_conflict"].astype(np.float32)
    ).astype(np.float32)

    df["v47_mutual_strict_score"] = (
        df["v47_mutual_score"]
        + 0.18 * df["v47_reverse_pct"]
        - 0.30 * (df["v47_reverse_pct"] < 0.30).astype(np.float32)
        - 0.20 * df["v47_shadow_conflict"].astype(np.float32)
    ).astype(np.float32)

    keep_cols = [
        "id", "term_id", "item_id",
        "v33", "v34", "v38", "v39", "v41", "v42", "v43", "v43pop",
        "v47_forward_base", "v47_reverse_base", "v47_forward_pct", "v47_reverse_pct",
        "v47_item_query_count", "v47_shadow_conflict", "v47_mutual_score", "v47_mutual_strict_score",
        "v47_v43_forward_pct", "v47_v34_forward_pct", "v47_v38_forward_pct",
    ]
    OUT_SCORE.parent.mkdir(parents=True, exist_ok=True)
    df[keep_cols].to_parquet(OUT_SCORE, index=False)
    print("wrote", OUT_SCORE)
    return df[keep_cols]


def same_quota_pred(df, anchor, score_col):
    quota = pd.Series(anchor, index=df["term_id"]).groupby(level=0).sum().astype(int)
    ranks = df.groupby("term_id", sort=False)[score_col].rank(method="first", ascending=False).astype(np.int32)
    q = df["term_id"].map(quota).fillna(0).astype(np.int32)
    return (ranks <= q).astype(np.int8).to_numpy()


def pair_swaps(df, anchor, target, score_col):
    add_mask = (anchor == 0) & (target == 1)
    drop_mask = (anchor == 1) & (target == 0)

    cols = [
        "id", "term_id", "item_id", score_col,
        "v47_mutual_score", "v47_reverse_pct", "v47_forward_pct",
        "v47_item_query_count", "v47_shadow_conflict",
        "v33", "v34", "v38", "v39", "v41", "v42", "v43", "v43pop",
    ]
    cols = list(dict.fromkeys(cols))  # score_col duplicate olmasin

    add = df.loc[add_mask, cols].copy().rename(columns={"id": "id_add", "item_id": "item_id_add"})
    drop = df.loc[drop_mask, cols].copy().rename(columns={"id": "id_drop", "item_id": "item_id_drop"})
    add = add.loc[:, ~add.columns.duplicated()].copy()
    drop = drop.loc[:, ~drop.columns.duplicated()].copy()

    add = add.sort_values(["term_id", score_col], ascending=[True, False])
    drop = drop.sort_values(["term_id", score_col], ascending=[True, True])

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    sw = add.merge(drop, on=["term_id", "pair_rank"], suffixes=("_add", "_drop"), how="inner")

    sw["score_gain"] = sw[f"{score_col}_add"] - sw[f"{score_col}_drop"]
    sw["mutual_gain"] = sw["v47_mutual_score_add"] - sw["v47_mutual_score_drop"]
    sw["reverse_gain"] = sw["v47_reverse_pct_add"] - sw["v47_reverse_pct_drop"]
    sw["forward_gain"] = sw["v47_forward_pct_add"] - sw["v47_forward_pct_drop"]

    signal_cols = ["v33", "v34", "v38", "v39", "v41", "v42", "v43", "v43pop"]
    sw["signal_win_count"] = sum((sw[f"{c}_add"] > sw[f"{c}_drop"]).astype(int) for c in signal_cols)
    sw["core_win_count"] = (
        (sw["v33_add"] > sw["v33_drop"]).astype(int)
        + (sw["v34_add"] > sw["v34_drop"]).astype(int)
        + (sw["v38_add"] > sw["v38_drop"]).astype(int)
        + (sw["v41_add"] > sw["v41_drop"]).astype(int)
    )

    sw["v47_swap_score"] = (
        0.50 * sw["mutual_gain"]
        + 0.25 * sw["reverse_gain"]
        + 0.18 * sw["forward_gain"]
        + 0.06 * sw["signal_win_count"]
        - 0.35 * sw["v47_shadow_conflict_add"].astype(float)
    )

    return sw.sort_values("v47_swap_score", ascending=False).reset_index(drop=True)


def filter_swaps(sw, mode):
    if mode == "safe":
        return (
            (sw["score_gain"] > 0.18)
            & (sw["mutual_gain"] > 0.16)
            & (sw["v47_reverse_pct_add"] >= 0.55)
            & (sw["v47_shadow_conflict_add"] == 0)
            & (sw["signal_win_count"] >= 5)
            & (sw["core_win_count"] >= 3)
        )
    if mode == "balanced":
        return (
            (sw["score_gain"] > 0.10)
            & (sw["mutual_gain"] > 0.08)
            & (sw["v47_reverse_pct_add"] >= 0.42)
            & (sw["v47_shadow_conflict_add"] == 0)
            & (sw["signal_win_count"] >= 4)
            & (sw["core_win_count"] >= 2)
        )
    if mode == "impact":
        return (
            (sw["score_gain"] > 0.04)
            & (sw["mutual_gain"] > 0.03)
            & (sw["v47_reverse_pct_add"] >= 0.30)
            & (sw["v47_shadow_conflict_add"] == 0)
            & (sw["signal_win_count"] >= 4)
        )
    if mode == "shadow_only":
        return (
            (sw["v47_reverse_pct_add"] >= 0.70)
            & (sw["reverse_gain"] > 0.20)
            & (sw["v47_shadow_conflict_add"] == 0)
            & (sw["score_gain"] > 0.00)
        )
    raise ValueError(mode)


def apply_swaps(sample, anchor, sw, cap):
    take = sw.sort_values("v47_swap_score", ascending=False).head(cap).copy()
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
    item_meta, terms = load_text_meta()

    add = item_meta.rename(columns={"item_id": "item_id_add", "title": "add_title", "brand": "add_brand", "category": "add_category"})
    drop = item_meta.rename(columns={"item_id": "item_id_drop", "title": "drop_title", "brand": "drop_brand", "category": "drop_category"})

    out = sw.merge(terms, on="term_id", how="left")
    out = out.merge(add, on="item_id_add", how="left")
    out = out.merge(drop, on="item_id_drop", how="left")

    front = [
        "variant_source", "mode", "term_id", "query",
        "id_add", "item_id_add", "add_title", "add_brand", "add_category",
        "id_drop", "item_id_drop", "drop_title", "drop_brand", "drop_category",
        "v47_swap_score", "score_gain", "mutual_gain", "reverse_gain", "forward_gain",
        "v47_reverse_pct_add", "v47_reverse_pct_drop", "v47_forward_pct_add", "v47_forward_pct_drop",
        "v47_item_query_count_add", "v47_shadow_conflict_add",
        "signal_win_count", "core_win_count",
        "v34_add", "v34_drop", "v38_add", "v38_drop", "v43_add", "v43_drop",
    ]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print("loading sample/pairs...")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not sample["id"].reset_index(drop=True).equals(pairs["id"].reset_index(drop=True)):
        raise RuntimeError("sample/pairs id order mismatch")

    anchor_path = first_existing(ANCHOR_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor submission not found")
    anchor = load_pred(anchor_path, sample)
    print("anchor ones", int(anchor.sum()), "pos", float(anchor.mean()))

    df = build_scores(sample, pairs)

    variants = {}
    summaries = []
    reviews = []

    # Baselines.
    variants["anchor_v33_qprob2000"] = anchor.copy()
    for name, paths in BASELINE_CSV_CANDIDATES.items():
        p = first_existing(paths)
        if p is not None:
            variants[name] = load_pred(p, sample)
            print("baseline", name, "diff", int((variants[name] != anchor).sum()))

    # Same-quota mutual rank outputs.
    same_specs = [
        ("v47_same_quota_mutual", "v47_mutual_score"),
        ("v47_same_quota_shadow_strict", "v47_mutual_strict_score"),
    ]
    for name, sc in same_specs:
        pred = same_quota_pred(df, anchor, sc)
        variants[name] = pred
        summaries.append({
            "variant": name, "source": "same_quota", "score_col": sc, "mode": "",
            "cap": "", "used_swaps": int((pred != anchor).sum() // 2), "accepted_pool": "",
            "score_gain_mean": "", "mutual_gain_mean": "", "reverse_gain_mean": "",
            "reverse_add_mean": "", "signal_win_mean": "", "core_win_mean": "",
        })

    # Swap filtered outputs from each same-quota target.
    caps = [700, 1200, 1800, 2600, 3800, 5500]
    modes = ["safe", "balanced", "impact", "shadow_only"]

    for base_name, score_col in same_specs:
        target = variants[base_name]
        sw0 = pair_swaps(df, anchor, target, score_col)
        sw0["variant_source"] = base_name
        print(base_name, "raw swaps", len(sw0))

        for mode in modes:
            acc = sw0[filter_swaps(sw0, mode)].copy()
            print(" ", mode, "accepted", len(acc))
            if len(acc) == 0:
                continue

            acc["mode"] = mode
            reviews.append(acc.head(350))

            for cap in caps:
                used = min(cap, len(acc))
                pred, take = apply_swaps(sample, anchor, acc, used)
                vname = f"v47_{mode}_{short_hash(base_name)}_cap{cap}"
                variants[vname] = pred

                summaries.append({
                    "variant": vname,
                    "source": "swap_filter",
                    "score_col": score_col,
                    "base": base_name,
                    "mode": mode,
                    "cap": cap,
                    "used_swaps": int(len(take)),
                    "accepted_pool": int(len(acc)),
                    "score_gain_mean": float(take["score_gain"].mean()),
                    "mutual_gain_mean": float(take["mutual_gain"].mean()),
                    "reverse_gain_mean": float(take["reverse_gain"].mean()),
                    "reverse_add_mean": float(take["v47_reverse_pct_add"].mean()),
                    "forward_add_mean": float(take["v47_forward_pct_add"].mean()),
                    "signal_win_mean": float(take["signal_win_count"].mean()),
                    "core_win_mean": float(take["core_win_count"].mean()),
                    "shadow_conflict_add_rate": float(take["v47_shadow_conflict_add"].mean()),
                })

    # Diff diagnostics.
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
    summary = summary.merge(aux, on="variant", how="right")

    # Decision score is diagnostic, not a real validation score.
    diff = summary["diff_vs_anchor"].fillna(0).astype(float)
    reverse = pd.to_numeric(summary.get("reverse_add_mean", 0), errors="coerce").fillna(0)
    sig = pd.to_numeric(summary.get("signal_win_mean", 0), errors="coerce").fillna(0)
    gain = pd.to_numeric(summary.get("mutual_gain_mean", 0), errors="coerce").fillna(0)

    summary["v47_decision_score"] = (
        0.020 * np.minimum(1, np.log1p(diff) / np.log1p(14000))
        + 0.012 * reverse
        + 0.004 * np.minimum(1, sig / 6)
        + 0.006 * np.tanh(gain)
        - np.maximum(0, diff - 22000) / 700000
    )

    summary = summary.sort_values(["v47_decision_score", "diff_vs_anchor"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if reviews:
        rev = enrich_review(pd.concat(reviews, ignore_index=True))
        rev.to_csv(OUT_REVIEW, index=False)
    else:
        pd.DataFrame().to_csv(OUT_REVIEW, index=False)

    # Save candidates. Avoid pure same-quota at top unless user wants high risk.
    save = summary[
        (summary["source"].eq("swap_filter"))
        & (summary["diff_vs_anchor"] >= 1000)
        & (summary["diff_vs_anchor"] <= 14000)
    ].head(40)

    # Add same quota and anchors for comparison.
    extra = ["v47_same_quota_shadow_strict", "v47_same_quota_mutual", "anchor_v33_qprob2000", "raw_v35_b5000"]
    for e in extra:
        if e in variants and e not in set(save["variant"]):
            row = summary[summary["variant"].eq(e)]
            if len(row):
                save = pd.concat([save, row], ignore_index=True)

    saved_rows = []
    for i, row in save.head(45).reset_index(drop=True).iterrows():
        name = row["variant"]
        pred = variants[name]
        out = SUB_DIR / f"FINAL_CANDIDATE_v47_shadow_{i+1:03d}_{short_hash(name)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)
        d = row.to_dict()
        d["file"] = str(out)
        saved_rows.append(d)
        print("saved", out, "<-", name)

    pd.DataFrame(saved_rows).to_csv(OUT_SAVED, index=False)

    print("\nTOP V47")
    cols = [
        "variant", "v47_decision_score", "source", "base", "mode", "cap", "used_swaps", "accepted_pool",
        "score_gain_mean", "mutual_gain_mean", "reverse_gain_mean", "reverse_add_mean",
        "forward_add_mean", "signal_win_mean", "core_win_mean", "shadow_conflict_add_rate",
        "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v41_same", "diff_vs_v42_qgraph", "ones", "pos_ratio",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(90).to_string(index=False))

    print("\nSAVED")
    sdf = pd.DataFrame(saved_rows)
    if len(sdf):
        print(sdf[[c for c in ["variant", "file", "v47_decision_score", "diff_vs_anchor", "used_swaps", "reverse_add_mean", "signal_win_mean"] if c in sdf.columns]].to_string(index=False))

    print("\noutputs:")
    print(OUT_SCORE)
    print(OUT_SUMMARY)
    print(OUT_SAVED)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
