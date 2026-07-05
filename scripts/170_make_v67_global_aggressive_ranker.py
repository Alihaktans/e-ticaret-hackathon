from pathlib import Path
import os
import re
import hashlib
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
ITEMS = ROOT / "data/raw/items.csv"
TERMS = ROOT / "data/raw/terms.csv"

# Use the strongest public-0.80 style file as anchor only for patch variants.
BASE_CANDIDATES = [
    ROOT / "submissions/final_candidates_v66/FINAL_CANDIDATE_v66_intent_ultra_swap_cap75_699710c5.csv",
    ROOT / "submissions/final_candidates_v66/FINAL_CANDIDATE_v66_intent_ultra_swap_cap150_5724fd6c.csv",
    ROOT / "submissions/final_candidates_v60/v60_public080_ultra_cap250.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    ROOT / "submissions/final_candidates_v40/A_v40_raw35_history_impact2200.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
]

SCORE_FILES = [
    ROOT / "data/processed/v33_ft_pair_scores.parquet",
    ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet",
    ROOT / "data/processed/v38_lexical_pair_scores.parquet",
    ROOT / "data/processed/v39_item_history_pair_scores.parquet",
    ROOT / "data/processed/v41_learned_meta_prob.parquet",
    ROOT / "data/processed/v42_query_graph_pair_scores.parquet",
    ROOT / "data/processed/v43_pop_brand_category_pair_scores.parquet",
    ROOT / "data/processed/v43_pop_brand_category_meta_prob.parquet",
    ROOT / "data/processed/v47_reciprocal_shadow_scores.parquet",
    ROOT / "data/processed/v48_train_memory_bridge_scores.parquet",
    ROOT / "data/processed/v49_query_twin_transfer_features.parquet",
]

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v67"

OUT_SCORE = ROOT / "data/processed/v67_global_aggressive_blend_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v67_global_aggressive_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v67_global_aggressive_saved_candidates.csv"
OUT_AUDIT = OUT_DIR / "review_v67_global_aggressive_top_changes.csv"

# Very aggressive. These are intentionally much larger than V66.
SWAP_CAPS = [500, 1000, 2000, 5000, 10000, 20000, 40000, 70000, 100000]
SHIFT_CAPS = [500, 1000, 2000, 5000, 10000, 20000, 40000, 70000]
GLOBAL_K_DELTAS = [-120000, -80000, -40000, -20000, -10000, -5000, 0, 5000, 10000, 20000, 40000, 80000, 120000, 180000]

BAD_COL_PAT = re.compile(r"(label|target|prediction|pred|fold|split|row|index|rank$|rank_|_rank|pctile)", re.I)
GOOD_COL_PAT = re.compile(r"(prob|score|sim|cos|ce|bge|ft|lex|hist|history|support|mutual|meta|graph|pop|root|bridge|twin|brand|cat|category|overlap|match|coverage)", re.I)
KEYS = {"id", "term_id", "item_id"}


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


def find_base():
    for p in BASE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No base candidate found. Checked:\n" + "\n".join(map(str, BASE_CANDIDATES)))


def dtype_is_numeric(dt):
    return dt in {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }


def choose_score_cols(path, schema, max_cols=6):
    candidates = []
    for c, dt in schema.items():
        lc = c.lower()
        if lc in KEYS:
            continue
        if BAD_COL_PAT.search(c):
            continue
        if not dtype_is_numeric(dt):
            continue
        if not GOOD_COL_PAT.search(c):
            continue
        priority = 0
        if re.search(r"(meta.*prob|prob)", c, re.I):
            priority += 60
        if re.search(r"(ce|cross|bge|ft|bert|semantic)", c, re.I):
            priority += 45
        if re.search(r"(lex|overlap|match|coverage)", c, re.I):
            priority += 28
        if re.search(r"(hist|history|support|mutual|graph|root|bridge|twin)", c, re.I):
            priority += 24
        if re.search(r"(score|sim)", c, re.I):
            priority += 12
        # Prefer compact columns, not huge auxiliary sets.
        priority -= max(0, len(c) - 35) * 0.2
        candidates.append((priority, c))

    candidates = sorted(candidates, reverse=True)
    selected = [c for _, c in candidates[:max_cols]]
    return selected


def prefix_for_path(path):
    s = str(path).replace("\\", "/")
    m = re.search(r"(v\d+)", s)
    pref = m.group(1) if m else path.stem[:12]
    pref = re.sub(r"[^a-zA-Z0-9]+", "_", pref).strip("_")
    return pref


def read_score_file(path):
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    score_cols = choose_score_cols(path, schema)
    if not score_cols:
        print("SKIP no score cols:", path)
        return None, [], []

    has_id = "id" in schema
    has_pair = ("term_id" in schema and "item_id" in schema)
    if not has_id and not has_pair:
        print("SKIP no join keys:", path)
        return None, [], []

    pref = prefix_for_path(path)
    join_cols = ["id"] if has_id else ["term_id", "item_id"]
    select_cols = join_cols + score_cols

    lf = lf.select(select_cols)
    if has_id:
        lf = lf.with_columns(pl.col("id").cast(pl.Utf8))
    else:
        lf = lf.with_columns([
            pl.col("term_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
        ])

    renames = {}
    out_cols = []
    for c in score_cols:
        nc = f"{pref}_{c}"
        nc = re.sub(r"[^a-zA-Z0-9_]+", "_", nc)
        renames[c] = nc
        out_cols.append(nc)

    lf = lf.rename(renames).unique(subset=join_cols, keep="first")
    print("USE", path, "join", join_cols, "cols", out_cols)
    return lf, join_cols, out_cols


def feature_weight(col):
    lc = col.lower()
    w = 1.0
    if "meta" in lc or "prob" in lc:
        w += 1.25
    if any(x in lc for x in ["ce", "cross", "bge", "ft", "bert", "semantic"]):
        w += 0.95
    if any(x in lc for x in ["lex", "overlap", "match", "coverage"]):
        w += 0.45
    if any(x in lc for x in ["hist", "history", "support", "mutual", "graph", "root", "bridge", "twin"]):
        w += 0.40
    if "pop" in lc:
        w -= 0.15
    return max(0.25, w)


def build_blend_scores(force_rebuild=False):
    if OUT_SCORE.exists() and not force_rebuild:
        print("Using existing", OUT_SCORE)
        return pl.read_parquet(OUT_SCORE)

    if not SAMPLE.exists() or not PAIRS.exists():
        raise FileNotFoundError("sample_submission.csv or submission_pairs.csv missing")

    print("Reading sample+pairs")
    sample = pl.read_csv(SAMPLE).select(pl.col("id").cast(pl.Utf8))
    pairs = pl.read_csv(PAIRS).select([
        pl.col("id").cast(pl.Utf8),
        pl.col("term_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
    ])
    df = sample.join(pairs, on="id", how="left")

    all_score_cols = []
    for path in SCORE_FILES:
        if not path.exists():
            print("MISSING", path)
            continue

        lf, join_cols, out_cols = read_score_file(path)
        if lf is None:
            continue

        score_df = lf.collect(streaming=True)
        if join_cols == ["id"]:
            df = df.join(score_df, on="id", how="left")
        else:
            df = df.join(score_df, on=["term_id", "item_id"], how="left")

        all_score_cols.extend(out_cols)
        print("joined", path.name, "shape", df.shape)

    if not all_score_cols:
        raise RuntimeError("No score columns found. Check data/processed files.")

    n = df.height
    print("score columns", len(all_score_cols))

    # Fill nulls and convert to percentile ranks. Rank makes heterogeneous scores blendable.
    pct_cols = []
    for c in all_score_cols:
        med = df.select(pl.col(c).median()).item()
        if med is None or (isinstance(med, float) and np.isnan(med)):
            med = 0.0
        df = df.with_columns(pl.col(c).cast(pl.Float64).fill_null(float(med)).alias(c))

        # If a score is mostly constant, rank is not useful; still produce neutral 0.5.
        nunique = df.select(pl.col(c).n_unique()).item()
        pc = c + "_pct"
        if nunique <= 2:
            df = df.with_columns(pl.lit(0.5).alias(pc))
        else:
            df = df.with_columns(((pl.col(c).rank("average") - 1.0) / max(1, n - 1)).alias(pc))
        pct_cols.append(pc)

    weights = [feature_weight(c) for c in all_score_cols]
    denom = sum(weights)

    expr = None
    for pc, w in zip(pct_cols, weights):
        term = pl.col(pc) * float(w)
        expr = term if expr is None else expr + term
    df = df.with_columns((expr / float(denom)).alias("v67_blend"))

    # Two alternative blends: semantic-heavy and lexical/model-consensus.
    semantic_cols = [pc for pc in pct_cols if re.search(r"(ce|cross|bge|ft|bert|meta|prob)", pc, re.I)]
    lexical_cols = [pc for pc in pct_cols if re.search(r"(lex|overlap|match|coverage|brand|cat|category|root|hist|support|mutual)", pc, re.I)]
    if semantic_cols:
        df = df.with_columns(pl.mean_horizontal([pl.col(c) for c in semantic_cols]).alias("v67_semantic_blend"))
    else:
        df = df.with_columns(pl.col("v67_blend").alias("v67_semantic_blend"))

    if lexical_cols:
        df = df.with_columns(pl.mean_horizontal([pl.col(c) for c in lexical_cols]).alias("v67_lex_consensus_blend"))
    else:
        df = df.with_columns(pl.col("v67_blend").alias("v67_lex_consensus_blend"))

    # Final aggressive blend: weighted main + semantic + consensus.
    df = df.with_columns((
        0.56 * pl.col("v67_blend")
        + 0.27 * pl.col("v67_semantic_blend")
        + 0.17 * pl.col("v67_lex_consensus_blend")
    ).alias("v67_final_score"))

    keep = ["id", "term_id", "item_id", "v67_final_score", "v67_blend", "v67_semantic_blend", "v67_lex_consensus_blend"] + all_score_cols[:40]
    df = df.select([c for c in keep if c in df.columns])
    OUT_SCORE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_SCORE)
    print("saved score table", OUT_SCORE, df.shape)
    return df


def load_titles_for_audit(changes):
    # Best-effort audit enrichment; if columns differ, still output ids and scores.
    out = changes.copy()

    if TERMS.exists():
        try:
            terms = pd.read_csv(TERMS)
            term_col = "term_id" if "term_id" in terms.columns else None
            text_cols = [c for c in terms.columns if c != term_col and re.search(r"(query|term|text|name|arama)", c, re.I)]
            if term_col and text_cols:
                terms = terms[[term_col, text_cols[0]]].rename(columns={text_cols[0]: "query_text"})
                out = out.merge(terms, on="term_id", how="left")
        except Exception as e:
            print("audit terms skipped:", e)

    if ITEMS.exists():
        try:
            items = pd.read_csv(ITEMS)
            if "item_id" in items.columns:
                title_cols = [c for c in items.columns if re.search(r"(title|name|urun|ürün|product)", c, re.I)]
                brand_cols = [c for c in items.columns if re.search(r"(brand|marka)", c, re.I)]
                cat_cols = [c for c in items.columns if re.search(r"(category|kategori|cat)", c, re.I)]
                use = ["item_id"]
                ren = {}
                if title_cols:
                    use.append(title_cols[0]); ren[title_cols[0]] = "item_title"
                if brand_cols:
                    use.append(brand_cols[0]); ren[brand_cols[0]] = "item_brand"
                if cat_cols:
                    use.append(cat_cols[0]); ren[cat_cols[0]] = "item_category"
                items = items[use].rename(columns=ren)
                out = out.merge(items, on="item_id", how="left")
        except Exception as e:
            print("audit items skipped:", e)

    return out


def save_pred(name, pred, sample_ids, base_pred, score, meta, saved_rows, changes_for_audit=None):
    file = SUB_DIR / f"FINAL_CANDIDATE_{name}_{short_hash(name)}.csv"
    pd.DataFrame({"id": sample_ids, "prediction": pred.astype(np.int8)}).to_csv(file, index=False)

    row = {
        "variant": name,
        "file": str(file),
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_base": int((pred != base_pred).sum()),
        "score_mean_pred1": float(score[pred == 1].mean()),
        "score_mean_pred0": float(score[pred == 0].mean()),
        **meta,
    }
    saved_rows.append(row)
    print("saved", file, row)

    return file


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    force = "--force" in os.sys.argv
    score_df = build_blend_scores(force_rebuild=force)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    sample_ids = sample["id"].to_numpy()
    n = len(sample)

    base_path = find_base()
    base = pd.read_csv(base_path)
    base["id"] = base["id"].astype(str)
    if not base["id"].equals(sample["id"]):
        raise RuntimeError(f"Base id order mismatch: {base_path}")
    base_pred = base["prediction"].astype(np.int8).to_numpy()
    base_ones = int(base_pred.sum())
    print("BASE", base_path, "ones", base_ones, "ratio", base_pred.mean())

    pdf = score_df.select(["id", "term_id", "item_id", "v67_final_score", "v67_blend", "v67_semantic_blend", "v67_lex_consensus_blend"]).to_pandas()
    if not pdf["id"].astype(str).equals(sample["id"]):
        # Align just in case.
        pdf["id"] = pdf["id"].astype(str)
        pdf = sample.merge(pdf, on="id", how="left")
    score = pdf["v67_final_score"].fillna(pdf["v67_final_score"].median()).astype(float).to_numpy()

    order_desc = np.argsort(-score, kind="mergesort")
    order_asc = np.argsort(score, kind="mergesort")
    base0_order_desc = np.array([i for i in order_desc if base_pred[i] == 0], dtype=np.int64)
    base1_order_asc = np.array([i for i in order_asc if base_pred[i] == 1], dtype=np.int64)

    saved = []
    audit_parts = []

    # 1) Global rethreshold: no respect for base labels. This is the most aggressive family.
    for delta in GLOBAL_K_DELTAS:
        k = base_ones + delta
        if k <= 0 or k >= n:
            continue
        pred = np.zeros(n, dtype=np.int8)
        pred[order_desc[:k]] = 1
        name = f"v67_GLOBAL_RETHRESH_k{delta:+d}"
        save_pred(
            name, pred, sample_ids, base_pred, score,
            {"family": "global_rethreshold", "delta_ones": int(delta), "k": int(k)},
            saved,
        )

    # 2) Same-quota massive swaps: aggressive but keeps prevalence.
    for cap in SWAP_CAPS:
        cap = min(cap, len(base0_order_desc), len(base1_order_asc))
        if cap <= 0:
            continue
        pred = base_pred.copy()
        add_idx = base0_order_desc[:cap]
        drop_idx = base1_order_asc[:cap]
        pred[add_idx] = 1
        pred[drop_idx] = 0

        name = f"v67_BIGSWAP_cap{cap}"
        save_pred(
            name, pred, sample_ids, base_pred, score,
            {
                "family": "big_swap",
                "used_swaps": int(cap),
                "delta_ones": 0,
                "add_score_mean": float(score[add_idx].mean()),
                "drop_score_mean": float(score[drop_idx].mean()),
                "margin": float(score[add_idx].mean() - score[drop_idx].mean()),
            },
            saved,
        )

        # Audit the largest and a medium cap.
        if cap in {5000, 20000, 70000, 100000}:
            ch_add = pdf.iloc[add_idx[:250]].copy()
            ch_add["action"] = "add"
            ch_add["variant"] = name
            ch_drop = pdf.iloc[drop_idx[:250]].copy()
            ch_drop["action"] = "drop"
            ch_drop["variant"] = name
            audit_parts.extend([ch_add, ch_drop])

    # 3) Pure quota shifts: if true positive count is wrong.
    for cap in SHIFT_CAPS:
        cap_add = min(cap, len(base0_order_desc))
        pred = base_pred.copy()
        pred[base0_order_desc[:cap_add]] = 1
        name = f"v67_PLUS_cap{cap_add}"
        save_pred(
            name, pred, sample_ids, base_pred, score,
            {
                "family": "plus",
                "used_plus": int(cap_add),
                "delta_ones": int(cap_add),
                "add_score_mean": float(score[base0_order_desc[:cap_add]].mean()),
            },
            saved,
        )

        cap_drop = min(cap, len(base1_order_asc))
        pred = base_pred.copy()
        pred[base1_order_asc[:cap_drop]] = 0
        name = f"v67_MINUS_cap{cap_drop}"
        save_pred(
            name, pred, sample_ids, base_pred, score,
            {
                "family": "minus",
                "used_minus": int(cap_drop),
                "delta_ones": -int(cap_drop),
                "drop_score_mean": float(score[base1_order_asc[:cap_drop]].mean()),
            },
            saved,
        )

    # 4) Hybrid: big swaps + quota shift. Aggressive but not full rethreshold.
    hybrid_specs = [
        (5000, 5000, 0),
        (10000, 10000, 0),
        (20000, 20000, 0),
        (40000, 20000, 0),
        (70000, 20000, 0),
        (10000, 0, 10000),
        (20000, 0, 20000),
        (40000, 0, 20000),
        (70000, 0, 20000),
        (20000, 10000, 10000),
        (40000, 20000, 20000),
        (70000, 20000, 20000),
    ]
    for swap_cap, plus_cap, minus_cap in hybrid_specs:
        swap_cap = min(swap_cap, len(base0_order_desc), len(base1_order_asc))
        plus_cap = min(plus_cap, max(0, len(base0_order_desc) - swap_cap))
        minus_cap = min(minus_cap, max(0, len(base1_order_asc) - swap_cap))

        pred = base_pred.copy()
        add_swap = base0_order_desc[:swap_cap]
        drop_swap = base1_order_asc[:swap_cap]
        pred[add_swap] = 1
        pred[drop_swap] = 0

        add_plus = base0_order_desc[swap_cap:swap_cap + plus_cap]
        drop_minus = base1_order_asc[swap_cap:swap_cap + minus_cap]
        pred[add_plus] = 1
        pred[drop_minus] = 0

        name = f"v67_HYBRID_swap{swap_cap}_plus{plus_cap}_minus{minus_cap}"
        save_pred(
            name, pred, sample_ids, base_pred, score,
            {
                "family": "hybrid",
                "used_swaps": int(swap_cap),
                "used_plus": int(plus_cap),
                "used_minus": int(minus_cap),
                "delta_ones": int(plus_cap - minus_cap),
                "add_score_mean": float(score[np.r_[add_swap, add_plus]].mean()) if len(add_swap) + len(add_plus) else np.nan,
                "drop_score_mean": float(score[np.r_[drop_swap, drop_minus]].mean()) if len(drop_swap) + len(drop_minus) else np.nan,
            },
            saved,
        )

    summary = pd.DataFrame(saved)
    # Diagnostic: favors large moves but punishes totally radical global-threshold less only as sorting aid.
    summary["v67_diagnostic_score"] = (
        0.45 * np.log1p(summary["diff_vs_base"]) / np.log1p(max(1, summary["diff_vs_base"].max()))
        + 0.25 * summary["score_mean_pred1"]
        - 0.10 * summary["score_mean_pred0"]
        + 0.10 * (summary["family"].eq("big_swap")).astype(float)
        + 0.05 * (summary["family"].eq("hybrid")).astype(float)
        - 0.06 * (summary["family"].eq("global_rethreshold")).astype(float)
    )
    summary = summary.sort_values(["v67_diagnostic_score", "diff_vs_base"], ascending=[False, False])
    summary.to_csv(OUT_SUMMARY, index=False)
    summary.to_csv(OUT_SAVED, index=False)

    if audit_parts:
        audit = pd.concat(audit_parts, ignore_index=True)
        audit = load_titles_for_audit(audit)
        audit.to_csv(OUT_AUDIT, index=False)

    print("\nTOP SUMMARY")
    cols = [
        "variant", "file", "family", "ones", "pos_ratio", "diff_vs_base",
        "delta_ones", "used_swaps", "used_plus", "used_minus",
        "score_mean_pred1", "score_mean_pred0", "add_score_mean", "drop_score_mean",
        "v67_diagnostic_score",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(100).to_string(index=False))

    print("\noutputs:")
    print(OUT_SCORE)
    print(OUT_SUMMARY)
    print(OUT_SAVED)
    print(OUT_AUDIT)
    print(SUB_DIR)


if __name__ == "__main__":
    main()
