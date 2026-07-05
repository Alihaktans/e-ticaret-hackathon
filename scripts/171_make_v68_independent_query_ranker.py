from pathlib import Path
import re
import hashlib
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V67_SCORE = ROOT / "data/processed/v67_global_aggressive_blend_scores.parquet"

# Only for diagnostics/audit diff. Predictions DO NOT use this base.
OPTIONAL_BASES = [
    ROOT / "submissions/final_candidates_v66/FINAL_CANDIDATE_v66_intent_ultra_swap_cap75_699710c5.csv",
    ROOT / "submissions/final_candidates_v60/v60_public080_ultra_cap250.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
]

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v68"

OUT_FEATURES = ROOT / "data/processed/v68_independent_query_rank_features.parquet"
OUT_SUMMARY = OUT_DIR / "v68_independent_ranker_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v68_independent_ranker_saved_candidates.csv"
OUT_AUDIT = OUT_DIR / "review_v68_independent_ranker_audit.csv"

# Independent, risky settings. These are not patch sizes; these create full prediction files.
GLOBAL_RATIOS = [0.24, 0.27, 0.30, 0.31684, 0.34, 0.37, 0.40, 0.44, 0.48]
QUERY_RATIOS = [0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.50]
Z_THRESHOLDS = [-0.45, -0.25, 0.00, 0.20, 0.40, 0.65, 0.90]
CONSENSUS_SPECS = [
    # name, global_pct_min, term_top_pct_min, min_rank_always
    ("consensus_g55_t45_top1", 0.55, 0.45, 1),
    ("consensus_g60_t40_top1", 0.60, 0.40, 1),
    ("consensus_g60_t50_top1", 0.60, 0.50, 1),
    ("consensus_g65_t45_top1", 0.65, 0.45, 1),
    ("consensus_g65_t55_top1", 0.65, 0.55, 1),
    ("consensus_g70_t50_top1", 0.70, 0.50, 1),
    ("consensus_g50_t35_top2", 0.50, 0.35, 2),
    ("consensus_g55_t35_top2", 0.55, 0.35, 2),
]
HYBRID_SPECS = [
    # global target ratio + query ratio union/intersection
    ("hybrid_union_global30_query22", 0.30, 0.22, "union"),
    ("hybrid_union_global34_query26", 0.34, 0.26, "union"),
    ("hybrid_union_global37_query30", 0.37, 0.30, "union"),
    ("hybrid_inter_global34_query34", 0.34, 0.34, "inter"),
    ("hybrid_inter_global40_query38", 0.40, 0.38, "inter"),
]


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


def find_optional_base(sample_ids):
    for p in OPTIONAL_BASES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if list(df.columns) == ["id", "prediction"] and df["id"].astype(str).equals(pd.Series(sample_ids).astype(str)):
                    return p, df["prediction"].astype(np.int8).to_numpy()
            except Exception:
                pass
    return None, None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def build_features(force=False):
    if OUT_FEATURES.exists() and not force:
        print("Using existing", OUT_FEATURES)
        return pl.read_parquet(OUT_FEATURES)

    if not V67_SCORE.exists():
        raise FileNotFoundError(f"{V67_SCORE} yok. Önce script 170 çalışmalı.")

    print("Building independent query-rank features from", V67_SCORE)
    lf = pl.scan_parquet(V67_SCORE)

    needed = ["id", "term_id", "item_id", "v67_final_score", "v67_blend", "v67_semantic_blend", "v67_lex_consensus_blend"]
    schema = lf.collect_schema()
    missing = [c for c in needed if c not in schema]
    if missing:
        raise RuntimeError(f"V67 score table missing columns: {missing}")

    lf = lf.select([
        pl.col("id").cast(pl.Utf8),
        pl.col("term_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("v67_final_score").cast(pl.Float64),
        pl.col("v67_blend").cast(pl.Float64),
        pl.col("v67_semantic_blend").cast(pl.Float64),
        pl.col("v67_lex_consensus_blend").cast(pl.Float64),
    ])

    n = lf.select(pl.len()).collect().item()
    print("rows", n)

    # Higher score is better. Create global percentile and per-query rank features.
    lf = lf.with_columns([
        ((pl.col("v67_final_score").rank("average") - 1.0) / max(1, n - 1)).alias("global_pct"),
        pl.col("v67_final_score").rank("ordinal", descending=True).over("term_id").cast(pl.Int32).alias("term_rank"),
        pl.len().over("term_id").cast(pl.Int32).alias("term_n"),
        pl.col("v67_final_score").mean().over("term_id").alias("term_mean"),
        pl.col("v67_final_score").std().over("term_id").alias("term_std"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("term_n") <= 1)
          .then(1.0)
          .otherwise(1.0 - ((pl.col("term_rank") - 1).cast(pl.Float64) / (pl.col("term_n") - 1).cast(pl.Float64)))
          .alias("term_top_pct"),
        pl.when((pl.col("term_std").is_null()) | (pl.col("term_std") < 1e-9))
          .then(0.0)
          .otherwise((pl.col("v67_final_score") - pl.col("term_mean")) / pl.col("term_std"))
          .alias("term_z"),
    ])

    # Independent meta score. No base labels. It combines global quality, query-relative quality and score distribution.
    lf = lf.with_columns([
        (1.0 / (1.0 + (-pl.col("term_z")).exp())).alias("term_z_sigmoid")
    ])

    lf = lf.with_columns([
        (
            0.46 * pl.col("global_pct")
            + 0.31 * pl.col("term_top_pct")
            + 0.15 * pl.col("term_z_sigmoid")
            + 0.05 * pl.col("v67_semantic_blend")
            + 0.03 * pl.col("v67_lex_consensus_blend")
        ).alias("v68_ind_score")
    ])

    df = lf.collect(streaming=True)
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FEATURES)
    print("saved", OUT_FEATURES, df.shape)
    return df


def load_audit_text(pdf):
    out = pdf.copy()

    if TERMS.exists():
        try:
            terms = pd.read_csv(TERMS)
            term_col = "term_id" if "term_id" in terms.columns else None
            text_cols = [c for c in terms.columns if c != term_col and re.search(r"(query|term|text|name|arama)", c, re.I)]
            if term_col and text_cols:
                tmp = terms[[term_col, text_cols[0]]].rename(columns={text_cols[0]: "query_text"})
                tmp["term_id"] = tmp["term_id"].astype(str)
                out = out.merge(tmp, on="term_id", how="left")
        except Exception as e:
            print("terms audit skip:", e)

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
                tmp = items[use].rename(columns=ren)
                tmp["item_id"] = tmp["item_id"].astype(str)
                out = out.merge(tmp, on="item_id", how="left")
        except Exception as e:
            print("items audit skip:", e)

    return out


def save_variant(name, pred, sample_ids, pdf, optional_base_pred, summary_rows, audit_frames, family, meta):
    file = SUB_DIR / f"FINAL_CANDIDATE_{name}_{short_hash(name)}.csv"
    pd.DataFrame({"id": sample_ids, "prediction": pred.astype(np.int8)}).to_csv(file, index=False)

    score = pdf["v68_ind_score"].to_numpy()
    row = {
        "variant": name,
        "file": str(file),
        "family": family,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "score_mean_pred1": float(score[pred == 1].mean()) if pred.sum() else np.nan,
        "score_mean_pred0": float(score[pred == 0].mean()) if (pred == 0).sum() else np.nan,
        **meta,
    }
    if optional_base_pred is not None:
        row["diff_vs_optional_public80_base"] = int((pred != optional_base_pred).sum())
        row["base_ones"] = int(optional_base_pred.sum())

    summary_rows.append(row)
    print("saved", file, row)

    # Audit: add/drop vs optional base if exists; otherwise top positives only.
    if optional_base_pred is not None:
        add_idx = np.where((pred == 1) & (optional_base_pred == 0))[0]
        drop_idx = np.where((pred == 0) & (optional_base_pred == 1))[0]
        if len(add_idx):
            x = pdf.iloc[add_idx].sort_values("v68_ind_score", ascending=False).head(150).copy()
            x["variant"] = name
            x["action"] = "add_vs_base"
            audit_frames.append(x)
        if len(drop_idx):
            x = pdf.iloc[drop_idx].sort_values("v68_ind_score", ascending=True).head(150).copy()
            x["variant"] = name
            x["action"] = "drop_vs_base"
            audit_frames.append(x)
    else:
        pos_idx = np.where(pred == 1)[0]
        if len(pos_idx):
            x = pdf.iloc[pos_idx].sort_values("v68_ind_score", ascending=False).head(200).copy()
            x["variant"] = name
            x["action"] = "top_positive"
            audit_frames.append(x)


def main():
    force = "--force" in set(__import__("sys").argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    sample_ids = sample["id"].to_numpy()
    n = len(sample_ids)

    df = build_features(force=force)
    pdf = df.to_pandas()
    pdf["id"] = pdf["id"].astype(str)

    # Align to sample order.
    if not pdf["id"].equals(sample["id"]):
        pdf = sample.merge(pdf, on="id", how="left")
    for c in ["v68_ind_score", "global_pct", "term_top_pct", "term_z", "v67_final_score"]:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(pdf[c].median())

    optional_base_path, optional_base_pred = find_optional_base(sample_ids)
    print("Optional base for diff only:", optional_base_path)

    score = pdf["v68_ind_score"].to_numpy()
    global_pct = pdf["global_pct"].to_numpy()
    term_rank = pdf["term_rank"].astype(int).to_numpy()
    term_n = pdf["term_n"].astype(int).to_numpy()
    term_top_pct = pdf["term_top_pct"].to_numpy()
    term_z = pdf["term_z"].to_numpy()

    order = np.argsort(-score, kind="mergesort")

    summary_rows = []
    audit_frames = []

    # Family 1: pure global top-K by independent score.
    for r in GLOBAL_RATIOS:
        k = int(round(n * r))
        pred = np.zeros(n, dtype=np.int8)
        pred[order[:k]] = 1
        save_variant(
            f"v68_IND_GLOBAL_ratio{str(r).replace('.', 'p')}",
            pred, sample_ids, pdf, optional_base_pred, summary_rows, audit_frames,
            "independent_global_topk",
            {"target_ratio": float(r), "k": int(k)}
        )

    # Family 2: query-adaptive top ratio. Every query gets its own top percentage.
    for r in QUERY_RATIOS:
        keep = np.ceil(term_n * r).astype(int)
        keep = np.maximum(1, keep)
        pred = (term_rank <= keep).astype(np.int8)
        save_variant(
            f"v68_IND_QUERY_ratio{str(r).replace('.', 'p')}",
            pred, sample_ids, pdf, optional_base_pred, summary_rows, audit_frames,
            "independent_query_ratio",
            {"query_ratio": float(r)}
        )

    # Family 3: query z-threshold. Variable positives per query based on query score distribution.
    for z in Z_THRESHOLDS:
        pred = (term_z >= z).astype(np.int8)
        # Always keep top1 per query so short queries don't get empty if scores are flat.
        pred = np.maximum(pred, (term_rank <= 1).astype(np.int8))
        save_variant(
            f"v68_IND_QUERY_Z{str(z).replace('-', 'm').replace('.', 'p')}",
            pred, sample_ids, pdf, optional_base_pred, summary_rows, audit_frames,
            "independent_query_z",
            {"z_threshold": float(z), "top1_forced": 1}
        )

    # Family 4: consensus gates. No base; this is rank + global confidence.
    for name, gmin, tmin, top_always in CONSENSUS_SPECS:
        pred = (((global_pct >= gmin) & (term_top_pct >= tmin)) | (term_rank <= top_always)).astype(np.int8)
        save_variant(
            f"v68_IND_{name}",
            pred, sample_ids, pdf, optional_base_pred, summary_rows, audit_frames,
            "independent_consensus",
            {"global_pct_min": float(gmin), "term_top_pct_min": float(tmin), "top_rank_always": int(top_always)}
        )

    # Family 5: hybrid global/query union/intersection.
    for name, gr, qr, mode in HYBRID_SPECS:
        k = int(round(n * gr))
        global_mask = np.zeros(n, dtype=bool)
        global_mask[order[:k]] = True
        qkeep = np.maximum(1, np.ceil(term_n * qr).astype(int))
        query_mask = term_rank <= qkeep
        if mode == "union":
            pred = (global_mask | query_mask).astype(np.int8)
        else:
            pred = (global_mask & query_mask).astype(np.int8)
            pred = np.maximum(pred, (term_rank <= 1).astype(np.int8))
        save_variant(
            f"v68_IND_{name}",
            pred, sample_ids, pdf, optional_base_pred, summary_rows, audit_frames,
            "independent_hybrid",
            {"global_ratio": float(gr), "query_ratio": float(qr), "hybrid_mode": mode}
        )

    summary = pd.DataFrame(summary_rows)

    # Diagnostic only. This is NOT validation. It ranks aggressive candidates by separation and scale.
    summary["v68_diagnostic_score"] = (
        0.30 * summary["score_mean_pred1"]
        - 0.12 * summary["score_mean_pred0"]
        + 0.18 * np.log1p(summary["ones"]) / np.log1p(n)
        - 0.20 * (summary["pos_ratio"] - 0.34).abs()
    )
    if "diff_vs_optional_public80_base" in summary.columns:
        summary["v68_diagnostic_score"] += 0.12 * np.minimum(1.0, np.log1p(summary["diff_vs_optional_public80_base"]) / np.log1p(n))

    summary = summary.sort_values(["v68_diagnostic_score", "pos_ratio"], ascending=[False, False])
    summary.to_csv(OUT_SUMMARY, index=False)
    summary.to_csv(OUT_SAVED, index=False)

    if audit_frames:
        audit = pd.concat(audit_frames, ignore_index=True)
        # Keep only a few variants in audit to avoid huge file.
        top_variants = summary.head(12)["variant"].tolist()
        audit = audit[audit["variant"].isin(top_variants)].copy()
        audit = load_audit_text(audit)
        audit.to_csv(OUT_AUDIT, index=False)

    print("\nTOP SUMMARY")
    cols = [
        "variant", "file", "family", "ones", "pos_ratio",
        "diff_vs_optional_public80_base", "score_mean_pred1", "score_mean_pred0",
        "target_ratio", "query_ratio", "z_threshold",
        "global_pct_min", "term_top_pct_min", "hybrid_mode",
        "v68_diagnostic_score",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(100).to_string(index=False))

    print("\noutputs:")
    print(OUT_FEATURES)
    print(OUT_SUMMARY)
    print(OUT_SAVED)
    print(OUT_AUDIT)
    print(SUB_DIR)


if __name__ == "__main__":
    main()
