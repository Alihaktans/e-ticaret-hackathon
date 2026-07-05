from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(".")

COMPACT = ROOT / "data/processed/v70_trendyol_embedding/v70_compact_rank_features.parquet"
POISON = ROOT / "data/processed/v69_fast_poison_guarded_scores.parquet"

LABELS = [
    ("v29", ROOT / "reports/manual_review/review_v29_qswap_targets_assistant_labeled.csv"),
    ("v33", ROOT / "reports/manual_review/review_v33_final_qswap_targets_assistant_labeled.csv"),
]

OUT_DIR = ROOT / "reports/experiments"
MANUAL_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v72"

OUT_FEATURES = ROOT / "data/processed/v72_anchorless_trendlex_features.parquet"
OUT_SEARCH = OUT_DIR / "v72_anchorless_weight_search.csv"
OUT_REPORT = OUT_DIR / "v72_anchorless_ranker_report.json"
OUT_SUMMARY = MANUAL_DIR / "v72_anchorless_candidate_summary.csv"
OUT_AUDIT = MANUAL_DIR / "v72_anchorless_audit_sample.csv"

# Fully anchor-free families. Ratios are intentionally broad because we do not
# assume a public-base positive count.
GLOBAL_RATIOS = [0.26, 0.28, 0.30, 0.32, 0.34, 0.36]
QUERY_RATIOS = [0.20, 0.24, 0.28, 0.32, 0.36]
Z_THRESHOLDS = [0.35, 0.55, 0.75, 0.95]
CONSENSUS_SPECS = [
    ("g58_t45_top1", 0.58, 0.45, 1),
    ("g60_t50_top1", 0.60, 0.50, 1),
    ("g62_t52_top1", 0.62, 0.52, 1),
    ("g65_t55_top1", 0.65, 0.55, 1),
    ("g55_t40_top2", 0.55, 0.40, 2),
]
HYBRID_SPECS = [
    ("union_g30_q24", 0.30, 0.24, "union"),
    ("union_g32_q28", 0.32, 0.28, "union"),
    ("inter_g34_q32", 0.34, 0.32, "inter"),
    ("inter_g36_q36", 0.36, 0.36, "inter"),
]
ELBOW_MAX_RANKS = [4, 6, 8, 12]


def short_name(name: str) -> str:
    import hashlib

    return hashlib.md5(name.encode("utf-8")).hexdigest()[:8]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def series_pct(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True).astype(np.float32)


def clean_manual_labels() -> pd.DataFrame:
    parts = []
    for source, path in LABELS:
        d = pd.read_csv(path)
        label = pd.to_numeric(d["assistant_swap_label"], errors="coerce")
        recheck = pd.to_numeric(d.get("needs_recheck", 0), errors="coerce").fillna(1)
        confidence = d.get("assistant_confidence", "").astype(str).str.lower()
        keep = label.isin([0, 1]) & recheck.eq(0) & confidence.isin(["high", "medium"])
        x = d.loc[keep, ["term_id", "item_id_add", "item_id_drop"]].copy()
        x["term_id"] = x["term_id"].astype(str)
        x["item_id_add"] = x["item_id_add"].astype(str)
        x["item_id_drop"] = x["item_id_drop"].astype(str)
        x["label"] = label.loc[keep].astype(np.int8)
        x["source"] = source
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def load_feature_table() -> pd.DataFrame:
    compact = pd.read_parquet(COMPACT)
    compact["id"] = compact["id"].astype(str)
    compact["term_id"] = compact["term_id"].astype(str)
    compact["item_id"] = compact["item_id"].astype(str)

    poison = pd.read_parquet(POISON, columns=["id", "v69_poison_penalty"])
    poison["id"] = poison["id"].astype(str)

    df = compact.merge(poison, on="id", how="left")
    df["v69_poison_penalty"] = pd.to_numeric(df["v69_poison_penalty"], errors="coerce").fillna(0).astype(np.float32)

    # Safety: old compact files already contain v69_qpct. If not, derive it from poison-guard
    # ranks only when the column is missing.
    if "v69_qpct" not in df.columns:
        fallback = pd.read_parquet(POISON, columns=["id", "term_id", "v69_guarded_score"])
        fallback["id"] = fallback["id"].astype(str)
        fallback["term_id"] = fallback["term_id"].astype(str)
        fallback["v69_qpct"] = fallback.groupby("term_id")["v69_guarded_score"].rank(method="average", pct=True)
        df = df.drop(columns=["term_id"], errors="ignore").merge(
            fallback[["id", "term_id", "v69_qpct"]], on="id", how="left"
        )

    df["trendyol_qpct"] = pd.to_numeric(df["trendyol_qpct"], errors="coerce").fillna(0.5).astype(np.float32)
    df["lex_qpct"] = pd.to_numeric(df["lex_qpct"], errors="coerce").fillna(0.5).astype(np.float32)
    df["v69_qpct"] = pd.to_numeric(df["v69_qpct"], errors="coerce").fillna(0.5).astype(np.float32)
    df["trendyol_score"] = pd.to_numeric(df["trendyol_score"], errors="coerce").fillna(df["trendyol_score"].median()).astype(np.float32)
    df["trend_raw_pct"] = series_pct(df["trendyol_score"]).astype(np.float32)
    return df


def search_weights(features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    labels = clean_manual_labels()
    key = features[["term_id", "item_id", "trendyol_qpct", "lex_qpct", "v69_qpct", "v69_poison_penalty"]]
    add = key.rename(
        columns={
            "item_id": "item_id_add",
            "trendyol_qpct": "trendyol_qpct_add",
            "lex_qpct": "lex_qpct_add",
            "v69_qpct": "v69_qpct_add",
            "v69_poison_penalty": "v69_poison_penalty_add",
        }
    )
    drop = key.rename(
        columns={
            "item_id": "item_id_drop",
            "trendyol_qpct": "trendyol_qpct_drop",
            "lex_qpct": "lex_qpct_drop",
            "v69_qpct": "v69_qpct_drop",
            "v69_poison_penalty": "v69_poison_penalty_drop",
        }
    )
    data = labels.merge(add, on=["term_id", "item_id_add"], how="left", validate="many_to_one")
    data = data.merge(drop, on=["term_id", "item_id_drop"], how="left", validate="many_to_one")
    if data.filter(regex="_(add|drop)$").isna().any().any():
        raise RuntimeError("manual audit join failed for v72 weight search")

    rows = []
    trend_margin = data["trendyol_qpct_add"] - data["trendyol_qpct_drop"]
    lex_margin = data["lex_qpct_add"] - data["lex_qpct_drop"]
    v69_margin = data["v69_qpct_add"] - data["v69_qpct_drop"]
    poison_margin = data["v69_poison_penalty_add"] - data["v69_poison_penalty_drop"]

    for wt in np.arange(0.10, 0.41, 0.05):
        for wl in np.arange(0.25, 0.71, 0.05):
            for w69 in np.arange(0.00, 0.31, 0.05):
                if wt + wl + w69 <= 0:
                    continue
                total = wt + wl + w69
                wt_n, wl_n, w69_n = wt / total, wl / total, w69 / total
                base = wt_n * trend_margin + wl_n * lex_margin + w69_n * v69_margin
                for wp in np.arange(0.00, 0.31, 0.05):
                    margin = base - wp * poison_margin
                    row = {
                        "w_trend": round(float(wt_n), 4),
                        "w_lex": round(float(wl_n), 4),
                        "w_v69": round(float(w69_n), 4),
                        "w_poison": round(float(wp), 4),
                    }
                    aucs = []
                    aps = []
                    for source in ["v29", "v33", "all"]:
                        mask = np.ones(len(data), dtype=bool) if source == "all" else data["source"].eq(source).to_numpy()
                        y = data.loc[mask, "label"].to_numpy()
                        score = margin.to_numpy()[mask]
                        auc = roc_auc_score(y, score)
                        ap = average_precision_score(y, score)
                        row[f"auc_{source}"] = float(auc)
                        row[f"ap_{source}"] = float(ap)
                        aucs.append(auc)
                        aps.append(ap)
                    row["balanced_score"] = float(
                        0.60 * row["auc_all"]
                        + 0.20 * min(row["auc_v29"], row["auc_v33"])
                        + 0.10 * np.mean(aucs)
                        + 0.10 * np.mean(aps)
                        - 0.05 * abs(row["auc_v29"] - row["auc_v33"])
                    )
                    rows.append(row)

    search = pd.DataFrame(rows).sort_values(
        ["balanced_score", "auc_all", "ap_all"],
        ascending=[False, False, False],
    )
    best = search.iloc[0].to_dict()
    return best, search


def build_scored_features(base: pd.DataFrame, best: dict) -> pd.DataFrame:
    df = base.copy()
    blend_raw = (
        best["w_trend"] * df["trendyol_qpct"].to_numpy(np.float32)
        + best["w_lex"] * df["lex_qpct"].to_numpy(np.float32)
        + best["w_v69"] * df["v69_qpct"].to_numpy(np.float32)
        - best["w_poison"] * df["v69_poison_penalty"].to_numpy(np.float32)
    )
    df["v72_blend_raw"] = blend_raw.astype(np.float32)
    df["v72_blend_pct"] = series_pct(df["v72_blend_raw"]).astype(np.float32)

    df["v72_term_rank"] = (
        df.groupby("term_id")["v72_blend_raw"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    df["v72_term_n"] = df.groupby("term_id")["id"].transform("size").astype(np.int32)
    df["v72_term_top_pct"] = (
        1.0 - ((df["v72_term_rank"] - 1) / np.maximum(1, df["v72_term_n"] - 1))
    ).astype(np.float32)
    group = df.groupby("term_id")["v72_blend_raw"]
    df["v72_term_mean"] = group.transform("mean").astype(np.float32)
    df["v72_term_std"] = group.transform("std").replace(0, np.nan).fillna(0).astype(np.float32)
    df["v72_term_z"] = (
        (df["v72_blend_raw"] - df["v72_term_mean"]) / df["v72_term_std"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    df["v72_term_z_sigmoid"] = sigmoid(df["v72_term_z"].to_numpy(np.float32)).astype(np.float32)

    # Final score mixes global and query-local confidence, then reapplies a mild
    # poison penalty so obviously suspicious rows stay down-ranked.
    df["v72_final_score"] = (
        0.58 * df["v72_blend_pct"]
        + 0.22 * df["v72_term_top_pct"]
        + 0.12 * df["v72_term_z_sigmoid"]
        + 0.08 * df["trend_raw_pct"]
        - 0.14 * df["v69_poison_penalty"]
    ).astype(np.float32)

    df["v72_final_pct"] = series_pct(df["v72_final_score"]).astype(np.float32)
    df["v72_final_rank"] = (
        df.groupby("term_id")["v72_final_score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )
    df["v72_final_top_pct"] = (
        1.0 - ((df["v72_final_rank"] - 1) / np.maximum(1, df["v72_term_n"] - 1))
    ).astype(np.float32)
    group2 = df.groupby("term_id")["v72_final_score"]
    df["v72_final_z"] = (
        (df["v72_final_score"] - group2.transform("mean")) / group2.transform("std").replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    return df


def pred_global_topk(score: np.ndarray, ratio: float) -> np.ndarray:
    n = len(score)
    k = int(round(n * ratio))
    pred = np.zeros(n, dtype=np.int8)
    order = np.argsort(-score, kind="mergesort")
    pred[order[:k]] = 1
    return pred


def pred_query_ratio(rank: np.ndarray, term_n: np.ndarray, ratio: float) -> np.ndarray:
    keep = np.maximum(1, np.ceil(term_n * ratio).astype(np.int32))
    return (rank <= keep).astype(np.int8)


def pred_query_z(rank: np.ndarray, z: np.ndarray, threshold: float) -> np.ndarray:
    pred = (z >= threshold).astype(np.int8)
    pred = np.maximum(pred, (rank <= 1).astype(np.int8))
    return pred


def pred_consensus(global_pct: np.ndarray, top_pct: np.ndarray, rank: np.ndarray, gmin: float, tmin: float, top_always: int) -> np.ndarray:
    return (((global_pct >= gmin) & (top_pct >= tmin)) | (rank <= top_always)).astype(np.int8)


def pred_hybrid(score: np.ndarray, rank: np.ndarray, term_n: np.ndarray, global_ratio: float, query_ratio: float, mode: str) -> np.ndarray:
    global_pred = pred_global_topk(score, global_ratio).astype(bool)
    query_pred = pred_query_ratio(rank, term_n, query_ratio).astype(bool)
    if mode == "union":
        return (global_pred | query_pred).astype(np.int8)
    pred = (global_pred & query_pred).astype(np.int8)
    pred = np.maximum(pred, (rank <= 1).astype(np.int8))
    return pred


def pred_query_elbow(frame: pd.DataFrame, max_rank: int) -> np.ndarray:
    work = frame[["term_id", "v72_final_score"]].copy()
    work["row"] = np.arange(len(work), dtype=np.int32)
    work = work.sort_values(["term_id", "v72_final_score", "row"], ascending=[True, False, True]).reset_index(drop=True)
    work["rank"] = work.groupby("term_id", sort=False).cumcount() + 1
    work["next_score"] = work.groupby("term_id", sort=False)["v72_final_score"].shift(-1)
    work["gap"] = (work["v72_final_score"] - work["next_score"]).fillna(-1.0)

    picks = []
    for term_id, g in work.groupby("term_id", sort=False):
        x = g[g["rank"] <= max_rank].copy()
        if len(x) <= 1:
            picks.append((term_id, 1))
            continue
        x = x[x["rank"] < x["rank"].max()]
        if len(x) == 0:
            picks.append((term_id, 1))
            continue
        best = x.sort_values(["gap", "rank"], ascending=[False, True]).iloc[0]
        picks.append((term_id, int(best["rank"])))
    limit = pd.Series(dict(picks))
    cut = work["term_id"].map(limit).to_numpy()
    keep = (work["rank"].to_numpy() <= cut).astype(np.int8)
    pred = np.zeros(len(work), dtype=np.int8)
    pred[work["row"].to_numpy()] = keep
    return pred


def add_query_item_text(audit: pd.DataFrame) -> pd.DataFrame:
    out = audit.copy()

    terms = pd.read_csv(ROOT / "data/raw/terms.csv")
    query_cols = [c for c in terms.columns if c != "term_id" and any(k in c.lower() for k in ["query", "term", "text", "arama"])]
    if "term_id" in terms.columns and query_cols:
        q = terms[["term_id", query_cols[0]]].rename(columns={query_cols[0]: "query_text"})
        q["term_id"] = q["term_id"].astype(str)
        out = out.merge(q, on="term_id", how="left")

    items = pd.read_csv(ROOT / "data/raw/items.csv", low_memory=False)
    title_cols = [c for c in items.columns if any(k in c.lower() for k in ["title", "name", "urun", "ürün", "product"])]
    brand_cols = [c for c in items.columns if any(k in c.lower() for k in ["brand", "marka"])]
    cat_cols = [c for c in items.columns if any(k in c.lower() for k in ["category", "kategori", "cat"])]
    use = ["item_id"]
    ren = {}
    if title_cols:
        use.append(title_cols[0]); ren[title_cols[0]] = "item_title"
    if brand_cols:
        use.append(brand_cols[0]); ren[brand_cols[0]] = "item_brand"
    if cat_cols:
        use.append(cat_cols[0]); ren[cat_cols[0]] = "item_category"
    items = items[use].rename(columns=ren)
    items["item_id"] = items["item_id"].astype(str)
    out = out.merge(items, on="item_id", how="left")
    return out


def summarize_variant(name: str, pred: np.ndarray, frame: pd.DataFrame, family: str, extra: dict) -> dict:
    score = frame["v72_final_score"].to_numpy(np.float32)
    pen = frame["v69_poison_penalty"].to_numpy(np.float32)
    term_pos = (
        pd.DataFrame({"term_id": frame["term_id"], "pred": pred})
        .groupby("term_id", sort=False)["pred"]
        .sum()
    )
    row = {
        "variant": name,
        "family": family,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "score_mean_pred1": float(score[pred == 1].mean()) if pred.sum() else np.nan,
        "score_mean_pred0": float(score[pred == 0].mean()) if (pred == 0).sum() else np.nan,
        "poison_rate_pred1": float((pen[pred == 1] > 0).mean()) if pred.sum() else np.nan,
        "poison_penalty_mean_pred1": float(pen[pred == 1].mean()) if pred.sum() else np.nan,
        "mean_term_positives": float(term_pos.mean()),
        "median_term_positives": float(term_pos.median()),
        "p95_term_positives": float(term_pos.quantile(0.95)),
        "queries_with_zero_pos": int((term_pos == 0).sum()),
        "queries_with_gt20_pos": int((term_pos > 20).sum()),
        **extra,
    }
    row["v72_diagnostic_score"] = float(
        0.34 * row["score_mean_pred1"]
        - 0.12 * row["score_mean_pred0"]
        - 0.16 * row["poison_penalty_mean_pred1"]
        - 0.05 * row["poison_rate_pred1"]
        - 0.12 * abs(row["pos_ratio"] - 0.32)
        - 0.04 * (row["queries_with_zero_pos"] / max(1, len(term_pos)))
    )
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv", usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    base = load_feature_table()
    best, search = search_weights(base)
    search.to_csv(OUT_SEARCH, index=False)

    features = build_scored_features(base, best)
    if not features["id"].equals(sample["id"]):
        features = sample.merge(features, on="id", how="left")
        for col in [
            "v72_final_score", "v72_final_pct", "v72_final_top_pct", "v72_final_z",
            "v69_poison_penalty", "term_raw_pct", "trend_raw_pct"
        ]:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors="coerce").fillna(features[col].median())
        for col in ["v72_final_rank", "v72_term_n"]:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors="coerce").fillna(999999).astype(np.int32)
        features["term_id"] = features["term_id"].astype(str)
        features["item_id"] = features["item_id"].astype(str)

    features.to_parquet(OUT_FEATURES, index=False)

    summary_rows = []
    audit_frames = []

    ids = features["id"]
    score = features["v72_final_score"].to_numpy(np.float32)
    rank = features["v72_final_rank"].to_numpy(np.int32)
    term_n = features["v72_term_n"].to_numpy(np.int32)
    top_pct = features["v72_final_top_pct"].to_numpy(np.float32)
    global_pct = features["v72_final_pct"].to_numpy(np.float32)
    term_z = features["v72_final_z"].to_numpy(np.float32)

    def save_variant(name: str, pred: np.ndarray, family: str, meta: dict) -> None:
        path = SUB_DIR / f"FINAL_CANDIDATE_{name}_{short_name(name)}.csv"
        pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(path, index=False)
        row = summarize_variant(name, pred, features, family, meta)
        row["file"] = str(path)
        summary_rows.append(row)

    for ratio in GLOBAL_RATIOS:
        pred = pred_global_topk(score, ratio)
        save_variant(f"v72_GLOBAL_ratio{str(ratio).replace('.', 'p')}", pred, "global_topk", {"target_ratio": float(ratio)})

    for ratio in QUERY_RATIOS:
        pred = pred_query_ratio(rank, term_n, ratio)
        save_variant(f"v72_QUERY_ratio{str(ratio).replace('.', 'p')}", pred, "query_ratio", {"query_ratio": float(ratio)})

    for z in Z_THRESHOLDS:
        pred = pred_query_z(rank, term_z, z)
        save_variant(f"v72_QUERY_Z{str(z).replace('.', 'p')}", pred, "query_z", {"z_threshold": float(z)})

    for name, gmin, tmin, top_always in CONSENSUS_SPECS:
        pred = pred_consensus(global_pct, top_pct, rank, gmin, tmin, top_always)
        save_variant(
            f"v72_CONSENSUS_{name}",
            pred,
            "consensus",
            {"global_pct_min": float(gmin), "term_top_pct_min": float(tmin), "top_rank_always": int(top_always)},
        )

    for name, gr, qr, mode in HYBRID_SPECS:
        pred = pred_hybrid(score, rank, term_n, gr, qr, mode)
        save_variant(
            f"v72_HYBRID_{name}",
            pred,
            "hybrid",
            {"global_ratio": float(gr), "query_ratio": float(qr), "hybrid_mode": mode},
        )

    for max_rank in ELBOW_MAX_RANKS:
        pred = pred_query_elbow(features, max_rank)
        save_variant(f"v72_ELBOW_top{max_rank}", pred, "query_elbow", {"elbow_max_rank": int(max_rank)})

    summary = pd.DataFrame(summary_rows).sort_values(
        ["v72_diagnostic_score", "score_mean_pred1", "pos_ratio"],
        ascending=[False, False, False],
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    top_variants = summary.head(8)["variant"].tolist()
    for variant in top_variants:
        pred_path = summary.loc[summary["variant"].eq(variant), "file"].iloc[0]
        pred = pd.read_csv(pred_path)["prediction"].to_numpy(np.int8)
        pos = features.loc[pred == 1, ["id", "term_id", "item_id", "v72_final_score", "v69_poison_penalty"]].copy()
        pos["variant"] = variant
        audit_frames.append(pos.sort_values("v72_final_score", ascending=False).head(80))
        audit_frames.append(pos.sort_values("v72_final_score", ascending=True).head(80))
    audit = pd.concat(audit_frames, ignore_index=True)
    audit = add_query_item_text(audit)
    audit.to_csv(OUT_AUDIT, index=False)

    report = {
        "best_weights": best,
        "search_rows": int(len(search)),
        "top_candidates": summary.head(12).to_dict(orient="records"),
        "warning": "All v72 candidates are anchor-free. Diagnostics rely on manual swap labels and internal score quality, not hidden Kaggle truth.",
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")

    print(json.dumps({"best_weights": best}, ensure_ascii=False, indent=2))
    cols = [
        "variant", "family", "ones", "pos_ratio", "score_mean_pred1", "score_mean_pred0",
        "poison_rate_pred1", "poison_penalty_mean_pred1", "mean_term_positives",
        "queries_with_zero_pos", "queries_with_gt20_pos", "v72_diagnostic_score",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))
    print("outputs:")
    print(OUT_FEATURES)
    print(OUT_SEARCH)
    print(OUT_SUMMARY)
    print(OUT_AUDIT)
    print(OUT_REPORT)
    print(SUB_DIR)


if __name__ == "__main__":
    main()
