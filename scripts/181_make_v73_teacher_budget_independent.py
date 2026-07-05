from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(".")

FEATURES = ROOT / "data/processed/v72_anchorless_trendlex_features.parquet"
TEACHER = ROOT / "submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv"

OUT_DIR = ROOT / "reports/experiments"
MANUAL_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v73"

OUT_QUERY = ROOT / "data/processed/v73_query_budget_features.parquet"
OUT_REPORT = OUT_DIR / "v73_teacher_budget_report.json"
OUT_SUMMARY = MANUAL_DIR / "v73_teacher_budget_candidate_summary.csv"
OUT_AUDIT = MANUAL_DIR / "v73_teacher_budget_audit.csv"

SCALES = [0.90, 1.00, 1.10]
SCORE_FAMILIES = ["trendlex", "v72", "mixed"]
FLOORS = [0, 1]


def short_name(name: str) -> str:
    import hashlib

    return hashlib.md5(name.encode("utf-8")).hexdigest()[:8]


def safe_stat(values: np.ndarray, fn, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    x = fn(values)
    if isinstance(x, np.ndarray):
        x = float(x.item())
    return float(x)


def build_query_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    teacher = pd.read_csv(TEACHER)
    teacher["id"] = teacher["id"].astype(str)
    work = frame.copy()
    work["id"] = work["id"].astype(str)
    work["term_id"] = work["term_id"].astype(str)
    work["item_id"] = work["item_id"].astype(str)
    work = work.merge(teacher, on="id", how="left", validate="one_to_one")
    if work["prediction"].isna().any():
        raise RuntimeError("teacher join failed for v73")
    work["prediction"] = work["prediction"].astype(np.int8)

    work["trendlex_score"] = (
        0.55 * work["lex_qpct"].astype(np.float32)
        + 0.30 * work["trendyol_qpct"].astype(np.float32)
        + 0.15 * work["v69_qpct"].astype(np.float32)
        - 0.25 * work["v69_poison_penalty"].astype(np.float32)
    ).astype(np.float32)
    work["mixed_score"] = (
        0.65 * work["v72_final_score"].astype(np.float32)
        + 0.35 * work["trendlex_score"].astype(np.float32)
    ).astype(np.float32)

    query_rows = []
    for term_id, g in work.groupby("term_id", sort=False):
        v72 = np.sort(g["v72_final_score"].to_numpy(np.float32))[::-1]
        tl = np.sort(g["trendlex_score"].to_numpy(np.float32))[::-1]
        mix = np.sort(g["mixed_score"].to_numpy(np.float32))[::-1]
        n = len(g)

        def top(arr: np.ndarray, idx: int) -> float:
            return float(arr[idx]) if idx < len(arr) else float(arr[-1]) if len(arr) else 0.0

        def gap(arr: np.ndarray, i: int, j: int) -> float:
            return top(arr, i) - top(arr, j)

        query_rows.append(
            {
                "term_id": term_id,
                "term_n": n,
                "teacher_k": int(g["prediction"].sum()),
                "teacher_ratio": float(g["prediction"].mean()),
                "v72_mean": safe_stat(v72, np.mean),
                "v72_std": safe_stat(v72, np.std),
                "v72_max": top(v72, 0),
                "v72_p95": safe_stat(v72, lambda x: np.quantile(x, 0.95)),
                "v72_p90": safe_stat(v72, lambda x: np.quantile(x, 0.90)),
                "v72_top1": top(v72, 0),
                "v72_top2": top(v72, 1),
                "v72_top3": top(v72, 2),
                "v72_top5_mean": safe_stat(v72[:5], np.mean),
                "v72_top10_mean": safe_stat(v72[:10], np.mean),
                "v72_gap_1_2": gap(v72, 0, 1),
                "v72_gap_2_3": gap(v72, 1, 2),
                "v72_gap_3_5": top(v72, 2) - top(v72, 4),
                "v72_gap_5_10": top(v72, 4) - top(v72, 9),
                "v72_gap_10_20": top(v72, 9) - top(v72, 19),
                "tl_mean": safe_stat(tl, np.mean),
                "tl_std": safe_stat(tl, np.std),
                "tl_max": top(tl, 0),
                "tl_p95": safe_stat(tl, lambda x: np.quantile(x, 0.95)),
                "tl_p90": safe_stat(tl, lambda x: np.quantile(x, 0.90)),
                "tl_top1": top(tl, 0),
                "tl_top2": top(tl, 1),
                "tl_top3": top(tl, 2),
                "tl_top5_mean": safe_stat(tl[:5], np.mean),
                "tl_top10_mean": safe_stat(tl[:10], np.mean),
                "tl_gap_1_2": gap(tl, 0, 1),
                "tl_gap_2_3": gap(tl, 1, 2),
                "tl_gap_3_5": top(tl, 2) - top(tl, 4),
                "tl_gap_5_10": top(tl, 4) - top(tl, 9),
                "tl_gap_10_20": top(tl, 9) - top(tl, 19),
                "mix_mean": safe_stat(mix, np.mean),
                "mix_std": safe_stat(mix, np.std),
                "mix_max": top(mix, 0),
                "mix_p95": safe_stat(mix, lambda x: np.quantile(x, 0.95)),
                "mix_top1": top(mix, 0),
                "mix_top2": top(mix, 1),
                "mix_top3": top(mix, 2),
                "mix_top5_mean": safe_stat(mix[:5], np.mean),
                "mix_top10_mean": safe_stat(mix[:10], np.mean),
                "mix_gap_1_2": gap(mix, 0, 1),
                "mix_gap_2_3": gap(mix, 1, 2),
                "mix_gap_3_5": top(mix, 2) - top(mix, 4),
                "mix_gap_5_10": top(mix, 4) - top(mix, 9),
                "mix_gap_10_20": top(mix, 9) - top(mix, 19),
                "poison_mean": float(g["v69_poison_penalty"].mean()),
                "poison_top10_mean": float(g.sort_values("mixed_score", ascending=False)["v69_poison_penalty"].head(10).mean()),
                "poison_pos_rate_over_0": float((g["v69_poison_penalty"] > 0).mean()),
            }
        )

    return work, pd.DataFrame(query_rows)


def fit_budget_model(query_df: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, dict]:
    features = [c for c in query_df.columns if c not in {"term_id", "teacher_k", "teacher_ratio"}]
    x = query_df[features].to_numpy(np.float32)
    y = np.log1p(query_df["teacher_k"].to_numpy(np.float32))

    cv = KFold(n_splits=5, shuffle=True, random_state=20260703)
    preds = np.zeros(len(query_df), dtype=np.float32)
    fold_rows = []
    for fold, (tr, va) in enumerate(cv.split(x), start=1):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=500,
            max_depth=5,
            min_samples_leaf=40,
            l2_regularization=0.05,
            random_state=20260703 + fold,
        )
        model.fit(x[tr], y[tr])
        pred = model.predict(x[va]).astype(np.float32)
        preds[va] = pred
        y_true = np.expm1(y[va])
        y_hat = np.expm1(pred)
        fold_rows.append(
            {
                "fold": fold,
                "mae_k": float(mean_absolute_error(y_true, y_hat)),
                "r2_log": float(r2_score(y[va], pred)),
                "mean_true_k": float(y_true.mean()),
                "mean_pred_k": float(y_hat.mean()),
            }
        )

    final_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_depth=5,
        min_samples_leaf=40,
        l2_regularization=0.05,
        random_state=20260703,
    )
    final_model.fit(x, y)
    full_pred = final_model.predict(x).astype(np.float32)
    report = {
        "features": features,
        "folds": fold_rows,
        "cv_mae_k_mean": float(np.mean([r["mae_k"] for r in fold_rows])),
        "cv_r2_log_mean": float(np.mean([r["r2_log"] for r in fold_rows])),
        "teacher_mean_k": float(np.expm1(y).mean()),
        "pred_mean_k": float(np.expm1(full_pred).mean()),
    }
    query_df["pred_teacher_k_raw"] = np.expm1(full_pred).astype(np.float32)
    return final_model, report


def rank_with_budget(frame: pd.DataFrame, budget_by_term: pd.Series, score_col: str) -> np.ndarray:
    work = frame[["term_id", score_col]].copy()
    work["row"] = np.arange(len(frame), dtype=np.int32)
    work["rank"] = work.groupby("term_id", sort=False)[score_col].rank(method="first", ascending=False)
    limit = work["term_id"].map(budget_by_term).to_numpy()
    pred = (work["rank"].to_numpy() <= limit).astype(np.int8)
    return pred


def summarize_variant(name: str, pred: np.ndarray, frame: pd.DataFrame, family: str, extra: dict) -> dict:
    score = frame["mixed_score"].to_numpy(np.float32)
    poison = frame["v69_poison_penalty"].to_numpy(np.float32)
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
        "poison_rate_pred1": float((poison[pred == 1] > 0).mean()) if pred.sum() else np.nan,
        "poison_penalty_mean_pred1": float(poison[pred == 1].mean()) if pred.sum() else np.nan,
        "mean_term_pos": float(term_pos.mean()),
        "median_term_pos": float(term_pos.median()),
        "p95_term_pos": float(term_pos.quantile(0.95)),
        "queries_zero_pos": int((term_pos == 0).sum()),
        "queries_gt50_pos": int((term_pos > 50).sum()),
        **extra,
    }
    row["v73_diagnostic_score"] = float(
        0.34 * row["score_mean_pred1"]
        - 0.12 * row["score_mean_pred0"]
        - 0.18 * row["poison_penalty_mean_pred1"]
        - 0.05 * row["poison_rate_pred1"]
        - 0.10 * abs(row["pos_ratio"] - 0.32)
        - 0.03 * (row["queries_zero_pos"] / max(1, len(term_pos)))
    )
    return row


def add_audit_text(audit: pd.DataFrame) -> pd.DataFrame:
    out = audit.copy()
    terms = pd.read_csv(ROOT / "data/raw/terms.csv")
    query_cols = [c for c in terms.columns if c != "term_id" and any(k in c.lower() for k in ["query", "term", "text", "arama"])]
    if "term_id" in terms.columns and query_cols:
        tmp = terms[["term_id", query_cols[0]]].rename(columns={query_cols[0]: "query_text"})
        tmp["term_id"] = tmp["term_id"].astype(str)
        out = out.merge(tmp, on="term_id", how="left")
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(FEATURES)
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv", usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    features["id"] = features["id"].astype(str)
    if not features["id"].equals(sample["id"]):
        raise RuntimeError("v73 feature/sample alignment failed")

    row_frame, query_df = build_query_features(features)
    model, report = fit_budget_model(query_df)
    query_df.to_parquet(OUT_QUERY, index=False)

    # Predicted budgets, but clipped so the teacher guides us without hard dependency.
    teacher_k = query_df.set_index("term_id")["teacher_k"]
    pred_k = query_df.set_index("term_id")["pred_teacher_k_raw"]
    term_n = query_df.set_index("term_id")["term_n"]

    summary_rows = []
    audit_frames = []

    def save_variant(name: str, pred: np.ndarray, family: str, meta: dict) -> None:
        path = SUB_DIR / f"FINAL_CANDIDATE_{name}_{short_name(name)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(path, index=False)
        row = summarize_variant(name, pred, row_frame, family, meta)
        row["file"] = str(path)
        summary_rows.append(row)

    for score_family in SCORE_FAMILIES:
        score_col = {"trendlex": "trendlex_score", "v72": "v72_final_score", "mixed": "mixed_score"}[score_family]
        for scale in SCALES:
            for floor in FLOORS:
                budgets = np.rint(pred_k * scale).astype(int)
                budgets = budgets.clip(lower=floor)
                budgets = budgets.where(budgets <= term_n, term_n)
                # Don't let a few teacher outliers dominate. Cap very large budgets softly.
                cap = np.maximum(1, np.rint(term_n * 0.75)).astype(int)
                budgets = np.minimum(budgets, cap)
                pred = rank_with_budget(row_frame, budgets, score_col)
                save_variant(
                    f"v73_{score_family}_teacher_scale{str(scale).replace('.', 'p')}_floor{floor}",
                    pred,
                    "teacher_budget",
                    {"score_family": score_family, "scale": float(scale), "floor": int(floor)},
                )

    # Two anchored-to-teacher but still independent alternatives:
    for score_family in SCORE_FAMILIES:
        score_col = {"trendlex": "trendlex_score", "v72": "v72_final_score", "mixed": "mixed_score"}[score_family]
        teacher_soft = np.rint(0.70 * pred_k + 0.30 * teacher_k).astype(int)
        teacher_soft = np.minimum(teacher_soft, np.maximum(1, np.rint(term_n * 0.80)).astype(int))
        pred = rank_with_budget(row_frame, teacher_soft, score_col)
        save_variant(
            f"v73_{score_family}_teacherblend70_30",
            pred,
            "teacher_blend",
            {"score_family": score_family, "teacher_blend": "0.70_pred_0.30_teacher"},
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["v73_diagnostic_score", "score_mean_pred1", "pos_ratio"],
        ascending=[False, False, False],
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    for variant in summary.head(8)["variant"]:
        pred_path = summary.loc[summary["variant"].eq(variant), "file"].iloc[0]
        pred = pd.read_csv(pred_path)["prediction"].to_numpy(np.int8)
        pos = row_frame.loc[pred == 1, ["id", "term_id", "item_id", "mixed_score", "v69_poison_penalty"]].copy()
        pos["variant"] = variant
        audit_frames.append(pos.sort_values("mixed_score", ascending=False).head(80))
        audit_frames.append(pos.sort_values("mixed_score", ascending=True).head(80))
    audit = pd.concat(audit_frames, ignore_index=True)
    audit = add_audit_text(audit)
    audit.to_csv(OUT_AUDIT, index=False)

    report["top_candidates"] = summary.head(12).to_dict(orient="records")
    report["warning"] = "V73 uses V71 trendlex query-budget only as a teacher for query-level positive counts. Final rows are re-ranked independently."
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")

    cols = [
        "variant", "family", "ones", "pos_ratio", "score_mean_pred1", "score_mean_pred0",
        "poison_rate_pred1", "poison_penalty_mean_pred1", "mean_term_pos", "queries_zero_pos",
        "queries_gt50_pos", "v73_diagnostic_score",
    ]
    print(json.dumps({k: report[k] for k in ["cv_mae_k_mean", "cv_r2_log_mean", "teacher_mean_k", "pred_mean_k"]}, ensure_ascii=False, indent=2))
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))
    print("outputs:")
    print(OUT_QUERY)
    print(OUT_SUMMARY)
    print(OUT_AUDIT)
    print(OUT_REPORT)
    print(SUB_DIR)


if __name__ == "__main__":
    main()
