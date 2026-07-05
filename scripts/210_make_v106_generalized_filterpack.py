"""Build a more generalizable filterpack with multi-fold OOF calibration."""
from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(".")
BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TRAIN_FEATURES = ROOT / "data/processed/v82_v34_features.parquet"
TEST_FEATURES = ROOT / "data/processed/v21_test_features.parquet"
FEATURE_REPORT = ROOT / "reports/experiments/v83_independent_pairlocal_classifier.json"

OUT_DIR = ROOT / "submissions/final_candidates_v106"
REPORT = ROOT / "reports/experiments/v106_generalized_filterpack.json"
OOF_OUT = ROOT / "data/processed/v106_generalized_oof_scores.parquet"
TEST_SCORE_OUT = ROOT / "data/processed/v106_generalized_test_scores.parquet"
MODEL_OUT = ROOT / "models/v106_generalized_pairlocal_classifier.cbm"

SCRIPT_209 = ROOT / "scripts/209_make_v100_multi_filterpack.py"

N_FOLDS = 5
CHUNK = 300_000
ATTR_GRID = {
    "gender_score_max": [0.10, 0.12, 0.15],
    "color_score_max": [0.08, 0.10, 0.12],
    "brand_score_max": [0.04, 0.06, 0.08],
    "brand_wov_pct_max": [0.40, 0.50],
}
MODEL_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 900,
    "learning_rate": 0.04,
    "depth": 8,
    "l2_leaf_reg": 10,
    "random_strength": 0.7,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "random_seed": 20260705,
    "od_type": "Iter",
    "od_wait": 80,
    "verbose": 150,
    "allow_writing_files": False,
    "task_type": "GPU",
    "devices": "0",
}
ATTR_COLS = [
    "query_has_color",
    "query_has_known_brand",
    "gender_mismatch",
    "color_mismatch",
    "query_brand_mismatch",
    "weighted_overlap_pct_rank",
    "last_token_in_title",
    "brand_in_query_pair",
]
FLAG_COLS = [
    "query_has_color",
    "query_has_known_brand",
    "gender_mismatch",
    "color_mismatch",
    "query_brand_mismatch",
    "last_token_in_title",
    "brand_in_query_pair",
]
FLOAT_COLS = ["weighted_overlap_pct_rank"]


def load_filterpack_module():
    spec = importlib.util.spec_from_file_location("filterpack209", SCRIPT_209)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def term_fold(term_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(str(term_id).encode()).hexdigest()[:8], 16) % n_folds


def train_catboost(xtr, ytr, xva, yva):
    params = dict(MODEL_PARAMS)
    model = CatBoostClassifier(**params)
    try:
        model.fit(xtr, ytr, eval_set=(xva, yva), use_best_model=True)
    except Exception as exc:
        print("GPU fallback:", repr(exc), flush=True)
        params["task_type"] = "CPU"
        params["thread_count"] = max(1, (os.cpu_count() or 2) - 1)
        params.pop("devices", None)
        model = CatBoostClassifier(**params)
        model.fit(xtr, ytr, eval_set=(xva, yva), use_best_model=True)
    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter <= 0:
        best_iter = params["iterations"]
    return model, int(best_iter)


def predict_in_chunks(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    out = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), CHUNK):
        end = min(len(df), start + CHUNK)
        x = df.iloc[start:end][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        out[start:end] = model.predict_proba(x)[:, 1].astype(np.float32)
        print("score", end, "/", len(df), flush=True)
    return out


def build_or_load_scores(feature_cols: list[str]):
    train_cols = ["term_id", "item_id", "label"] + list(dict.fromkeys(feature_cols + ATTR_COLS))
    test_cols = ["id", "term_id", "item_id"] + list(dict.fromkeys(feature_cols + ATTR_COLS + ["title_cov_pct_rank"]))
    train = pd.read_parquet(TRAIN_FEATURES, columns=train_cols)
    test = pd.read_parquet(TEST_FEATURES, columns=test_cols)
    for df in (train, test):
        for col in ("term_id", "item_id"):
            df[col] = df[col].astype(str)
    test["id"] = test["id"].astype(str)
    train["fold5"] = train["term_id"].map(term_fold).astype(np.int8)

    if OOF_OUT.exists() and TEST_SCORE_OUT.exists():
        oof = pd.read_parquet(OOF_OUT)
        score = pd.read_parquet(TEST_SCORE_OUT)
        oof["term_id"] = oof["term_id"].astype(str)
        oof["item_id"] = oof["item_id"].astype(str)
        score["id"] = score["id"].astype(str)
        score["term_id"] = score["term_id"].astype(str)
        score["item_id"] = score["item_id"].astype(str)
        if len(oof) == len(train) and len(score) == len(test):
            print("loaded cached OOF/test scores", flush=True)
            return train, test, oof, score

    oof_scores = np.empty(len(train), dtype=np.float32)
    best_iters = []
    for fold in range(N_FOLDS):
        tr = train[train["fold5"] != fold].copy()
        va = train[train["fold5"] == fold].copy()
        xtr = tr[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        xva = va[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        ytr = tr["label"].astype("int8").to_numpy()
        yva = va["label"].astype("int8").to_numpy()
        model, best_iter = train_catboost(xtr, ytr, xva, yva)
        best_iters.append(best_iter)
        oof_scores[va.index.to_numpy()] = model.predict_proba(xva)[:, 1].astype(np.float32)
        print({"fold": fold, "rows": len(va), "best_iter": best_iter}, flush=True)

    full_iter = max(300, int(round(float(np.median(best_iters)) * 1.05)))
    full_params = dict(MODEL_PARAMS)
    full_params["iterations"] = full_iter
    full_params.pop("od_type", None)
    full_params.pop("od_wait", None)
    full_model = CatBoostClassifier(**full_params)
    try:
        xall = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        yall = train["label"].astype("int8").to_numpy()
        full_model.fit(xall, yall)
    except Exception as exc:
        print("GPU fallback full:", repr(exc), flush=True)
        full_params["task_type"] = "CPU"
        full_params["thread_count"] = max(1, (os.cpu_count() or 2) - 1)
        full_params.pop("devices", None)
        full_model = CatBoostClassifier(**full_params)
        xall = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        yall = train["label"].astype("int8").to_numpy()
        full_model.fit(xall, yall)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    full_model.save_model(str(MODEL_OUT))
    test_scores = predict_in_chunks(full_model, test, feature_cols)

    oof = train[["term_id", "item_id", "label", "fold5"] + ATTR_COLS].copy()
    oof["oof_score"] = oof_scores
    score = test[["id", "term_id", "item_id", "title_cov_pct_rank"] + ATTR_COLS].copy()
    score["v106_score"] = test_scores
    OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OOF_OUT, index=False)
    score.to_parquet(TEST_SCORE_OUT, index=False)
    return train, test, oof, score


def attach_attr_thresholds(df: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in FLAG_COLS:
        out[col] = out[col].fillna(0).astype(np.int8)
    for col in FLOAT_COLS:
        out[col] = out[col].fillna(1.0).astype(np.float32)
    out["gender_score_veto"] = (
        (out["gender_mismatch"].eq(1)) & out["score"].le(cfg["gender_score_max"])
    ).astype(np.int8)
    out["color_score_veto"] = (
        (out["query_has_color"].eq(1))
        & (out["color_mismatch"].eq(1))
        & out["score"].le(cfg["color_score_max"])
    ).astype(np.int8)
    out["gender_color_score_veto"] = out[["gender_score_veto", "color_score_veto"]].max(axis=1).astype(np.int8)
    out["brand_tail_veto"] = (
        (out["query_has_known_brand"].eq(1))
        & (out["query_brand_mismatch"].eq(1))
        & out["score"].le(cfg["brand_score_max"])
        & out["weighted_overlap_pct_rank"].le(cfg["brand_wov_pct_max"])
        & out["last_token_in_title"].eq(0)
        & out["brand_in_query_pair"].eq(0)
    ).astype(np.int8)
    out["safe_attr_brand_veto"] = out[["gender_color_score_veto", "brand_tail_veto"]].max(axis=1).astype(np.int8)
    out["generalized_safe_full_balanced_veto"] = out[
        ["safe_attr_brand_veto", "main_accessory_veto", "package_variant_veto", "head_conflict_veto", "family_veto_balanced"]
    ].max(axis=1).astype(np.int8)
    out["generalized_safe_full_balanced_plus_compat_veto"] = out[
        [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "compatibility_veto",
            "head_conflict_veto",
            "family_veto_balanced",
        ]
    ].max(axis=1).astype(np.int8)
    out["generalized_safe_full_strict_plus_compat_veto"] = out[
        [
            "safe_attr_brand_veto",
            "main_accessory_veto",
            "package_variant_veto",
            "compatibility_veto",
            "head_conflict_veto",
            "family_veto_strict",
        ]
    ].max(axis=1).astype(np.int8)
    out["generalized_full_balanced_plus_safe_attr_brand_veto"] = out[
        ["full_veto_balanced", "safe_attr_brand_veto"]
    ].max(axis=1).astype(np.int8)
    return out


def fold_metrics(df: pd.DataFrame, col: str) -> dict:
    rows = []
    for fold, grp in df.groupby("fold5", sort=True):
        mask = grp[col].eq(1)
        n = int(mask.sum())
        errs = int(grp.loc[mask, "label"].sum())
        prec = float((1 - grp.loc[mask, "label"]).mean()) if n else math.nan
        rows.append({"fold": int(fold), "rows": n, "positive_errors": errs, "negative_precision": prec})
    use = [r for r in rows if r["rows"] > 0]
    total_rows = sum(r["rows"] for r in rows)
    total_errs = sum(r["positive_errors"] for r in rows)
    total_prec = float((total_rows - total_errs) / total_rows) if total_rows else math.nan
    min_prec = min((r["negative_precision"] for r in use), default=math.nan)
    mean_prec = float(np.mean([r["negative_precision"] for r in use])) if use else math.nan
    return {
        "rule": col,
        "rows": total_rows,
        "positive_errors": total_errs,
        "negative_precision": total_prec,
        "min_fold_precision": min_prec,
        "mean_fold_precision": mean_prec,
        "by_fold": rows,
    }


def choose_attr_config(base_eval: pd.DataFrame) -> tuple[dict, list[dict]]:
    results = []
    for gthr in ATTR_GRID["gender_score_max"]:
        for cthr in ATTR_GRID["color_score_max"]:
            for bthr in ATTR_GRID["brand_score_max"]:
                for bwov in ATTR_GRID["brand_wov_pct_max"]:
                    cfg = {
                        "gender_score_max": float(gthr),
                        "color_score_max": float(cthr),
                        "brand_score_max": float(bthr),
                        "brand_wov_pct_max": float(bwov),
                    }
                    scored = attach_attr_thresholds(base_eval, cfg)
                    metrics = fold_metrics(scored, "safe_attr_brand_veto")
                    metrics["config"] = cfg
                    results.append(metrics)

    eligible = [
        r for r in results
        if r["rows"] >= 1000
        and r["positive_errors"] <= 2
        and (not math.isnan(r["min_fold_precision"]) and r["min_fold_precision"] >= 0.985)
        and r["negative_precision"] >= 0.995
    ]
    if not eligible:
        eligible = results
    eligible.sort(
        key=lambda r: (
            float("-inf") if math.isnan(r["min_fold_precision"]) else r["min_fold_precision"],
            r["negative_precision"],
            r["rows"],
        ),
        reverse=True,
    )
    return eligible[0]["config"], results


def build_base_eval(mod209, oof: pd.DataFrame) -> pd.DataFrame:
    train_pairs = pd.read_csv(TRAIN_PAIRS, usecols=["term_id", "item_id"])
    train_pairs["term_id"] = train_pairs["term_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)
    train_pairs["fold5"] = train_pairs["term_id"].map(term_fold).astype(np.int8)

    brand_tokens = mod209.build_brand_token_set()
    needed_term_ids = set(oof["term_id"]) | set(train_pairs["term_id"])
    needed_item_ids = set(oof["item_id"]) | set(train_pairs["item_id"])
    query_info = mod209.build_query_info(needed_term_ids, brand_tokens)
    item_info = mod209.build_item_meta(needed_item_ids)

    frames = []
    for fold in range(N_FOLDS):
        fold_pairs = train_pairs[train_pairs["fold5"] != fold][["term_id", "item_id"]].copy()
        fold_valid = oof[oof["fold5"] == fold].copy()
        family_stats = mod209.build_family_stats(fold_pairs, query_info, item_info)
        rows = fold_valid[["term_id", "item_id", "label"]].copy()
        rows["id"] = np.arange(len(rows)).astype(str)
        rows["score"] = fold_valid["oof_score"].astype(np.float32).to_numpy()
        rows["title_cov_pct_rank"] = np.float32(0.5)
        eval_df = mod209.evaluate_rows(
            rows[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]],
            query_info,
            item_info,
            family_stats,
        )
        eval_df = eval_df.merge(
            fold_valid[["term_id", "item_id", "fold5"] + ATTR_COLS],
            on=["term_id", "item_id"],
            how="left",
        )
        frames.append(eval_df)
        print({"eval_fold": fold, "rows": len(eval_df)}, flush=True)

    base_eval = pd.concat(frames, ignore_index=True)
    return base_eval


def pick_candidate_variants(scored: pd.DataFrame) -> list[dict]:
    variants = [
        "family_veto_strict",
        "family_veto_balanced",
        "generalized_safe_full_balanced_veto",
        "generalized_safe_full_balanced_plus_compat_veto",
        "generalized_safe_full_strict_plus_compat_veto",
        "generalized_full_balanced_plus_safe_attr_brand_veto",
    ]
    rows = [fold_metrics(scored, col) for col in variants]
    rows.sort(
        key=lambda r: (
            float("-inf") if math.isnan(r["min_fold_precision"]) else r["min_fold_precision"],
            r["negative_precision"],
            r["rows"],
        ),
        reverse=True,
    )
    return rows


def build_test_eval(mod209, test_score_df: pd.DataFrame, cfg: dict[str, float]) -> pd.DataFrame:
    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    for col in ("id", "term_id", "item_id"):
        pairs[col] = pairs[col].astype(str)
    test = base.merge(pairs, on="id", how="inner")
    test = test[test["prediction"].eq(1)].copy()
    test = test.merge(test_score_df, on=["id", "term_id", "item_id"], how="left")
    test["score"] = test["v106_score"].astype(np.float32)
    test["label"] = -1

    train_pairs = pd.read_csv(TRAIN_PAIRS, usecols=["term_id", "item_id"])
    train_pairs["term_id"] = train_pairs["term_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)

    brand_tokens = mod209.build_brand_token_set()
    needed_term_ids = set(test["term_id"]) | set(train_pairs["term_id"])
    needed_item_ids = set(test["item_id"]) | set(train_pairs["item_id"])
    query_info = mod209.build_query_info(needed_term_ids, brand_tokens)
    item_info = mod209.build_item_meta(needed_item_ids)
    family_stats = mod209.build_family_stats(train_pairs, query_info, item_info)

    test_eval = mod209.evaluate_rows(
        test[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]],
        query_info,
        item_info,
        family_stats,
    )
    test_eval = test_eval.merge(
        test_score_df[["id"] + ATTR_COLS],
        on="id",
        how="left",
    )
    test_eval = attach_attr_thresholds(test_eval, cfg)
    return test_eval


def save_variants(test_eval: pd.DataFrame) -> list[dict]:
    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    base_pred = base["prediction"].astype(np.int8).to_numpy()
    ids = base["id"]
    variant_map = {
        "generalized_family_strict_only": "family_veto_strict",
        "generalized_family_balanced_only": "family_veto_balanced",
        "generalized_safe_full_balanced": "generalized_safe_full_balanced_veto",
        "generalized_safe_full_balanced_plus_compat": "generalized_safe_full_balanced_plus_compat_veto",
        "generalized_safe_full_strict_plus_compat": "generalized_safe_full_strict_plus_compat_veto",
        "generalized_full_balanced_plus_safe_attr_brand": "generalized_full_balanced_plus_safe_attr_brand_veto",
        "generalized_safe_attr_brand_only": "safe_attr_brand_veto",
    }
    rows = []
    test_mask = test_eval.set_index("id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, col in variant_map.items():
        pred = base.copy()
        pred["prediction"] = pred["prediction"].astype(np.int8)
        veto_ids = set(test_mask.index[test_mask[col].eq(1)])
        if veto_ids:
            row_mask = pred["id"].isin(veto_ids) & pred["prediction"].eq(1)
            pred.loc[row_mask, "prediction"] = 0
        path = OUT_DIR / f"FINAL_CANDIDATE_v106_{name}.csv"
        pred[["id", "prediction"]].to_csv(path, index=False)
        rows.append(
            {
                "variant": name,
                "removed_vs_base": int((base_pred == 1).sum() - pred["prediction"].sum()),
                "positives": int(pred["prediction"].sum()),
                "ratio": float(pred["prediction"].mean()),
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def main():
    mod209 = load_filterpack_module()
    feature_cols = json.loads(FEATURE_REPORT.read_text(encoding="utf8"))["features"]
    train_df, test_df, oof_df, test_score_df = build_or_load_scores(feature_cols)
    base_eval = build_base_eval(mod209, oof_df)
    attr_cfg, attr_search = choose_attr_config(base_eval)
    scored_eval = attach_attr_thresholds(base_eval, attr_cfg)
    candidate_metrics = pick_candidate_variants(scored_eval)
    test_eval = build_test_eval(mod209, test_score_df, attr_cfg)
    candidates = save_variants(test_eval)

    report = {
        "n_folds": N_FOLDS,
        "train_rows": int(len(train_df)),
        "train_queries": int(train_df["term_id"].nunique()),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "selected_attr_config": attr_cfg,
        "attr_search_top10": sorted(
            attr_search,
            key=lambda r: (
                float("-inf") if math.isnan(r["min_fold_precision"]) else r["min_fold_precision"],
                r["negative_precision"],
                r["rows"],
            ),
            reverse=True,
        )[:10],
        "candidate_fold_metrics": candidate_metrics,
        "test_positive_hits": {
            row["rule"]: int(test_eval[row["rule"]].sum())
            for row in candidate_metrics
        },
        "candidates": candidates,
        "score_artifacts": {
            "oof": str(OOF_OUT),
            "test_scores": str(TEST_SCORE_OUT),
            "model": str(MODEL_OUT),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
