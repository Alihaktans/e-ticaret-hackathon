"""Build V108 with route certification, source-matched calibration, and alternative-aware veto."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import re

import numpy as np
import pandas as pd


ROOT = Path(".")
BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TRAIN_FEATURES = ROOT / "data/processed/v82_v34_features.parquet"
TEST_FEATURES = ROOT / "data/processed/v21_test_features.parquet"
LEGACY_SCORE_OOF = ROOT / "data/processed/v106_generalized_oof_scores.parquet"
LEGACY_SCORE_TEST = ROOT / "data/processed/v106_generalized_test_scores.parquet"
FEATURE_REPORT = ROOT / "reports/experiments/v83_independent_pairlocal_classifier.json"

OOF_OUT = ROOT / "data/processed/v108_source_matched_oof_scores.parquet"
TEST_SCORE_OUT = ROOT / "data/processed/v108_source_matched_test_scores.parquet"
MODEL_OUT = ROOT / "models/v108_source_matched_pairlocal_classifier.cbm"

OUT_DIR = ROOT / "submissions/final_candidates_v108"
REPORT = ROOT / "reports/experiments/v108_stable_route_filterpack.json"
HANDOFF = ROOT / "reports/HANDOFF_V108_STABLE_ROUTE_FILTERPACK_2026-07-05.md"

SCRIPT_209 = ROOT / "scripts/209_make_v100_multi_filterpack.py"
SCRIPT_210 = ROOT / "scripts/210_make_v106_generalized_filterpack.py"

CHUNK = 300_000
N_FOLDS = 5

EXTRA_MODEL_COLS = [
    "candidate_count",
    "brand_count_ratio",
    "root_count_ratio",
    "category_count_ratio",
    "weighted_overlap_pct_rank",
    "weighted_overlap_delta_top",
    "total_cov_pct_rank",
    "total_cov_delta_top",
    "title_cov_pct_rank",
    "title_cov_delta_top",
    "brand_cov_pct_rank",
    "category_cov_pct_rank",
    "digit_cov_pct_rank",
    "color_cov_pct_rank",
    "size_cov_pct_rank",
    "item_train_pos_log_pct_rank",
    "brand_train_pos_log_pct_rank",
    "root_train_pos_log_pct_rank",
]

ROUTE_QUERY_FAMILIES = [
    "phone",
    "tablet",
    "vacuum",
    "watch",
    "printer",
    "console",
    "laptop",
    "monitor",
    "camera",
    "audio",
    "other",
]

ROUTE_INTENTS = [
    "device_accessory",
    "device_main",
    "education",
    "gendered_fashion",
    "general",
]

ROUTE_ITEM_FAMILIES = ROUTE_QUERY_FAMILIES

REASON_CERTIFIED = {
    "class_grade_all": ("class_grade_mismatch", None),
    "tire_size_all": ("tire_size_mismatch", None),
    "diaper_no_all": ("diaper_no_mismatch", None),
    "formula_stage_all": ("formula_stage_mismatch", None),
    "jant_certified": ("jant_mismatch", 0.18),
}

BLOCK_SPECS = [
    {
        "name": "family_balanced_router",
        "grid": {
            "score_max": [0.12, 0.15, 0.18],
            "legacy_max": [0.12, 0.15, 0.18],
            "alt_gap_min": [0.00, 0.03, 0.05],
        },
    },
    {
        "name": "gender_fashion_alt",
        "grid": {
            "score_max": [0.18, 0.22, 0.26],
            "legacy_max": [0.15, 0.18, 0.22],
            "alt_gap_min": [0.04, 0.06, 0.08, 0.10],
        },
    },
    {
        "name": "gender_general_alt",
        "grid": {
            "score_max": [0.08, 0.10, 0.12],
            "legacy_max": [0.08, 0.10, 0.12],
            "alt_gap_min": [0.06, 0.08, 0.10, 0.12],
        },
    },
    {
        "name": "main_phone_alt",
        "grid": {
            "score_max": [0.05, 0.07, 0.09, 0.11],
            "legacy_max": [0.06, 0.08, 0.10, 0.12],
            "alt_gap_min": [0.08, 0.10, 0.12, 0.15],
        },
    },
    {
        "name": "main_vacuum_alt",
        "grid": {
            "score_max": [0.12, 0.15, 0.18, 0.22],
            "legacy_max": [0.12, 0.15, 0.18, 0.22],
            "alt_gap_min": [0.03, 0.05, 0.08, 0.10],
        },
    },
    {
        "name": "main_tablet_alt",
        "grid": {
            "score_max": [0.16, 0.20, 0.24, 0.30],
            "legacy_max": [0.18, 0.22, 0.26, 0.32],
            "alt_gap_min": [0.04, 0.06, 0.08],
        },
    },
    {
        "name": "accessory_vacuum_router",
        "grid": {
            "score_max": [0.20, 0.25, 0.30, 0.35],
            "legacy_max": [0.22, 0.28, 0.34],
            "alt_gap_min": [0.00, 0.03, 0.05],
        },
    },
    {
        "name": "compat_phone_router",
        "grid": {
            "score_max": [0.06, 0.08, 0.10, 0.12],
            "legacy_max": [0.06, 0.08, 0.10, 0.12],
            "alt_gap_min": [0.03, 0.05, 0.08],
        },
    },
    {
        "name": "compat_tablet_router",
        "grid": {
            "score_max": [0.08, 0.10, 0.12, 0.15],
            "legacy_max": [0.08, 0.10, 0.12, 0.15],
            "alt_gap_min": [0.03, 0.05, 0.08],
        },
    },
    {
        "name": "person_count_router",
        "grid": {
            "score_max": [0.10, 0.14, 0.18, 0.22],
            "legacy_max": [0.10, 0.14, 0.18, 0.22],
            "alt_gap_min": [0.05, 0.08, 0.10, 0.12],
        },
    },
]

VARIANT_TARGETS = {
    "stable_top": {
        "target_precision": 0.9725,
        "target_min_fold": 0.9000,
        "allow_family_balanced": True,
        "min_block_rows": 35,
    },
    "stable_balanced": {
        "target_precision": 0.9780,
        "target_min_fold": 0.9300,
        "allow_family_balanced": True,
        "min_block_rows": 35,
    },
    "stable_precision": {
        "target_precision": 0.9850,
        "target_min_fold": 0.9600,
        "allow_family_balanced": False,
        "min_block_rows": 25,
    },
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_source_bucket(source: str) -> str:
    source = str(source or "")
    if source == "train_positive":
        return "positive"
    match = re.search(r"_rank_(\d+)$", source)
    if not match:
        return "other_neg"
    rank = int(match.group(1))
    if rank <= 5:
        return "rank_01_05"
    if rank <= 10:
        return "rank_06_10"
    if rank <= 15:
        return "rank_11_15"
    if rank <= 20:
        return "rank_16_20"
    return "rank_21_plus"


def build_route_maps(mod209, needed_term_ids: set[str], needed_item_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    brand_tokens = mod209.build_brand_token_set()
    query_info = mod209.build_query_info(needed_term_ids, brand_tokens)
    item_info = mod209.build_item_meta(needed_item_ids)

    query_rows = []
    for term_id, q in query_info.items():
        row = {
            "term_id": str(term_id),
            "route_key": f"{q['intent']}:{q['device_family']}",
            "query_has_compat_tokens": int(bool(q["compat_tokens"])),
            "query_is_accessory_query": int(bool(q["is_accessory_query"])),
            "query_has_main_signal": int(bool(q["has_main_device_signal"])),
        }
        for intent in ROUTE_INTENTS:
            row[f"intent_{intent}"] = int(q["intent"] == intent)
        for family in ROUTE_QUERY_FAMILIES:
            row[f"qfam_{family}"] = int(q["device_family"] == family)
        query_rows.append(row)

    item_rows = []
    for item_id, it in item_info.items():
        row = {
            "item_id": str(item_id),
            "item_is_accessory_meta": int(bool(it["is_accessory"])),
            "item_is_main_device_meta": int(bool(it["is_main_device"])),
        }
        for family in ROUTE_ITEM_FAMILIES:
            row[f"ifam_{family}"] = int(it["device_family"] == family)
        item_rows.append(row)

    return pd.DataFrame(query_rows), pd.DataFrame(item_rows)


def enrich_route_features(df: pd.DataFrame, query_map: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(query_map, on="term_id", how="left").merge(item_map, on="item_id", how="left")
    route_cols = [
        "query_has_compat_tokens",
        "query_is_accessory_query",
        "query_has_main_signal",
        "item_is_accessory_meta",
        "item_is_main_device_meta",
    ]
    route_cols += [f"intent_{x}" for x in ROUTE_INTENTS]
    route_cols += [f"qfam_{x}" for x in ROUTE_QUERY_FAMILIES]
    route_cols += [f"ifam_{x}" for x in ROUTE_ITEM_FAMILIES]
    for col in route_cols:
        out[col] = out[col].fillna(0).astype(np.int8)
    out["route_key"] = out["route_key"].fillna("general:other")
    return out


def build_sample_weights(train: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(train), dtype=np.float32)
    pos_mask = train["label"].eq(1).to_numpy()

    train["source_bucket"] = train["candidate_source"].map(parse_source_bucket)
    neg = train.loc[~pos_mask, ["route_key", "source_bucket"]].copy()
    neg["cell"] = neg["route_key"].astype(str) + "|" + neg["source_bucket"].astype(str)
    neg_counts = neg["cell"].value_counts()
    neg_target = float(neg_counts.quantile(0.75)) if len(neg_counts) else 1.0
    neg_weight_map = {
        cell: float(np.clip(math.sqrt(neg_target / cnt), 0.70, 2.50))
        for cell, cnt in neg_counts.items()
    }

    pos = train.loc[pos_mask, ["route_key"]].copy()
    pos_counts = pos["route_key"].value_counts()
    pos_target = float(pos_counts.quantile(0.75)) if len(pos_counts) else 1.0
    pos_weight_map = {
        route_key: float(np.clip(math.sqrt(pos_target / cnt), 0.80, 1.80))
        for route_key, cnt in pos_counts.items()
    }

    neg_cells = (train.loc[~pos_mask, "route_key"].astype(str) + "|" + train.loc[~pos_mask, "source_bucket"].astype(str)).tolist()
    weights[~pos_mask] = np.asarray([neg_weight_map.get(cell, 1.0) for cell in neg_cells], dtype=np.float32)
    weights[pos_mask] = np.asarray(
        [pos_weight_map.get(route_key, 1.0) for route_key in train.loc[pos_mask, "route_key"].astype(str).tolist()],
        dtype=np.float32,
    )
    train.drop(columns=["source_bucket"], inplace=True)
    return weights


def build_or_load_source_matched_scores(mod210, mod209) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_feature_cols = json.loads(FEATURE_REPORT.read_text(encoding="utf8"))["features"]
    feature_cols = list(dict.fromkeys(base_feature_cols + EXTRA_MODEL_COLS))

    route_feature_cols = [
        "query_has_compat_tokens",
        "query_is_accessory_query",
        "query_has_main_signal",
        "item_is_accessory_meta",
        "item_is_main_device_meta",
    ]
    route_feature_cols += [f"intent_{x}" for x in ROUTE_INTENTS]
    route_feature_cols += [f"qfam_{x}" for x in ROUTE_QUERY_FAMILIES]
    route_feature_cols += [f"ifam_{x}" for x in ROUTE_ITEM_FAMILIES]
    model_feature_cols = feature_cols + route_feature_cols

    train_cols = ["term_id", "item_id", "label", "candidate_source"] + feature_cols
    test_cols = ["id", "term_id", "item_id"] + feature_cols
    train = pd.read_parquet(TRAIN_FEATURES, columns=train_cols)
    test = pd.read_parquet(TEST_FEATURES, columns=test_cols)
    for col in ("term_id", "item_id"):
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
    test["id"] = test["id"].astype(str)
    train["candidate_source"] = train["candidate_source"].astype(str)
    train["fold5"] = train["term_id"].map(mod210.term_fold).astype(np.int8)

    query_map, item_map = build_route_maps(
        mod209,
        set(train["term_id"]).union(set(test["term_id"])),
        set(train["item_id"]).union(set(test["item_id"])),
    )
    train = enrich_route_features(train, query_map, item_map)
    test = enrich_route_features(test, query_map, item_map)

    if OOF_OUT.exists() and TEST_SCORE_OUT.exists():
        oof = pd.read_parquet(OOF_OUT)
        score = pd.read_parquet(TEST_SCORE_OUT)
        if len(oof) == len(train) and len(score) == len(test):
            return oof, score, model_feature_cols

    sample_weight = build_sample_weights(train)
    oof_scores = np.empty(len(train), dtype=np.float32)
    best_iters = []
    for fold in range(N_FOLDS):
        tr = train[train["fold5"] != fold].copy()
        va = train[train["fold5"] == fold].copy()
        wtr = sample_weight[train["fold5"] != fold]
        xtr = tr[model_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        xva = va[model_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        ytr = tr["label"].astype("int8").to_numpy()
        yva = va["label"].astype("int8").to_numpy()
        params = dict(mod210.MODEL_PARAMS)
        model = mod210.CatBoostClassifier(**params)
        try:
            model.fit(xtr, ytr, sample_weight=wtr, eval_set=(xva, yva), use_best_model=True)
        except Exception as exc:
            print("GPU fallback weighted fold:", repr(exc), flush=True)
            params["task_type"] = "CPU"
            params["thread_count"] = max(1, (os.cpu_count() or 2) - 1)
            params.pop("devices", None)
            model = mod210.CatBoostClassifier(**params)
            model.fit(xtr, ytr, sample_weight=wtr, eval_set=(xva, yva), use_best_model=True)
        best_iter = model.get_best_iteration()
        if best_iter is None or best_iter <= 0:
            best_iter = params["iterations"]
        best_iters.append(best_iter)
        oof_scores[va.index.to_numpy()] = model.predict_proba(xva)[:, 1].astype(np.float32)
        print({"v108_fold": fold, "rows": len(va), "best_iter": best_iter}, flush=True)

    full_iter = max(300, int(round(float(np.median(best_iters)) * 1.05)))
    params = dict(mod210.MODEL_PARAMS)
    params["iterations"] = full_iter
    params.pop("od_type", None)
    params.pop("od_wait", None)
    model = mod210.CatBoostClassifier(**params)
    xall = train[model_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
    yall = train["label"].astype("int8").to_numpy()
    try:
        model.fit(xall, yall, sample_weight=sample_weight)
    except Exception as exc:
        print("GPU fallback full:", repr(exc), flush=True)
        params["task_type"] = "CPU"
        params["thread_count"] = max(1, 6)
        params.pop("devices", None)
        model = mod210.CatBoostClassifier(**params)
        model.fit(xall, yall, sample_weight=sample_weight)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    test_scores = mod210.predict_in_chunks(model, test, model_feature_cols)

    oof = train[["term_id", "item_id", "label", "fold5", "route_key", "candidate_source"]].copy()
    oof["v108_score"] = oof_scores
    score = test[["id", "term_id", "item_id", "route_key", "title_cov_pct_rank"]].copy()
    score["v108_score"] = test_scores
    OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OOF_OUT, index=False)
    score.to_parquet(TEST_SCORE_OUT, index=False)
    return oof, score, model_feature_cols


def build_base_eval(mod209, mod210, oof_new: pd.DataFrame) -> pd.DataFrame:
    train_pairs = pd.read_csv(TRAIN_PAIRS, usecols=["term_id", "item_id"])
    train_pairs["term_id"] = train_pairs["term_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)
    train_pairs["fold5"] = train_pairs["term_id"].map(mod210.term_fold).astype(np.int8)

    legacy_oof = pd.read_parquet(LEGACY_SCORE_OOF)
    legacy_oof["term_id"] = legacy_oof["term_id"].astype(str)
    legacy_oof["item_id"] = legacy_oof["item_id"].astype(str)

    brand_tokens = mod209.build_brand_token_set()
    needed_term_ids = set(oof_new["term_id"]) | set(train_pairs["term_id"])
    needed_item_ids = set(oof_new["item_id"]) | set(train_pairs["item_id"])
    query_info = mod209.build_query_info(needed_term_ids, brand_tokens)
    item_info = mod209.build_item_meta(needed_item_ids)

    frames = []
    for fold in range(N_FOLDS):
        fold_pairs = train_pairs[train_pairs["fold5"] != fold][["term_id", "item_id"]].copy()
        fold_valid = oof_new[oof_new["fold5"] == fold].copy()
        family_stats = mod209.build_family_stats(fold_pairs, query_info, item_info)
        rows = fold_valid[["term_id", "item_id", "label"]].copy()
        rows["id"] = np.arange(len(rows)).astype(str)
        rows["score"] = fold_valid["v108_score"].astype(np.float32).to_numpy()
        rows["title_cov_pct_rank"] = np.float32(0.5)
        eval_df = mod209.evaluate_rows(
            rows[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]],
            query_info,
            item_info,
            family_stats,
        )
        eval_df = eval_df.merge(
            fold_valid[["term_id", "item_id", "fold5", "candidate_source", "route_key", "v108_score"]],
            on=["term_id", "item_id"],
            how="left",
        )
        eval_df = eval_df.merge(
            legacy_oof[["term_id", "item_id", "oof_score"]].rename(columns={"oof_score": "legacy_score"}),
            on=["term_id", "item_id"],
            how="left",
        )
        frames.append(eval_df)
        print({"v108_eval_fold": fold, "rows": len(eval_df)}, flush=True)

    return pd.concat(frames, ignore_index=True)


def build_test_eval(mod209, test_new: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    for col in ("id", "term_id", "item_id"):
        pairs[col] = pairs[col].astype(str)
    legacy_test = pd.read_parquet(LEGACY_SCORE_TEST)
    for col in ("id", "term_id", "item_id"):
        legacy_test[col] = legacy_test[col].astype(str)
    test = base.merge(pairs, on="id", how="inner")
    test = test[test["prediction"].eq(1)].copy()
    test = test.merge(
        test_new[["id", "term_id", "item_id", "v108_score", "title_cov_pct_rank"]],
        on=["id", "term_id", "item_id"],
        how="left",
    )
    test = test.merge(
        legacy_test[["id", "term_id", "item_id", "v106_score"]].rename(columns={"v106_score": "legacy_score"}),
        on=["id", "term_id", "item_id"],
        how="left",
    )
    test["score"] = test["v108_score"].astype(np.float32)
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
        test[["id", "v108_score", "legacy_score"]],
        on="id",
        how="left",
    )
    return test_eval


def add_alt_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["legacy_score"] = out["legacy_score"].fillna(out["score"]).astype(np.float32)
    for rule in [
        "gender_veto",
        "main_accessory_veto",
        "package_variant_veto",
        "compatibility_veto",
        "family_veto_balanced",
        "family_veto_strict",
    ]:
        safe_max = out["score"].where(out[rule].eq(0)).groupby(out["term_id"]).transform("max")
        out[f"best_safe_{rule}_score"] = safe_max.fillna(-1.0).astype(np.float32)
        out[f"alt_gap_{rule}"] = (out[f"best_safe_{rule}_score"] - out["score"]).astype(np.float32)
    out["reason_text"] = out["reason_text"].fillna("")
    return out


def fold_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    rows = int(mask.sum())
    errs = int(df.loc[mask, "label"].sum())
    prec = float((rows - errs) / rows) if rows else float("nan")
    by_fold = []
    if rows:
        for fold, grp in df.loc[mask].groupby("fold5", sort=True):
            n = int(len(grp))
            e = int(grp["label"].sum())
            p = float((n - e) / n)
            by_fold.append({"fold": int(fold), "rows": n, "positive_errors": e, "negative_precision": p})
    return {
        "rows": rows,
        "positive_errors": errs,
        "negative_precision": prec,
        "min_fold_precision": min((r["negative_precision"] for r in by_fold), default=float("nan")),
        "by_fold": by_fold,
    }


def reason_mask(df: pd.DataFrame, token: str, score_max: float | None) -> np.ndarray:
    mask = df["reason_text"].str.contains(token).to_numpy(copy=True)
    if score_max is not None:
        mask &= df["score"].le(score_max).to_numpy()
    return mask


def block_mask(df: pd.DataFrame, name: str, cfg: dict[str, float]) -> np.ndarray:
    if name == "family_balanced_router":
        return (
            df["family_veto_balanced"].eq(1).to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_family_veto_balanced"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "gender_fashion_alt":
        return (
            df["gender_veto"].eq(1).to_numpy()
            & df["query_intent"].eq("gendered_fashion").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_gender_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "gender_general_alt":
        return (
            df["gender_veto"].eq(1).to_numpy()
            & df["query_intent"].eq("general").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_gender_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "main_phone_alt":
        return (
            df["main_accessory_veto"].eq(1).to_numpy()
            & df["query_intent"].eq("device_main").to_numpy()
            & df["query_device_family"].eq("phone").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_main_accessory_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "main_vacuum_alt":
        return (
            df["main_accessory_veto"].eq(1).to_numpy()
            & df["query_intent"].eq("device_main").to_numpy()
            & df["query_device_family"].eq("vacuum").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_main_accessory_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "main_tablet_alt":
        return (
            df["main_accessory_veto"].eq(1).to_numpy()
            & df["query_intent"].eq("device_main").to_numpy()
            & df["query_device_family"].eq("tablet").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_main_accessory_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "accessory_vacuum_router":
        return (
            df["main_accessory_veto"].eq(1).to_numpy()
            & df["query_intent"].eq("device_accessory").to_numpy()
            & df["query_device_family"].eq("vacuum").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_main_accessory_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "compat_phone_router":
        return (
            df["compatibility_veto"].eq(1).to_numpy()
            & df["query_device_family"].eq("phone").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_compatibility_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "compat_tablet_router":
        return (
            df["compatibility_veto"].eq(1).to_numpy()
            & df["query_device_family"].eq("tablet").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_compatibility_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    if name == "person_count_router":
        return (
            df["reason_text"].str.contains("person_count_mismatch").to_numpy()
            & df["score"].le(cfg["score_max"]).to_numpy()
            & df["legacy_score"].le(cfg["legacy_max"]).to_numpy()
            & df["alt_gap_package_variant_veto"].ge(cfg["alt_gap_min"]).to_numpy()
        )
    raise KeyError(name)


def enumerate_grid(spec: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = list(spec)
    vals = [spec[k] for k in keys]
    out = []
    for combo in product(*vals):
        out.append({k: float(v) for k, v in zip(keys, combo)})
    return out


def search_blocks(base_eval: pd.DataFrame, test_eval: pd.DataFrame) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    block_results = []
    base_masks: dict[str, np.ndarray] = {}
    test_masks: dict[str, np.ndarray] = {}

    fixed_blocks = [
        {"name": "family_strict_all", "cfg": None, "mask_fn": lambda df: df["family_veto_strict"].eq(1).to_numpy()},
    ]
    for block_name, (reason_token, score_max) in REASON_CERTIFIED.items():
        fixed_blocks.append(
            {
                "name": block_name,
                "cfg": None,
                "mask_fn": lambda df, token=reason_token, score_limit=score_max: reason_mask(df, token, score_limit),
            }
        )

    for block in fixed_blocks:
        mask_oof = block["mask_fn"](base_eval)
        mask_test = block["mask_fn"](test_eval)
        metrics = fold_metrics(base_eval, mask_oof)
        block_results.append(
            {
                "name": block["name"],
                "config": None,
                "oof_metrics": metrics,
                "test_positive_hits": int(mask_test.sum()),
                "kind": "fixed",
            }
        )
        base_masks[block["name"]] = mask_oof
        test_masks[block["name"]] = mask_test

    for spec in BLOCK_SPECS:
        best = None
        for cfg in enumerate_grid(spec["grid"]):
            mask_oof = block_mask(base_eval, spec["name"], cfg)
            metrics = fold_metrics(base_eval, mask_oof)
            if metrics["rows"] < 15:
                continue
            mask_test = block_mask(test_eval, spec["name"], cfg)
            row = {
                "name": spec["name"],
                "config": cfg,
                "oof_metrics": metrics,
                "test_positive_hits": int(mask_test.sum()),
                "kind": "searched",
            }
            if best is None:
                best = row
                continue
            cur = row["oof_metrics"]
            prv = best["oof_metrics"]
            cur_key = (
                float("-inf") if math.isnan(cur["min_fold_precision"]) else cur["min_fold_precision"],
                cur["negative_precision"],
                cur["rows"],
            )
            prv_key = (
                float("-inf") if math.isnan(prv["min_fold_precision"]) else prv["min_fold_precision"],
                prv["negative_precision"],
                prv["rows"],
            )
            if cur_key > prv_key:
                best = row
        if best is not None:
            base_masks[spec["name"]] = block_mask(base_eval, spec["name"], best["config"])
            test_masks[spec["name"]] = block_mask(test_eval, spec["name"], best["config"])
            block_results.append(best)
    return block_results, base_masks, test_masks


def choose_variant(
    name: str,
    target_cfg: dict,
    block_results: list[dict],
    base_masks: dict[str, np.ndarray],
    test_masks: dict[str, np.ndarray],
    base_eval: pd.DataFrame,
) -> dict:
    selected = []
    union_mask = np.zeros(len(base_eval), dtype=bool)

    fixed_order = ["family_strict_all", "class_grade_all", "tire_size_all", "diaper_no_all", "formula_stage_all", "jant_certified"]
    if target_cfg["allow_family_balanced"] and "family_balanced_router" in base_masks:
        fixed_order.append("family_balanced_router")

    by_name = {row["name"]: row for row in block_results}
    for block_name in fixed_order:
        if block_name not in base_masks:
            continue
        candidate = union_mask | base_masks[block_name]
        metrics = fold_metrics(base_eval, candidate)
        if metrics["rows"] == 0:
            continue
        if metrics["negative_precision"] >= target_cfg["target_precision"] and metrics["min_fold_precision"] >= target_cfg["target_min_fold"]:
            union_mask = candidate
            selected.append(block_name)

    eligible = []
    for row in block_results:
        if row["name"] in selected:
            continue
        if row["oof_metrics"]["rows"] < target_cfg["min_block_rows"]:
            continue
        if row["name"] == "family_balanced_router" and not target_cfg["allow_family_balanced"]:
            continue
        eligible.append(row)
    eligible.sort(
        key=lambda r: (
            r["test_positive_hits"],
            float("-inf") if math.isnan(r["oof_metrics"]["min_fold_precision"]) else r["oof_metrics"]["min_fold_precision"],
            r["oof_metrics"]["negative_precision"],
        ),
        reverse=True,
    )

    for row in eligible:
        block_name = row["name"]
        candidate = union_mask | base_masks[block_name]
        metrics = fold_metrics(base_eval, candidate)
        if metrics["rows"] == 0:
            continue
        if metrics["negative_precision"] >= target_cfg["target_precision"] and metrics["min_fold_precision"] >= target_cfg["target_min_fold"]:
            union_mask = candidate
            selected.append(block_name)

    union_metrics = fold_metrics(base_eval, union_mask)
    test_union = np.zeros(len(next(iter(test_masks.values()))), dtype=bool)
    for block_name in selected:
        test_union |= test_masks[block_name]
    return {
        "variant": name,
        "selected_blocks": selected,
        "oof_metrics": union_metrics,
        "test_positive_hits": int(test_union.sum()),
        "test_mask": test_union,
    }


def save_candidate(mask_ids: set[str], name: str) -> dict:
    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    base_pred = base["prediction"].astype(np.int8).to_numpy()
    pred = base.copy()
    pred["prediction"] = pred["prediction"].astype(np.int8)
    row_mask = pred["id"].isin(mask_ids) & pred["prediction"].eq(1)
    pred.loc[row_mask, "prediction"] = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"FINAL_CANDIDATE_v108_{name}.csv"
    pred[["id", "prediction"]].to_csv(path, index=False)
    new_pred = pred["prediction"].to_numpy(np.int8)
    return {
        "variant": name,
        "removed_vs_base": int((base_pred == 1).sum() - new_pred.sum()),
        "positives": int(new_pred.sum()),
        "ratio": float(new_pred.mean()),
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    mod209 = load_module(SCRIPT_209, "mod209")
    mod210 = load_module(SCRIPT_210, "mod210")
    oof_new, test_new, model_feature_cols = build_or_load_source_matched_scores(mod210, mod209)
    base_eval = add_alt_features(build_base_eval(mod209, mod210, oof_new))
    test_eval = add_alt_features(build_test_eval(mod209, test_new))
    block_results, base_masks, test_masks = search_blocks(base_eval, test_eval)

    variants = []
    candidates = []
    for variant_name, target_cfg in VARIANT_TARGETS.items():
        variant = choose_variant(variant_name, target_cfg, block_results, base_masks, test_masks, base_eval)
        variants.append(
            {
                "variant": variant_name,
                "selected_blocks": variant["selected_blocks"],
                "oof_metrics": variant["oof_metrics"],
                "test_positive_hits": variant["test_positive_hits"],
            }
        )
        mask_ids = set(test_eval.loc[variant["test_mask"], "id"])
        candidates.append(save_candidate(mask_ids, variant_name))

    report = {
        "base_file": str(BASE),
        "base_positives": int(pd.read_csv(BASE)["prediction"].sum()),
        "score_artifacts": {
            "oof": str(OOF_OUT),
            "test_scores": str(TEST_SCORE_OUT),
            "model": str(MODEL_OUT),
        },
        "model_feature_count": int(len(model_feature_cols)),
        "model_features": model_feature_cols,
        "block_results": block_results,
        "variants": variants,
        "candidates": candidates,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")

    lines = [
        "# V108 Stable Route Filterpack",
        "",
        "Date: 2026-07-05",
        "Branch: `hybrid-reranker-v1`",
        "",
        "## Model",
        "",
        "- Source-matched CatBoost calibrator with route features and stable rank features.",
        "- Rule application is restricted by route certification, model confidence, and alternative-aware veto.",
        "",
        "## Variants",
        "",
    ]
    for row in variants:
        m = row["oof_metrics"]
        lines.extend(
            [
                f"- `{row['variant']}`",
                f"  - selected blocks: `{', '.join(row['selected_blocks'])}`",
                f"  - OOF rows: `{m['rows']}`",
                f"  - OOF positive errors: `{m['positive_errors']}`",
                f"  - OOF negative precision: `{m['negative_precision']:.4f}`",
                f"  - OOF min fold precision: `{m['min_fold_precision']:.4f}`",
                f"  - test hits: `{row['test_positive_hits']}`",
            ]
        )
    HANDOFF.write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
