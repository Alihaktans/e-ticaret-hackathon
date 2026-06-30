from __future__ import annotations

import sys
import gc
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR, EXPERIMENT_REPORTS_DIR
from trendyol.text_preprocess import normalize_text, extract_root_category


RUN_NAME = "v6_dual_e5_minilm"

TRAIN_PAIRS_PATH = PROCESSED_DIR / "train_pairs_v4_semantic_hard.parquet"
TRAIN_ENRICHED_PATH = PROCESSED_DIR / "train_enriched_v4_semantic_hard.parquet"
CANDIDATE_PATH = PROCESSED_DIR / "v4_semantic_hard_negative_candidates.parquet"

# MiniLM embeddings
MINI_TERM_EMB_PATH = PROCESSED_DIR / "minilm_all_terms_emb.npy"
MINI_ITEM_EMB_PATH = PROCESSED_DIR / "minilm_all_items_emb.npy"
MINI_TERM_IDS_PATH = PROCESSED_DIR / "minilm_all_term_ids.parquet"
MINI_ITEM_IDS_PATH = PROCESSED_DIR / "minilm_all_item_ids.parquet"

# E5 embeddings
E5_TERM_EMB_PATH = PROCESSED_DIR / "v5_e5base_all_terms_emb.npy"
E5_ITEM_EMB_PATH = PROCESSED_DIR / "v5_e5base_all_items_emb.npy"
E5_TERM_IDS_PATH = PROCESSED_DIR / "v5_e5base_all_term_ids.parquet"
E5_ITEM_IDS_PATH = PROCESSED_DIR / "v5_e5base_all_item_ids.parquet"

# Existing test score caches
MINI_TEST_SCORE_PATH = PROCESSED_DIR / "embedding_v1_minilm_topratio_008_scores.parquet"
E5_TEST_SCORE_PATH = PROCESSED_DIR / "embedding_v5_e5base_test_scores.parquet"

TRAIN_FRAME_PATH = PROCESSED_DIR / f"{RUN_NAME}_train_frame.parquet"
TEST_FRAME_PATH = PROCESSED_DIR / f"{RUN_NAME}_test_frame.parquet"

CV_RAW_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_realistic_cv_raw.csv"
CV_SUMMARY_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_realistic_cv_summary.csv"

FINAL_PROBA_PATH = PROCESSED_DIR / f"{RUN_NAME}_full_test_proba.parquet"
FINAL_SUMMARY_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_full_submission_summary.csv"
REPORT_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_report.json"

SCORE_CHUNK_SIZE = 100_000

CV_FOLD_SEEDS = [4242, 2026, 777]
SEMANTIC_WEIGHTS = [0.20, 0.35, 0.50]
CV_THRESHOLDS = np.round(np.arange(0.84, 0.991, 0.005), 3)

VALID_TERM_SIZE = 0.20
MAX_NEG_PER_TERM = 100

FINAL_SEEDS = [42, 2026, 777]
FINAL_ITERATIONS = 350

FINAL_THRESHOLDS = np.round(np.arange(0.88, 0.981, 0.005), 3)


TEXT_NUMERIC_FEATURES = [
    "query_char_len",
    "title_char_len",
    "category_char_len",
    "query_token_count",
    "title_token_count",
    "category_token_count",
    "qt_overlap_count",
    "qt_overlap_query_ratio",
    "qt_overlap_title_ratio",
    "qc_overlap_count",
    "qc_overlap_query_ratio",
    "qc_overlap_category_ratio",
    "query_in_title",
    "title_in_query",
    "brand_in_query",
]

CAT_FEATURES = [
    "brand",
    "root_category",
    "gender",
    "age_group",
]


def token_set(text: object) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    return {t for t in text.split() if len(t) >= 2}


def required_files() -> None:
    paths = [
        TRAIN_PAIRS_PATH,
        TRAIN_ENRICHED_PATH,
        CANDIDATE_PATH,
        MINI_TERM_EMB_PATH,
        MINI_ITEM_EMB_PATH,
        MINI_TERM_IDS_PATH,
        MINI_ITEM_IDS_PATH,
        E5_TERM_EMB_PATH,
        E5_ITEM_EMB_PATH,
        E5_TERM_IDS_PATH,
        E5_ITEM_IDS_PATH,
        MINI_TEST_SCORE_PATH,
        E5_TEST_SCORE_PATH,
        RAW_DIR / "terms.csv",
        RAW_DIR / "items.csv",
        RAW_DIR / "sample_submission.csv",
    ]

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")


def load_embedding_index(term_ids_path: Path, item_ids_path: Path):
    term_ids = pd.read_parquet(term_ids_path)
    item_ids = pd.read_parquet(item_ids_path)

    term_ids["term_id"] = term_ids["term_id"].astype(str)
    item_ids["item_id"] = item_ids["item_id"].astype(str)

    term_to_idx = dict(zip(term_ids["term_id"], range(len(term_ids))))
    item_to_idx = dict(zip(item_ids["item_id"], range(len(item_ids))))

    return term_to_idx, item_to_idx


def score_pairs_with_embedding(
    pairs: pd.DataFrame,
    term_emb_path: Path,
    item_emb_path: Path,
    term_ids_path: Path,
    item_ids_path: Path,
    score_col: str,
) -> np.ndarray:
    pairs = pairs.copy()
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    term_to_idx, item_to_idx = load_embedding_index(term_ids_path, item_ids_path)

    term_emb = np.load(term_emb_path, mmap_mode="r")
    item_emb = np.load(item_emb_path, mmap_mode="r")

    scores = np.empty(len(pairs), dtype=np.float32)

    for start in tqdm(range(0, len(pairs), SCORE_CHUNK_SIZE), desc=f"score {score_col}"):
        end = min(start + SCORE_CHUNK_SIZE, len(pairs))
        chunk = pairs.iloc[start:end]

        term_idx = chunk["term_id"].map(term_to_idx).to_numpy(dtype=np.int64)
        item_idx = chunk["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)

        a = np.asarray(term_emb[term_idx], dtype=np.float32)
        b = np.asarray(item_emb[item_idx], dtype=np.float32)

        scores[start:end] = np.sum(a * b, axis=1)

        del a, b
        gc.collect()

    return scores


def add_rank_features(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    df = df.copy()

    g = df.groupby("term_id")[score_col]

    df[f"{prefix}_candidate_count"] = g.transform("size").astype(np.float32)
    df[f"{prefix}_score_mean"] = g.transform("mean").astype(np.float32)
    df[f"{prefix}_score_std"] = g.transform("std").fillna(0).astype(np.float32)
    df[f"{prefix}_score_max"] = g.transform("max").astype(np.float32)
    df[f"{prefix}_score_min"] = g.transform("min").astype(np.float32)
    df[f"{prefix}_score_range"] = (df[f"{prefix}_score_max"] - df[f"{prefix}_score_min"]).astype(np.float32)

    df[f"{prefix}_score_z"] = (
        (df[score_col] - df[f"{prefix}_score_mean"]) / (df[f"{prefix}_score_std"] + 1e-6)
    ).astype(np.float32)

    df[f"{prefix}_rank_desc"] = g.rank(method="first", ascending=False).astype(np.float32)

    df[f"{prefix}_rank_pct"] = (
        (df[f"{prefix}_rank_desc"] - 1.0) / (df[f"{prefix}_candidate_count"] - 1.0).clip(lower=1.0)
    ).astype(np.float32)

    df[f"{prefix}_gap_to_top"] = (df[f"{prefix}_score_max"] - df[score_col]).astype(np.float32)

    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["query", "title", "category", "brand", "gender", "age_group"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").map(normalize_text)

    if "root_category" not in df.columns:
        df["root_category"] = df["category"].map(extract_root_category)
    else:
        df["root_category"] = df["root_category"].fillna("").map(normalize_text)

    q_tokens = df["query"].map(token_set)
    t_tokens = df["title"].map(token_set)
    c_tokens = df["category"].map(token_set)

    qt_inter = [q & t for q, t in zip(q_tokens, t_tokens)]
    qc_inter = [q & c for q, c in zip(q_tokens, c_tokens)]

    df["query_char_len"] = df["query"].str.len().astype(np.float32)
    df["title_char_len"] = df["title"].str.len().astype(np.float32)
    df["category_char_len"] = df["category"].str.len().astype(np.float32)

    df["query_token_count"] = q_tokens.map(len).astype(np.float32)
    df["title_token_count"] = t_tokens.map(len).astype(np.float32)
    df["category_token_count"] = c_tokens.map(len).astype(np.float32)

    df["qt_overlap_count"] = pd.Series(qt_inter).map(len).astype(np.float32)
    df["qt_overlap_query_ratio"] = (
        df["qt_overlap_count"] / df["query_token_count"].clip(lower=1)
    ).astype(np.float32)
    df["qt_overlap_title_ratio"] = (
        df["qt_overlap_count"] / df["title_token_count"].clip(lower=1)
    ).astype(np.float32)

    df["qc_overlap_count"] = pd.Series(qc_inter).map(len).astype(np.float32)
    df["qc_overlap_query_ratio"] = (
        df["qc_overlap_count"] / df["query_token_count"].clip(lower=1)
    ).astype(np.float32)
    df["qc_overlap_category_ratio"] = (
        df["qc_overlap_count"] / df["category_token_count"].clip(lower=1)
    ).astype(np.float32)

    df["query_in_title"] = [
        int(q != "" and q in t)
        for q, t in zip(df["query"], df["title"])
    ]

    df["title_in_query"] = [
        int(t != "" and t in q)
        for q, t in zip(df["query"], df["title"])
    ]

    df["brand_in_query"] = [
        int(b != "" and b in q)
        for q, b in zip(df["query"], df["brand"])
    ]

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").replace("", "unknown").astype(str)

    return df


def add_dual_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_rank_features(df, "e5_score", "e5")
    df = add_rank_features(df, "mini_score", "mini")

    df["score_diff_e5_minus_mini"] = (df["e5_score"] - df["mini_score"]).astype(np.float32)
    df["score_product"] = (df["e5_score"] * df["mini_score"]).astype(np.float32)
    df["score_mean_dual"] = ((df["e5_score"] + df["mini_score"]) / 2.0).astype(np.float32)

    df["rank_pct_mean_dual"] = ((df["e5_rank_pct"] + df["mini_rank_pct"]) / 2.0).astype(np.float32)
    df["rank_pct_diff_e5_minus_mini"] = (df["e5_rank_pct"] - df["mini_rank_pct"]).astype(np.float32)

    df["score_z_mean_dual"] = ((df["e5_score_z"] + df["mini_score_z"]) / 2.0).astype(np.float32)
    df["score_z_diff_e5_minus_mini"] = (df["e5_score_z"] - df["mini_score_z"]).astype(np.float32)

    df["both_top_5pct"] = ((df["e5_rank_pct"] <= 0.05) & (df["mini_rank_pct"] <= 0.05)).astype(np.int8)
    df["both_top_10pct"] = ((df["e5_rank_pct"] <= 0.10) & (df["mini_rank_pct"] <= 0.10)).astype(np.int8)
    df["either_top_5pct"] = ((df["e5_rank_pct"] <= 0.05) | (df["mini_rank_pct"] <= 0.05)).astype(np.int8)

    df = add_text_features(df)

    numeric_cols = get_numeric_features()

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)

    return df


def get_numeric_features() -> list[str]:
    rank_base = [
        "candidate_count",
        "score_mean",
        "score_std",
        "score_max",
        "score_min",
        "score_range",
        "score_z",
        "rank_desc",
        "rank_pct",
        "gap_to_top",
    ]

    features = [
        "e5_score",
        "mini_score",
    ]

    for prefix in ["e5", "mini"]:
        features.extend([f"{prefix}_{x}" for x in rank_base])

    features.extend(
        [
            "score_diff_e5_minus_mini",
            "score_product",
            "score_mean_dual",
            "rank_pct_mean_dual",
            "rank_pct_diff_e5_minus_mini",
            "score_z_mean_dual",
            "score_z_diff_e5_minus_mini",
            "both_top_5pct",
            "both_top_10pct",
            "either_top_5pct",
        ]
    )

    features.extend(TEXT_NUMERIC_FEATURES)

    return features


def get_feature_cols() -> list[str]:
    return get_numeric_features() + CAT_FEATURES


def build_train_frame() -> pd.DataFrame:
    if TRAIN_FRAME_PATH.exists():
        print(f"Train frame cache exists: {TRAIN_FRAME_PATH}")
        return pd.read_parquet(TRAIN_FRAME_PATH)

    print("=" * 100)
    print("BUILD V6 TRAIN FRAME")
    print("=" * 100)

    pairs = pd.read_parquet(TRAIN_PAIRS_PATH)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)
    pairs["id"] = pairs["id"].astype(str)

    text = pd.read_parquet(
        TRAIN_ENRICHED_PATH,
        columns=[
            "id",
            "query",
            "title",
            "category",
            "brand",
            "gender",
            "age_group",
            "root_category",
        ],
    )
    text["id"] = text["id"].astype(str)

    df = pairs.merge(text, on="id", how="left")

    if df["query"].isna().any():
        raise ValueError("Missing train text rows")

    print("Scoring train E5...")
    df["e5_score"] = score_pairs_with_embedding(
        df,
        E5_TERM_EMB_PATH,
        E5_ITEM_EMB_PATH,
        E5_TERM_IDS_PATH,
        E5_ITEM_IDS_PATH,
        "e5_score",
    )

    print("Scoring train MiniLM...")
    df["mini_score"] = score_pairs_with_embedding(
        df,
        MINI_TERM_EMB_PATH,
        MINI_ITEM_EMB_PATH,
        MINI_TERM_IDS_PATH,
        MINI_ITEM_IDS_PATH,
        "mini_score",
    )

    print("Adding train features...")
    df = add_dual_features(df)

    df.to_parquet(TRAIN_FRAME_PATH, index=False)

    print(f"Saved train frame: {TRAIN_FRAME_PATH}")
    print("Label distribution:")
    print(df["label"].value_counts())
    print("Negative type distribution:")
    print(df["negative_type"].value_counts())

    return df


def build_test_frame() -> pd.DataFrame:
    if TEST_FRAME_PATH.exists():
        print(f"Test frame cache exists: {TEST_FRAME_PATH}")
        return pd.read_parquet(TEST_FRAME_PATH)

    print("=" * 100)
    print("BUILD V6 TEST FRAME")
    print("=" * 100)

    e5 = pd.read_parquet(E5_TEST_SCORE_PATH)
    mini = pd.read_parquet(MINI_TEST_SCORE_PATH)

    e5 = e5.rename(columns={"score": "e5_score"})
    mini = mini.rename(columns={"score": "mini_score"})

    e5["id"] = e5["id"].astype(str)
    mini["id"] = mini["id"].astype(str)

    df = e5[["id", "term_id", "item_id", "e5_score"]].merge(
        mini[["id", "mini_score"]],
        on="id",
        how="left",
    )

    terms = pd.read_csv(RAW_DIR / "terms.csv")
    items = pd.read_csv(
        RAW_DIR / "items.csv",
        usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    )

    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    df["term_id"] = df["term_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    df = df.merge(terms, on="term_id", how="left")
    df = df.merge(items, on="item_id", how="left")

    if df["query"].isna().any() or df["title"].isna().any() or df["mini_score"].isna().any():
        raise ValueError("Missing test metadata or scores")

    print("Adding test features...")
    df = add_dual_features(df)

    df.to_parquet(TEST_FRAME_PATH, index=False)

    print(f"Saved test frame: {TEST_FRAME_PATH}")

    return df


def build_realistic_eval_frame(heldout_terms: set[str]) -> pd.DataFrame:
    print("Building V6 realistic eval frame...")

    train_pairs = pd.read_parquet(TRAIN_PAIRS_PATH)
    train_pairs["term_id"] = train_pairs["term_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)
    train_pairs["id"] = train_pairs["id"].astype(str)

    positives = train_pairs[
        (train_pairs["label"] == 1)
        & (train_pairs["term_id"].isin(heldout_terms))
    ][["id", "term_id", "item_id", "label"]].copy()

    positives["source"] = "positive"

    cand = pd.read_parquet(CANDIDATE_PATH)
    cand["term_id"] = cand["term_id"].astype(str)
    cand["item_id"] = cand["item_id"].astype(str)

    cand = cand[cand["term_id"].isin(heldout_terms)].copy()
    cand = cand.sort_values(["term_id", "semantic_rank"])

    neg = (
        cand
        .groupby("term_id")
        .head(MAX_NEG_PER_TERM)
        [["term_id", "item_id"]]
        .copy()
    )

    neg["id"] = "V6REALNEG_" + neg["term_id"] + "_" + neg["item_id"]
    neg["label"] = 0
    neg["source"] = "semantic_candidate"

    eval_pairs = pd.concat(
        [
            positives[["id", "term_id", "item_id", "label", "source"]],
            neg[["id", "term_id", "item_id", "label", "source"]],
        ],
        ignore_index=True,
    )

    eval_pairs = eval_pairs.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)

    print("Scoring eval E5...")
    eval_pairs["e5_score"] = score_pairs_with_embedding(
        eval_pairs,
        E5_TERM_EMB_PATH,
        E5_ITEM_EMB_PATH,
        E5_TERM_IDS_PATH,
        E5_ITEM_IDS_PATH,
        "eval_e5",
    )

    print("Scoring eval MiniLM...")
    eval_pairs["mini_score"] = score_pairs_with_embedding(
        eval_pairs,
        MINI_TERM_EMB_PATH,
        MINI_ITEM_EMB_PATH,
        MINI_TERM_IDS_PATH,
        MINI_ITEM_IDS_PATH,
        "eval_mini",
    )

    terms = pd.read_csv(RAW_DIR / "terms.csv")
    items = pd.read_csv(
        RAW_DIR / "items.csv",
        usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    )

    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    eval_df = eval_pairs.merge(terms, on="term_id", how="left")
    eval_df = eval_df.merge(items, on="item_id", how="left")

    if eval_df["query"].isna().any() or eval_df["title"].isna().any():
        raise ValueError("Missing eval metadata")

    eval_df = add_dual_features(eval_df)

    return eval_df


def evaluate_thresholds(y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
    rows = []

    for th in CV_THRESHOLDS:
        pred = (proba >= th).astype(np.int8)
        rows.append(
            {
                "threshold": float(th),
                "macro_f1": float(f1_score(y_true, pred, average="macro")),
                "pred_pos_ratio": float(pred.mean()),
            }
        )

    return rows


def run_cv(full_train: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("RUN V6 REALISTIC CV")
    print("=" * 100)

    if CV_RAW_PATH.exists() and CV_SUMMARY_PATH.exists():
        print("CV cache exists.")
        return pd.read_csv(CV_SUMMARY_PATH)

    feature_cols = get_feature_cols()
    cat_indices = [feature_cols.index(c) for c in CAT_FEATURES]

    terms = np.array(sorted(full_train["term_id"].astype(str).unique()))
    all_rows = []

    for fold_id, seed in enumerate(CV_FOLD_SEEDS, start=1):
        print("\n" + "=" * 100)
        print(f"V6 FOLD {fold_id}, seed={seed}")
        print("=" * 100)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALID_TERM_SIZE,
            random_state=seed,
        )

        dummy_y = np.zeros(len(terms))
        train_term_idx, valid_term_idx = next(splitter.split(terms, dummy_y, groups=terms))

        train_terms = set(terms[train_term_idx])
        valid_terms = set(terms[valid_term_idx])

        train_df = full_train[full_train["term_id"].astype(str).isin(train_terms)].reset_index(drop=True)
        eval_df = build_realistic_eval_frame(valid_terms)

        print(f"Train rows: {len(train_df):,}")
        print(f"Eval rows : {len(eval_df):,}")
        print("Eval label ratio:")
        print(eval_df["label"].value_counts(normalize=True))

        y_true = eval_df["label"].astype(int).to_numpy()

        for semw in SEMANTIC_WEIGHTS:
            print("\n" + "-" * 100)
            print(f"Training V6 CV fold={fold_id}, semantic_weight={semw}")
            print("-" * 100)

            tr = train_df.copy()
            tr["cv_weight"] = 1.0
            tr.loc[tr["negative_type"] == "semantic_hard", "cv_weight"] = semw
            tr["cv_weight"] = tr["cv_weight"].astype(np.float32)

            train_pool = Pool(
                tr[feature_cols],
                label=tr["label"].astype(int),
                weight=tr["cv_weight"].astype(float),
                cat_features=cat_indices,
            )

            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="F1",
                iterations=1000,
                learning_rate=0.035,
                depth=7,
                l2_leaf_reg=7.0,
                random_seed=seed,
                auto_class_weights="Balanced",
                verbose=100,
                allow_writing_files=False,
                task_type="GPU",
                devices="0",
            )

            model.fit(train_pool)

            proba = model.predict_proba(eval_df[feature_cols])[:, 1].astype(np.float32)
            rows = evaluate_thresholds(y_true, proba)

            for r in rows:
                r.update(
                    {
                        "fold": fold_id,
                        "seed": seed,
                        "semantic_weight": semw,
                        "eval_rows": int(len(eval_df)),
                        "true_pos_ratio": float(y_true.mean()),
                    }
                )

            all_rows.extend(rows)

            best = max(rows, key=lambda x: x["macro_f1"])
            print("Best:", best)

            del model, train_pool, tr, proba
            gc.collect()

    raw = pd.DataFrame(all_rows)
    raw.to_csv(CV_RAW_PATH, index=False)

    summary = (
        raw
        .groupby(["semantic_weight", "threshold"])
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
            mean_pred_pos_ratio=("pred_pos_ratio", "mean"),
            mean_true_pos_ratio=("true_pos_ratio", "mean"),
        )
        .reset_index()
        .sort_values("mean_macro_f1", ascending=False)
    )

    summary.to_csv(CV_SUMMARY_PATH, index=False)

    print("\nV6 CV top 40:")
    print(summary.head(40).to_string(index=False))

    return summary


def threshold_tag(th: float) -> str:
    return f"{float(th):.3f}".replace(".", "p")


def make_submission(sample: pd.DataFrame, ids: pd.Series, proba: np.ndarray, threshold: float, name: str) -> dict:
    pred = (proba >= threshold).astype(np.int8)

    tmp = pd.DataFrame({"id": ids.values, "prediction": pred})
    sub = sample[["id"]].merge(tmp, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)

    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])
    assert set(sub["prediction"].unique()) <= {0, 1}

    out_path = SUBMISSIONS_DIR / f"{RUN_NAME}_{name}_threshold_{threshold_tag(threshold)}.csv"
    sub.to_csv(out_path, index=False)

    return {
        "name": name,
        "threshold": float(threshold),
        "path": str(out_path),
        "ones": int(sub["prediction"].sum()),
        "zeros": int(len(sub) - sub["prediction"].sum()),
        "pos_ratio": float(sub["prediction"].mean()),
    }


def train_full_and_submit(full_train: pd.DataFrame, test: pd.DataFrame, cv_summary: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("TRAIN V6 FULL ENSEMBLE")
    print("=" * 100)

    best = cv_summary.iloc[0]
    best_semw = float(best["semantic_weight"])
    best_threshold = float(best["threshold"])
    best_cv_pos_ratio = float(best["mean_pred_pos_ratio"])

    print(f"Best CV semw={best_semw}, threshold={best_threshold}, cv_pos_ratio={best_cv_pos_ratio}")

    feature_cols = get_feature_cols()
    cat_indices = [feature_cols.index(c) for c in CAT_FEATURES]

    tr = full_train.copy()
    tr["final_weight"] = 1.0
    tr.loc[tr["negative_type"] == "semantic_hard", "final_weight"] = best_semw
    tr["final_weight"] = tr["final_weight"].astype(np.float32)

    train_pool = Pool(
        tr[feature_cols],
        label=tr["label"].astype(int),
        weight=tr["final_weight"].astype(float),
        cat_features=cat_indices,
    )

    proba_sum = np.zeros(len(test), dtype=np.float32)
    proba_cols = {}

    for seed in FINAL_SEEDS:
        print("\n" + "=" * 100)
        print(f"Training V6 full seed={seed}, semw={best_semw}")
        print("=" * 100)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            iterations=FINAL_ITERATIONS,
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=7.0,
            random_seed=seed,
            auto_class_weights="Balanced",
            verbose=50,
            allow_writing_files=False,
            task_type="GPU",
            devices="0",
        )

        model.fit(train_pool)

        model_path = MODELS_DIR / f"{RUN_NAME}_full_seed{seed}.cbm"
        model.save_model(model_path)
        print("Saved model:", model_path)

        proba = model.predict_proba(test[feature_cols])[:, 1].astype(np.float32)
        proba_cols[f"proba_seed{seed}"] = proba
        proba_sum += proba

        del model
        gc.collect()

    proba_avg = proba_sum / float(len(FINAL_SEEDS))
    proba_cols["proba_avg"] = proba_avg

    proba_df = pd.DataFrame(
        {
            "id": test["id"].values,
            "term_id": test["term_id"].values,
            "item_id": test["item_id"].values,
            "proba_avg": proba_avg,
        }
    )

    for col, arr in proba_cols.items():
        if col != "proba_avg":
            proba_df[col] = arr

    proba_path = PROCESSED_DIR / f"{RUN_NAME}_full_test_proba.parquet"
    proba_df.to_parquet(proba_path, index=False)
    print("Saved proba:", proba_path)

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    sample["id"] = sample["id"].astype(str)

    rows = []

    for name, proba in proba_cols.items():
        for th in FINAL_THRESHOLDS:
            row = make_submission(sample, test["id"], proba, float(th), name)
            row["distance_to_cv_pos_ratio"] = abs(row["pos_ratio"] - best_cv_pos_ratio)
            rows.append(row)
            print(row)

    summary = pd.DataFrame(rows).sort_values(["distance_to_cv_pos_ratio", "name", "threshold"])
    summary.to_csv(FINAL_SUMMARY_PATH, index=False)

    print("\nV6 final summary closest to CV pos ratio:")
    print(summary.head(40).to_string(index=False))

    return summary


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("30 V6 DUAL E5 + MINILM CV THRESHOLD")
    print("=" * 100)

    required_files()

    train = build_train_frame()
    test = build_test_frame()

    print(f"Train rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    cv_summary = run_cv(train)
    final_summary = train_full_and_submit(train, test, cv_summary)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "cv_raw_path": str(CV_RAW_PATH),
        "cv_summary_path": str(CV_SUMMARY_PATH),
        "final_summary_path": str(FINAL_SUMMARY_PATH),
        "best_cv": cv_summary.head(20).to_dict(orient="records"),
        "best_final_candidates": final_summary.head(20).to_dict(orient="records"),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved report:", REPORT_PATH)

    print("\nBest V6 CV:")
    print(cv_summary.head(30).to_string(index=False))

    print("\nBest V6 final candidates:")
    print(final_summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
