from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score

from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR, EXPERIMENT_REPORTS_DIR
from trendyol.text_preprocess import normalize_text, extract_root_category


RUN_NAME = "embedding_ranker_v3_catboost_text_no_brand"

RANDOM_STATE = 42
VALID_SIZE = 0.20

TRAIN_ENRICHED_PATH = PROCESSED_DIR / "train_enriched_v1.parquet"
TRAIN_SCORE_PATH = PROCESSED_DIR / "embedding_v1_train_scores.parquet"
TEST_SCORE_PATH = PROCESSED_DIR / "embedding_v1_minilm_topratio_008_scores.parquet"

TOP_RATIOS = [0.08, 0.10, 0.12, 0.14]

NUMERIC_FEATURES = [
    "embedding_score",
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
]

CAT_FEATURES = [
    "root_category",
    "gender",
    "age_group",
]

FEATURE_COLS = NUMERIC_FEATURES + CAT_FEATURES


def token_set(text: object) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    return {t for t in text.split() if len(t) >= 2}


def add_rank_features(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df = df.copy()

    if score_col != "embedding_score":
        df = df.rename(columns={score_col: "embedding_score"})

    g = df.groupby("term_id")["embedding_score"]

    df["candidate_count"] = g.transform("size").astype(np.float32)
    df["score_mean"] = g.transform("mean").astype(np.float32)
    df["score_std"] = g.transform("std").fillna(0).astype(np.float32)
    df["score_max"] = g.transform("max").astype(np.float32)
    df["score_min"] = g.transform("min").astype(np.float32)
    df["score_range"] = (df["score_max"] - df["score_min"]).astype(np.float32)

    df["score_z"] = (
        (df["embedding_score"] - df["score_mean"]) / (df["score_std"] + 1e-6)
    ).astype(np.float32)

    df["rank_desc"] = g.rank(method="first", ascending=False).astype(np.float32)

    df["rank_pct"] = (
        (df["rank_desc"] - 1.0) / (df["candidate_count"] - 1.0).clip(lower=1.0)
    ).astype(np.float32)

    df["gap_to_top"] = (df["score_max"] - df["embedding_score"]).astype(np.float32)

    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # V3 no-brand modelinde brand feature olarak kullanılmıyor.
    # Ama fonksiyon brand kolonunu bekleyebiliyor; yoksa boş kolon ekliyoruz.
    for required_col in ["query", "title", "category", "brand", "gender", "age_group"]:
        if required_col not in df.columns:
            df[required_col] = ""

    for col in ["query", "title", "category", "brand", "gender", "age_group"]:
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

    for col in NUMERIC_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").replace("", "unknown").astype(str)

    return df


def build_train_frame() -> pd.DataFrame:
    print("Reading train enriched...")
    train_text = pd.read_parquet(
        TRAIN_ENRICHED_PATH,
        columns=[
            "id",
            "query",
            "title",
            "category",
                    "gender",
            "age_group",
            "root_category",
        ],
    )

    print("Reading train scores...")
    train_score = pd.read_parquet(TRAIN_SCORE_PATH)

    train = train_score.merge(train_text, on="id", how="left")

    missing = int(train["query"].isna().sum())
    if missing:
        raise ValueError(f"Missing train text rows: {missing}")

    print("Adding train rank features...")
    train = add_rank_features(train, score_col="embedding_score")

    print("Adding train text features...")
    train = add_text_features(train)

    return train


def build_test_frame() -> pd.DataFrame:
    print("Reading test scores...")
    test = pd.read_parquet(TEST_SCORE_PATH)

    print("Reading metadata...")
    terms = pd.read_csv(RAW_DIR / "terms.csv")
    items = pd.read_csv(
        RAW_DIR / "items.csv",
        usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    )

    items["root_category"] = items["category"].map(extract_root_category)

    test = test.merge(terms, on="term_id", how="left")
    test = test.merge(items, on="item_id", how="left")

    missing_q = int(test["query"].isna().sum())
    missing_t = int(test["title"].isna().sum())

    if missing_q or missing_t:
        raise ValueError(f"Missing test metadata: query={missing_q}, title={missing_t}")

    print("Adding test rank features...")
    test = add_rank_features(test, score_col="score")

    print("Adding test text features...")
    test = add_text_features(test)

    return test


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict:
    best = {"threshold": 0.5, "macro_f1": -1.0}

    for th in np.linspace(0.01, 0.99, 99):
        pred = (proba >= th).astype(int)
        score = f1_score(y_true, pred, average="macro")

        if score > best["macro_f1"]:
            best = {"threshold": float(th), "macro_f1": float(score)}

    return best


def make_topratio_submission(test_df: pd.DataFrame, ratio: float) -> Path:
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

    pred = np.zeros(len(test_df), dtype=np.int8)
    proba_series = test_df["final_score"]

    for _, idx in tqdm(test_df.groupby("term_id").groups.items(), desc=f"topratio {ratio}"):
        n = len(idx)

        k = round(n * ratio)
        k = max(1, k)
        k = min(20, k)
        k = min(k, n)

        top_idx = proba_series.loc[idx].nlargest(k).index
        pred[top_idx] = 1

    tmp = test_df[["id"]].copy()
    tmp["prediction"] = pred

    sub = sample[["id"]].merge(tmp, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)

    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])
    assert set(sub["prediction"].unique()) <= {0, 1}

    tag = f"{int(round(ratio * 100)):03d}"
    out_path = SUBMISSIONS_DIR / f"{RUN_NAME}_blend_topratio_{tag}.csv"
    sub.to_csv(out_path, index=False)

    print(f"\nSaved submission: {out_path}")
    print(sub["prediction"].value_counts())
    print(sub["prediction"].value_counts(normalize=True))

    return out_path


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("13 TRAIN EMBEDDING RANKER V3 NO BRAND")
    print("=" * 100)

    train = build_train_frame()
    test = build_test_frame()

    print(f"\nTrain rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, valid_idx = next(
        splitter.split(train, train["label"], groups=train["term_id"])
    )

    tr = train.iloc[train_idx].reset_index(drop=True)
    va = train.iloc[valid_idx].reset_index(drop=True)

    print(f"\nTrain split rows: {len(tr):,}")
    print(f"Valid split rows: {len(va):,}")

    print("\nTrain label ratio:")
    print(tr["label"].value_counts(normalize=True))
    print("\nValid label ratio:")
    print(va["label"].value_counts(normalize=True))

    cat_indices = [FEATURE_COLS.index(c) for c in CAT_FEATURES]

    train_pool = Pool(
        tr[FEATURE_COLS],
        label=tr["label"].astype(int),
        cat_features=cat_indices,
    )

    valid_pool = Pool(
        va[FEATURE_COLS],
        label=va["label"].astype(int),
        cat_features=cat_indices,
    )

    params = {
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "iterations": 1500,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "random_seed": RANDOM_STATE,
        "auto_class_weights": "Balanced",
        "verbose": 100,
        "allow_writing_files": False,
    }

    print("\nTraining CatBoost V2...")

    try:
        model = CatBoostClassifier(**params, task_type="GPU", devices="0")
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        task_type = "GPU"
    except Exception as exc:
        print("\nGPU CatBoost failed, falling back to CPU.")
        print(f"Reason: {exc}")
        model = CatBoostClassifier(**params, task_type="CPU", thread_count=-1)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        task_type = "CPU"

    valid_proba = model.predict_proba(va[FEATURE_COLS])[:, 1]
    best = find_best_threshold(va["label"].astype(int).to_numpy(), valid_proba)

    valid_pred = (valid_proba >= best["threshold"]).astype(int)
    valid_macro_f1 = f1_score(va["label"].astype(int), valid_pred, average="macro")
    report_text = classification_report(va["label"].astype(int), valid_pred, digits=5)

    print("\nValidation report:")
    print(report_text)
    print(f"Best threshold: {best['threshold']:.5f}")
    print(f"Valid Macro-F1: {valid_macro_f1:.5f}")
    print(f"Task type     : {task_type}")

    print("\nPredicting test probabilities...")
    test["proba"] = model.predict_proba(test[FEATURE_COLS])[:, 1].astype(np.float32)

    # Saf model proba synthetic negatiflere fazla uyarsa diye cosine ile blend yapıyoruz.
    # 0.70 embedding rank güveni + 0.30 model güveni.
    test["embedding_rank_score"] = 1.0 - test["rank_pct"]
    test["final_score"] = (
        0.80 * test["embedding_rank_score"].astype(np.float32)
        + 0.20 * test["proba"].astype(np.float32)
    ).astype(np.float32)

    model_path = MODELS_DIR / f"{RUN_NAME}.cbm"
    model.save_model(model_path)

    meta_path = MODELS_DIR / f"{RUN_NAME}_meta.joblib"
    joblib.dump(
        {
            "feature_cols": FEATURE_COLS,
            "numeric_features": NUMERIC_FEATURES,
            "cat_features": CAT_FEATURES,
            "cat_indices": cat_indices,
            "best_threshold": best["threshold"],
            "run_name": RUN_NAME,
            "task_type": task_type,
            "blend": "final_score = 0.80 * embedding_rank_score + 0.20 * proba",
        },
        meta_path,
    )

    proba_path = PROCESSED_DIR / f"{RUN_NAME}_test_proba.parquet"
    save_cols = ["id", "term_id", "item_id", "embedding_score", "proba", "embedding_rank_score", "final_score"]
    test[save_cols].to_parquet(proba_path, index=False)

    print(f"Saved test proba: {proba_path}")

    outputs = []
    for ratio in TOP_RATIOS:
        outputs.append(str(make_topratio_submission(test, ratio)))

    experiment_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "model": "CatBoostClassifier on MiniLM + rank + text/category features",
        "task_type": task_type,
        "feature_cols": FEATURE_COLS,
        "valid_macro_f1": float(valid_macro_f1),
        "best_threshold": float(best["threshold"]),
        "classification_report": report_text,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "test_proba_path": str(proba_path),
        "submission_outputs": outputs,
        "warning": "Validation uses synthetic negatives. Use Kaggle score as final source of truth.",
    }

    report_path = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_report.json"
    report_path.write_text(json.dumps(experiment_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved model : {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
