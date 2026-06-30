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


RUN_NAME = "embedding_ranker_v1_catboost"
RANDOM_STATE = 42
VALID_SIZE = 0.20

TRAIN_SCORE_PATH = PROCESSED_DIR / "embedding_v1_train_scores.parquet"
TEST_SCORE_PATH = PROCESSED_DIR / "embedding_v1_minilm_topratio_008_scores.parquet"

FEATURE_COLS = [
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
]

TOP_RATIOS = [0.05, 0.08, 0.10, 0.12, 0.16]


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

    df["rank_desc"] = (
        g.rank(method="first", ascending=False)
        .astype(np.float32)
    )

    df["rank_pct"] = (
        (df["rank_desc"] - 1.0) / (df["candidate_count"] - 1.0).clip(lower=1.0)
    ).astype(np.float32)

    df["gap_to_top"] = (
        df["score_max"] - df["embedding_score"]
    ).astype(np.float32)

    for col in FEATURE_COLS:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)

    return df


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict:
    best = {
        "threshold": 0.5,
        "macro_f1": -1.0,
    }

    for th in np.linspace(0.01, 0.99, 99):
        pred = (proba >= th).astype(int)
        score = f1_score(y_true, pred, average="macro")

        if score > best["macro_f1"]:
            best = {
                "threshold": float(th),
                "macro_f1": float(score),
            }

    return best


def make_topratio_submission(test_df: pd.DataFrame, ratio: float) -> Path:
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

    pred = np.zeros(len(test_df), dtype=np.int8)
    proba_series = test_df["proba"]

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
    out_path = SUBMISSIONS_DIR / f"{RUN_NAME}_topratio_{tag}.csv"
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
    print("05 TRAIN EMBEDDING RANKER")
    print("=" * 100)

    if not TRAIN_SCORE_PATH.exists():
        raise FileNotFoundError(f"Missing train scores: {TRAIN_SCORE_PATH}")

    if not TEST_SCORE_PATH.exists():
        raise FileNotFoundError(f"Missing test scores: {TEST_SCORE_PATH}")

    print("Reading train scores...")
    train = pd.read_parquet(TRAIN_SCORE_PATH)

    print("Reading test scores...")
    test = pd.read_parquet(TEST_SCORE_PATH)

    print(f"Train rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    print("\nBuilding train rank features...")
    train = add_rank_features(train, score_col="embedding_score")

    print("Building test rank features...")
    test = add_rank_features(test, score_col="score")

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

    train_pool = Pool(
        tr[FEATURE_COLS],
        label=tr["label"].astype(int),
    )

    valid_pool = Pool(
        va[FEATURE_COLS],
        label=va["label"].astype(int),
    )

    params = {
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "iterations": 1200,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_STATE,
        "auto_class_weights": "Balanced",
        "verbose": 100,
        "allow_writing_files": False,
    }

    print("\nTraining CatBoost...")

    try:
        model = CatBoostClassifier(
            **params,
            task_type="GPU",
            devices="0",
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        task_type = "GPU"
    except Exception as exc:
        print("\nGPU CatBoost failed, falling back to CPU.")
        print(f"Reason: {exc}")

        model = CatBoostClassifier(
            **params,
            task_type="CPU",
            thread_count=-1,
        )
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

    model_path = MODELS_DIR / f"{RUN_NAME}.cbm"
    model.save_model(model_path)

    joblib_path = MODELS_DIR / f"{RUN_NAME}_meta.joblib"
    joblib.dump(
        {
            "feature_cols": FEATURE_COLS,
            "best_threshold": best["threshold"],
            "run_name": RUN_NAME,
            "task_type": task_type,
        },
        joblib_path,
    )

    print("\nPredicting test probabilities...")
    test["proba"] = model.predict_proba(test[FEATURE_COLS])[:, 1].astype(np.float32)

    proba_path = PROCESSED_DIR / f"{RUN_NAME}_test_proba.parquet"

    # FEATURE_COLS içinde embedding_score zaten var.
    # Duplicate column hatasını önlemek için kolonları unique hale getiriyoruz.
    save_cols = ["id", "term_id", "item_id", "proba"] + FEATURE_COLS
    save_cols = list(dict.fromkeys(save_cols))

    test[save_cols].to_parquet(
        proba_path,
        index=False,
    )

    print(f"Saved test proba: {proba_path}")

    outputs = []
    for ratio in TOP_RATIOS:
        outputs.append(str(make_topratio_submission(test, ratio)))

    experiment_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "model": "CatBoostClassifier on MiniLM cosine rank features",
        "task_type": task_type,
        "feature_cols": FEATURE_COLS,
        "valid_macro_f1": float(valid_macro_f1),
        "best_threshold": float(best["threshold"]),
        "classification_report": report_text,
        "model_path": str(model_path),
        "meta_path": str(joblib_path),
        "test_proba_path": str(proba_path),
        "submission_outputs": outputs,
        "warning": (
            "Validation uses synthetic negatives. It is useful for relative comparison, "
            "not a direct Kaggle score estimate."
        ),
    }

    report_path = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_report.json"
    report_path.write_text(json.dumps(experiment_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved model : {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

