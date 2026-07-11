"""V224 query-grouped CatBoost ranker with teacher-regularized decoding.

Uses V219 cached lexical features. Training groups never cross folds. Test output
preserves V104's positive count separately for every term_id; only within-query
ordering may change.

Run from repository root:
    python scripts/224_train_query_grouped_ranker.py train
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

ROOT = Path(".")
PROC = ROOT / "data/processed"
RAW = ROOT / "data/raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports/experiments"
OUT = ROOT / "submissions/final_candidates_v224"

TRAIN = PROC / "v219_train_features.parquet"
TEST = PROC / "v219_test_features.parquet"
TEACHER = ROOT / "submissions/reference/FINAL_CANDIDATE_v104_full_balanced.csv"
TEST_SCORE = PROC / "v224_ranker_test_scores.parquet"

SEED = 20260711
ITERATIONS = int(os.environ.get("V224_ITERATIONS", "900"))
N_FOLDS = int(os.environ.get("V224_N_FOLDS", "3"))
TEST_CHUNK = int(os.environ.get("V224_TEST_CHUNK", "300000"))
TEACHER_BONUSES = [0.80, 0.65, 0.50]

IDENTITY = {"id", "term_id", "item_id", "label", "fold", "negative_type", "sample_weight"}


def require_inputs() -> None:
    missing = [str(p) for p in [TRAIN, TEST, TEACHER, RAW / "submission_pairs.csv"] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing V224 inputs:\n- " + "\n- ".join(missing))


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c not in IDENTITY]
    if not cols:
        raise ValueError("No feature columns found")
    return cols


def make_model(iterations: int) -> CatBoostRanker:
    params = dict(
        loss_function="YetiRankPairwise",
        eval_metric="NDCG:top=10",
        iterations=iterations,
        learning_rate=0.045,
        depth=7,
        l2_leaf_reg=10.0,
        random_seed=SEED,
        verbose=100,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",
    )
    return CatBoostRanker(**params)


def sorted_pool(frame: pd.DataFrame, features: list[str], with_label: bool = True) -> tuple[Pool, pd.DataFrame]:
    frame = frame.sort_values(["term_id", "id"], kind="stable").reset_index(drop=True)
    kwargs = dict(data=frame[features], group_id=frame["term_id"].astype(str))
    if with_label:
        kwargs["label"] = frame["label"].astype(np.float32)
        # Pairwise ranking losses expect group-consistent weights. V219 contains
        # per-object PU weights, so passing them here would be invalid/misleading.
    return Pool(**kwargs), frame


def fit_with_fallback(train_pool: Pool, valid_pool: Pool, iterations: int) -> CatBoostRanker:
    model = make_model(iterations)
    try:
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=100)
    except Exception as exc:
        print("GPU fallback:", repr(exc), flush=True)
        params = model.get_params()
        params["task_type"] = "CPU"
        params.pop("devices", None)
        params["thread_count"] = max(1, (os.cpu_count() or 4) - 2)
        model = CatBoostRanker(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=100)
    return model


def predict_chunks(model: CatBoostRanker, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    out = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), TEST_CHUNK):
        end = min(start + TEST_CHUNK, len(frame))
        out[start:end] = model.predict(frame.iloc[start:end][features]).astype(np.float32)
        print("predict", end, "/", len(frame), flush=True)
    return out


def within_group_percentile(frame: pd.DataFrame, score: str) -> np.ndarray:
    return frame.groupby("term_id", sort=False)[score].rank(method="average", pct=True).to_numpy(np.float32)


def decode_candidates(test: pd.DataFrame) -> list[dict]:
    teacher = pd.read_csv(TEACHER, usecols=["id", "prediction"], dtype={"id": str, "prediction": np.int8})
    if not teacher["id"].equals(test["id"].astype(str).reset_index(drop=True)):
        raise ValueError("Teacher/test ID order mismatch")
    work = test[["id", "term_id", "ranker_score"]].copy()
    work["teacher"] = teacher["prediction"].to_numpy(np.int8)
    work["rank_pct"] = within_group_percentile(work, "ranker_score")
    target_counts = work.groupby("term_id", sort=False)["teacher"].sum().astype(int)
    summaries = []
    OUT.mkdir(parents=True, exist_ok=True)

    for bonus in TEACHER_BONUSES:
        work["decode_score"] = work["rank_pct"] + np.float32(bonus) * work["teacher"]
        # Stable ranking and group-wise top-k with exactly the teacher's k.
        order = work.sort_values(["term_id", "decode_score", "id"], ascending=[True, False, True], kind="stable")
        order["position"] = order.groupby("term_id", sort=False).cumcount()
        order["k"] = order["term_id"].map(target_counts)
        order["prediction"] = (order["position"] < order["k"]).astype(np.int8)
        pred = order.set_index("id")["prediction"].reindex(work["id"]).to_numpy(np.int8)
        changed = pred != teacher["prediction"].to_numpy(np.int8)
        path = OUT / f"FINAL_CANDIDATE_v224_ranker_teacher_bonus_{str(bonus).replace('.', 'p')}.csv"
        pd.DataFrame({"id": work["id"], "prediction": pred}).to_csv(path, index=False)
        summaries.append({
            "teacher_bonus": bonus,
            "changes_vs_v104": int(changed.sum()),
            "positive_ratio": float(pred.mean()),
            "path": str(path),
        })
    return summaries


def train() -> None:
    require_inputs()
    tr = pd.read_parquet(TRAIN)
    te = pd.read_parquet(TEST)
    tr["term_id"] = tr["term_id"].astype(str)
    te["term_id"] = te["term_id"].astype(str)
    features = feature_columns(tr)
    missing = [c for c in features if c not in te]
    if missing:
        raise ValueError(f"Test cache missing features: {missing}")

    test_sum = np.zeros(len(te), dtype=np.float64)
    best_iterations = []
    MODELS.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        fit = tr[tr["fold"].ne(fold)].copy()
        valid = tr[tr["fold"].eq(fold)].copy()
        fit_pool, fit_sorted = sorted_pool(fit, features)
        valid_pool, valid_sorted = sorted_pool(valid, features)
        model = fit_with_fallback(fit_pool, valid_pool, ITERATIONS)
        best = int(model.get_best_iteration())
        if best < 0:
            best = ITERATIONS - 1
        best_iterations.append(best + 1)
        test_sum += predict_chunks(model, te, features) / N_FOLDS
        model.save_model(str(MODELS / f"v224_query_ranker_fold{fold}.cbm"))
        print({"fold": fold, "train": len(fit_sorted), "valid": len(valid_sorted), "best": best + 1}, flush=True)

    # Fold models are used for cross-fitted test averaging; synthetic validation
    # ranking is not reported as a competition metric.
    pd.DataFrame({"id": te["id"].astype(str), "term_id": te["term_id"], "ranker_score": test_sum.astype(np.float32)}).to_parquet(TEST_SCORE, index=False)
    scored_test = te[["id", "term_id"]].copy()
    scored_test["ranker_score"] = test_sum.astype(np.float32)
    summaries = decode_candidates(scored_test.reset_index(drop=True))
    report = {
        "run": "v224_query_grouped_ranker",
        "loss": "YetiRankPairwise",
        "features": features,
        "folds": N_FOLDS,
        "best_iterations": best_iterations,
        "train_rows": len(tr),
        "test_rows": len(te),
        "decoding": "per-term positive count preserved; within-query ranker + teacher bonus",
        "candidates": summaries,
        "warning": "Training negatives are synthetic; offline ranking metrics are not ground-truth leaderboard metrics.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "v224_query_grouped_ranker.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train"])
    args = parser.parse_args()
    if args.stage == "train":
        train()


if __name__ == "__main__":
    main()
