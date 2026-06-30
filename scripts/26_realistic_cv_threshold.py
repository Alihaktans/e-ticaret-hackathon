from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

V4_SCRIPT = ROOT / "scripts" / "18_train_v4_weighted_semantic_ranker.py"

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "experiments"

CANDIDATE_PATH = PROCESSED_DIR / "v4_semantic_hard_negative_candidates.parquet"
TRAIN_SCORE_PATH = PROCESSED_DIR / "embedding_v4_train_scores.parquet"

RUN_NAME = "realistic_cv_threshold_v4"

FOLD_SEEDS = [4242, 2026, 777]
SEMANTIC_WEIGHTS = [0.10, 0.20, 0.35, 0.50]

THRESHOLDS = np.round(np.arange(0.90, 0.991, 0.005), 3)

VALID_TERM_SIZE = 0.20
MAX_NEG_PER_TERM = 100


def load_v4_module():
    spec = importlib.util.spec_from_file_location("v4mod", V4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load: {V4_SCRIPT}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_eval_frame(mod, heldout_terms: set[str]) -> pd.DataFrame:
    train_scores = pd.read_parquet(TRAIN_SCORE_PATH)
    train_scores["term_id"] = train_scores["term_id"].astype(str)
    train_scores["item_id"] = train_scores["item_id"].astype(str)

    positives = train_scores[
        (train_scores["label"] == 1)
        & (train_scores["term_id"].isin(heldout_terms))
    ][["id", "term_id", "item_id", "label", "embedding_score"]].copy()

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
        [["term_id", "item_id", "embedding_score"]]
        .copy()
    )

    neg["id"] = "REALNEG_" + neg["term_id"] + "_" + neg["item_id"]
    neg["label"] = 0
    neg["source"] = "semantic_candidate"

    eval_pairs = pd.concat(
        [
            positives[["id", "term_id", "item_id", "label", "embedding_score", "source"]],
            neg[["id", "term_id", "item_id", "label", "embedding_score", "source"]],
        ],
        ignore_index=True,
    )

    eval_pairs = eval_pairs.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)

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
        raise ValueError("Missing metadata in eval frame")

    eval_df = mod.add_rank_features(eval_df, score_col="embedding_score")
    eval_df = mod.add_text_features(eval_df)

    return eval_df


def evaluate_thresholds(y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
    rows = []

    for th in THRESHOLDS:
        pred = (proba >= th).astype(np.int8)

        rows.append(
            {
                "threshold": float(th),
                "macro_f1": float(f1_score(y_true, pred, average="macro")),
                "pred_pos_ratio": float(pred.mean()),
            }
        )

    return rows


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("26 REALISTIC CV THRESHOLD V4")
    print("=" * 100)

    mod = load_v4_module()

    print("Building full V4 train frame...")
    full_train = mod.build_train_frame()

    terms = np.array(sorted(full_train["term_id"].astype(str).unique()))

    feature_cols = mod.FEATURE_COLS
    cat_features = mod.CAT_FEATURES
    cat_indices = [feature_cols.index(c) for c in cat_features]

    all_rows = []

    for fold_id, seed in enumerate(FOLD_SEEDS, start=1):
        print("\n" + "=" * 100)
        print(f"FOLD {fold_id} seed={seed}")
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
        eval_df = build_eval_frame(mod, valid_terms)

        print(f"Train rows: {len(train_df):,}")
        print(f"Eval rows : {len(eval_df):,}")
        print("Eval label ratio:")
        print(eval_df["label"].value_counts(normalize=True))

        y_true = eval_df["label"].astype(int).to_numpy()

        for semw in SEMANTIC_WEIGHTS:
            print("\n" + "-" * 100)
            print(f"Training fold={fold_id}, semantic_weight={semw}")
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
                iterations=900,
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

    df = pd.DataFrame(all_rows)

    raw_path = REPORT_DIR / f"{RUN_NAME}_raw.csv"
    df.to_csv(raw_path, index=False)

    summary = (
        df
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

    summary_path = REPORT_DIR / f"{RUN_NAME}_summary.csv"
    summary.to_csv(summary_path, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "fold_seeds": FOLD_SEEDS,
        "semantic_weights": SEMANTIC_WEIGHTS,
        "thresholds": [float(x) for x in THRESHOLDS],
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "top_30": summary.head(30).to_dict(orient="records"),
    }

    report_path = REPORT_DIR / f"{RUN_NAME}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved:", raw_path)
    print("Saved:", summary_path)
    print("Saved:", report_path)

    print("\nTop 40 summary:")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
