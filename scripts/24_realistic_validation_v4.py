from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report

from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

SOURCE_SCRIPT = ROOT / "scripts" / "18_train_v4_weighted_semantic_ranker.py"

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "experiments"

CANDIDATE_PATH = PROCESSED_DIR / "v4_semantic_hard_negative_candidates.parquet"
TRAIN_SCORE_PATH = PROCESSED_DIR / "embedding_v4_train_scores.parquet"

RANDOM_STATE = 4242
VALID_TERM_SIZE = 0.20

SEMANTIC_WEIGHTS = [0.10, 0.20, 0.35, 0.50]
BLENDS = [0.90, 0.85, 0.80, 0.75, 0.70]
TOP_RATIOS = [0.08, 0.10, 0.12, 0.14, 0.16]

MAX_NEG_PER_TERM = 100


def load_v4_module():
    spec = importlib.util.spec_from_file_location("v4mod", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {SOURCE_SCRIPT}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_topratio_pred(df: pd.DataFrame, score_col: str, ratio: float) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    score_series = df[score_col]

    for _, idx in df.groupby("term_id").groups.items():
        n = len(idx)
        k = round(n * ratio)
        k = max(1, k)
        k = min(20, k)
        k = min(k, n)

        top_idx = score_series.loc[idx].nlargest(k).index
        pred[top_idx] = 1

    return pred


def evaluate_topratio(df: pd.DataFrame, score_col: str, ratio: float) -> dict:
    pred = make_topratio_pred(df, score_col, ratio)
    y = df["label"].astype(int).to_numpy()

    return {
        "score_col": score_col,
        "top_ratio": ratio,
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "pred_pos_ratio": float(pred.mean()),
        "true_pos_ratio": float(y.mean()),
    }


def build_realistic_eval_frame(mod, heldout_terms: set[str]) -> pd.DataFrame:
    print("Building realistic validation frame...")

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

    print("Eval rows:", len(eval_pairs))
    print("Eval label counts:")
    print(eval_pairs["label"].value_counts())
    print("Eval source counts:")
    print(eval_pairs["source"].value_counts())

    terms = pd.read_csv(RAW_DIR / "terms.csv")
    items = pd.read_csv(
        RAW_DIR / "items.csv",
        usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    )

    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    eval_df = eval_pairs.merge(terms, on="term_id", how="left")
    eval_df = eval_df.merge(items, on="item_id", how="left")

    missing_q = int(eval_df["query"].isna().sum())
    missing_t = int(eval_df["title"].isna().sum())

    if missing_q or missing_t:
        raise ValueError(f"Missing metadata: query={missing_q}, title={missing_t}")

    eval_df = mod.add_rank_features(eval_df, score_col="embedding_score")
    eval_df = mod.add_text_features(eval_df)

    eval_df["embedding_rank_score"] = 1.0 - eval_df["rank_pct"].astype(np.float32)

    return eval_df


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("24 REALISTIC VALIDATION V4")
    print("=" * 100)

    mod = load_v4_module()

    print("Building full V4 train frame...")
    full_train = mod.build_train_frame()

    terms = np.array(sorted(full_train["term_id"].astype(str).unique()))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALID_TERM_SIZE,
        random_state=RANDOM_STATE,
    )

    dummy_y = np.zeros(len(terms))
    train_term_idx, valid_term_idx = next(splitter.split(terms, dummy_y, groups=terms))

    train_terms = set(terms[train_term_idx])
    valid_terms = set(terms[valid_term_idx])

    train_df = full_train[full_train["term_id"].astype(str).isin(train_terms)].reset_index(drop=True)
    eval_df = build_realistic_eval_frame(mod, valid_terms)

    print("\nTrain rows:", len(train_df))
    print("Eval rows :", len(eval_df))
    print("Train terms:", len(train_terms))
    print("Eval terms :", len(valid_terms))

    feature_cols = mod.FEATURE_COLS
    cat_features = mod.CAT_FEATURES
    cat_indices = [feature_cols.index(c) for c in cat_features]

    results = []

    print("\nEmbedding baseline realistic validation:")
    for ratio in TOP_RATIOS:
        row = evaluate_topratio(eval_df, "embedding_rank_score", ratio)
        row.update({"model": "embedding_rank_baseline", "semantic_weight": None, "blend": None})
        results.append(row)
        print(row)

    for semantic_weight in SEMANTIC_WEIGHTS:
        print("\n" + "=" * 100)
        print(f"Training realistic model semantic_weight={semantic_weight}")
        print("=" * 100)

        tr = train_df.copy()
        tr["realistic_weight"] = 1.0
        tr.loc[tr["negative_type"] == "semantic_hard", "realistic_weight"] = semantic_weight
        tr["realistic_weight"] = tr["realistic_weight"].astype(np.float32)

        train_pool = Pool(
            tr[feature_cols],
            label=tr["label"].astype(int),
            weight=tr["realistic_weight"].astype(float),
            cat_features=cat_indices,
        )

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            iterations=900,
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=7.0,
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced",
            verbose=100,
            allow_writing_files=False,
            task_type="GPU",
            devices="0",
        )

        model.fit(train_pool)

        eval_df[f"proba_semw_{semantic_weight}"] = model.predict_proba(eval_df[feature_cols])[:, 1].astype(np.float32)

        # threshold report sadece bilgi için
        y = eval_df["label"].astype(int).to_numpy()
        prob = eval_df[f"proba_semw_{semantic_weight}"].to_numpy()
        best_th = 0.5
        best_f1 = -1

        for th in np.linspace(0.01, 0.99, 99):
            pred = (prob >= th).astype(int)
            f1 = f1_score(y, pred, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        print(f"Best raw threshold F1: {best_f1:.5f} at th={best_th:.2f}")
        print(classification_report(y, (prob >= best_th).astype(int), digits=5))

        for blend in BLENDS:
            score_col = f"final_semw_{semantic_weight}_blend_{blend}"
            eval_df[score_col] = (
                blend * eval_df["embedding_rank_score"].astype(np.float32)
                + (1.0 - blend) * eval_df[f"proba_semw_{semantic_weight}"].astype(np.float32)
            ).astype(np.float32)

            for ratio in TOP_RATIOS:
                row = evaluate_topratio(eval_df, score_col, ratio)
                row.update(
                    {
                        "model": "v4_realistic",
                        "semantic_weight": semantic_weight,
                        "blend": blend,
                        "raw_threshold_macro_f1": float(best_f1),
                        "raw_best_threshold": float(best_th),
                    }
                )
                results.append(row)

                print(row)

    result_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)

    out_path = REPORT_DIR / "realistic_validation_v4_results.csv"
    result_df.to_csv(out_path, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "valid_term_size": VALID_TERM_SIZE,
        "max_neg_per_term": MAX_NEG_PER_TERM,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_terms": int(len(train_terms)),
        "eval_terms": int(len(valid_terms)),
        "out_path": str(out_path),
        "top_results": result_df.head(30).to_dict(orient="records"),
    }

    report_path = REPORT_DIR / "realistic_validation_v4_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved:", out_path)
    print("Saved:", report_path)
    print("\nTop 40:")
    print(result_df.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
