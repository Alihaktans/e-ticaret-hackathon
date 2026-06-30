from __future__ import annotations

import sys
import gc
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier, Pool
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from trendyol.config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR, EXPERIMENT_REPORTS_DIR
from trendyol.text_preprocess import normalize_text


SOURCE_SCRIPT = ROOT / "scripts" / "18_train_v4_weighted_semantic_ranker.py"

RUN_NAME = "v5_e5base"

MODEL_NAME = "intfloat/multilingual-e5-base"

TRAIN_PAIRS_PATH = PROCESSED_DIR / "train_pairs_v4_semantic_hard.parquet"
CANDIDATE_PATH = PROCESSED_DIR / "v4_semantic_hard_negative_candidates.parquet"

TERM_EMB_PATH = PROCESSED_DIR / f"{RUN_NAME}_all_terms_emb.npy"
ITEM_EMB_PATH = PROCESSED_DIR / f"{RUN_NAME}_all_items_emb.npy"
TERM_IDS_PATH = PROCESSED_DIR / f"{RUN_NAME}_all_term_ids.parquet"
ITEM_IDS_PATH = PROCESSED_DIR / f"{RUN_NAME}_all_item_ids.parquet"

TRAIN_SCORE_PATH = PROCESSED_DIR / f"embedding_{RUN_NAME}_train_scores.parquet"
TEST_SCORE_PATH = PROCESSED_DIR / f"embedding_{RUN_NAME}_test_scores.parquet"

CV_RAW_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_realistic_cv_raw.csv"
CV_SUMMARY_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_realistic_cv_summary.csv"

FINAL_PROBA_PATH = PROCESSED_DIR / f"{RUN_NAME}_full_test_proba.parquet"
FINAL_SUMMARY_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_full_submission_summary.csv"
REPORT_PATH = EXPERIMENT_REPORTS_DIR / f"{RUN_NAME}_report.json"

ENCODE_BATCH_SIZE = 256
ENCODE_CHUNK_SIZE = 25_000
SCORE_CHUNK_SIZE = 100_000

CV_FOLD_SEEDS = [4242, 2026, 777]
SEMANTIC_WEIGHTS = [0.20, 0.35, 0.50]
THRESHOLDS = np.round(np.arange(0.90, 0.991, 0.005), 3)

VALID_TERM_SIZE = 0.20
MAX_NEG_PER_TERM = 100

FINAL_SEEDS = [42, 2026, 777]
FINAL_ITERATIONS = 300


def load_v4_module():
    spec = importlib.util.spec_from_file_location("v4mod", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {SOURCE_SCRIPT}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_device() -> str:
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.get_device_name(0))
        return "cuda"
    print("CUDA not available, using CPU. This will be slow.")
    return "cpu"


def build_term_texts() -> pd.DataFrame:
    print("Reading terms...")
    terms = pl.read_csv(RAW_DIR / "terms.csv").to_pandas()
    terms["term_id"] = terms["term_id"].astype(str)
    terms["query"] = terms["query"].fillna("").map(normalize_text)
    terms["e5_text"] = "query: " + terms["query"]
    return terms[["term_id", "e5_text"]]


def build_item_texts() -> pd.DataFrame:
    print("Reading items...")
    items = pl.read_csv(RAW_DIR / "items.csv").to_pandas()
    items["item_id"] = items["item_id"].astype(str)

    for col in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if col not in items.columns:
            items[col] = ""
        items[col] = items[col].fillna("").map(normalize_text)

    # E5 passage prefix önemli.
    items["item_text"] = (
        items["title"] + " " +
        items["category"] + " " +
        items["brand"] + " " +
        items["gender"] + " " +
        items["age_group"] + " " +
        items["attributes"]
    ).str.strip()

    items["e5_text"] = "passage: " + items["item_text"]

    return items[["item_id", "e5_text"]]


def encode_to_memmap(
    model: SentenceTransformer,
    texts: list[str],
    out_path: Path,
    batch_size: int,
    chunk_size: int,
) -> None:
    if out_path.exists():
        print(f"Embedding cache exists, skipping: {out_path}")
        return

    if not texts:
        raise ValueError("No texts to encode.")

    print(f"Encoding {len(texts):,} texts -> {out_path}")

    probe = model.encode(
        [texts[0]],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    dim = int(probe.shape[1])
    mmap = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(texts), dim),
    )

    for start in tqdm(range(0, len(texts), chunk_size), desc=f"encode {out_path.name}"):
        end = min(start + chunk_size, len(texts))
        emb = model.encode(
            texts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        mmap[start:end] = emb.astype(np.float16)

        del emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del mmap
    gc.collect()

    print(f"Saved embeddings: {out_path}")


def build_embeddings() -> None:
    print("=" * 100)
    print("BUILD V5 E5 EMBEDDINGS")
    print("=" * 100)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    need_terms = not TERM_EMB_PATH.exists() or not TERM_IDS_PATH.exists()
    need_items = not ITEM_EMB_PATH.exists() or not ITEM_IDS_PATH.exists()

    if not need_terms and not need_items:
        print("All E5 embedding caches already exist.")
        return

    device = get_device()
    model = SentenceTransformer(MODEL_NAME, device=device)

    if need_terms:
        terms = build_term_texts()
        terms[["term_id"]].to_parquet(TERM_IDS_PATH, index=False)
        encode_to_memmap(
            model=model,
            texts=terms["e5_text"].tolist(),
            out_path=TERM_EMB_PATH,
            batch_size=ENCODE_BATCH_SIZE,
            chunk_size=ENCODE_CHUNK_SIZE,
        )
        del terms
        gc.collect()

    if need_items:
        items = build_item_texts()
        items[["item_id"]].to_parquet(ITEM_IDS_PATH, index=False)
        encode_to_memmap(
            model=model,
            texts=items["e5_text"].tolist(),
            out_path=ITEM_EMB_PATH,
            batch_size=ENCODE_BATCH_SIZE,
            chunk_size=ENCODE_CHUNK_SIZE,
        )
        del items
        gc.collect()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_embedding_ids():
    term_ids = pd.read_parquet(TERM_IDS_PATH)
    item_ids = pd.read_parquet(ITEM_IDS_PATH)

    term_ids["term_id"] = term_ids["term_id"].astype(str)
    item_ids["item_id"] = item_ids["item_id"].astype(str)

    term_to_idx = dict(zip(term_ids["term_id"], range(len(term_ids))))
    item_to_idx = dict(zip(item_ids["item_id"], range(len(item_ids))))

    return term_to_idx, item_to_idx


def score_pairs_df(
    pairs: pd.DataFrame,
    score_col: str,
    keep_cols: list[str],
) -> pd.DataFrame:
    term_to_idx, item_to_idx = load_embedding_ids()

    term_emb = np.load(TERM_EMB_PATH, mmap_mode="r")
    item_emb = np.load(ITEM_EMB_PATH, mmap_mode="r")

    out = pairs[keep_cols].copy()
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

    out[score_col] = scores
    return out


def build_train_test_scores() -> None:
    print("=" * 100)
    print("BUILD V5 TRAIN/TEST SCORES")
    print("=" * 100)

    if not TRAIN_SCORE_PATH.exists():
        print("Scoring train pairs...")
        train_pairs = pd.read_parquet(TRAIN_PAIRS_PATH)
        train_pairs["term_id"] = train_pairs["term_id"].astype(str)
        train_pairs["item_id"] = train_pairs["item_id"].astype(str)

        train_scores = score_pairs_df(
            train_pairs,
            score_col="embedding_score",
            keep_cols=["id", "term_id", "item_id", "label", "negative_type"],
        )

        train_scores.to_parquet(TRAIN_SCORE_PATH, index=False)
        print(f"Saved: {TRAIN_SCORE_PATH}")
        print(train_scores.groupby("label")["embedding_score"].describe())
        print(train_scores.groupby("negative_type")["embedding_score"].describe())
    else:
        print(f"Train score cache exists: {TRAIN_SCORE_PATH}")

    if not TEST_SCORE_PATH.exists():
        print("Scoring test pairs...")
        test_pairs = pl.read_csv(RAW_DIR / "submission_pairs.csv").to_pandas()
        test_pairs["term_id"] = test_pairs["term_id"].astype(str)
        test_pairs["item_id"] = test_pairs["item_id"].astype(str)

        test_scores = score_pairs_df(
            test_pairs,
            score_col="score",
            keep_cols=["id", "term_id", "item_id"],
        )

        test_scores.to_parquet(TEST_SCORE_PATH, index=False)
        print(f"Saved: {TEST_SCORE_PATH}")
        print(test_scores["score"].describe())
    else:
        print(f"Test score cache exists: {TEST_SCORE_PATH}")


def patch_v4_module_paths(mod):
    mod.TRAIN_SCORE_PATH = TRAIN_SCORE_PATH
    mod.TEST_SCORE_PATH = TEST_SCORE_PATH
    mod.RUN_NAME = RUN_NAME
    return mod


def build_realistic_eval_frame(mod, heldout_terms: set[str]) -> pd.DataFrame:
    print("Building V5 realistic eval frame...")

    train_scores = pd.read_parquet(TRAIN_SCORE_PATH)
    train_scores["term_id"] = train_scores["term_id"].astype(str)
    train_scores["item_id"] = train_scores["item_id"].astype(str)

    positives = train_scores[
        (train_scores["label"] == 1)
        & (train_scores["term_id"].isin(heldout_terms))
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

    neg["id"] = "V5REALNEG_" + neg["term_id"] + "_" + neg["item_id"]
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

    eval_scored = score_pairs_df(
        eval_pairs,
        score_col="embedding_score",
        keep_cols=["id", "term_id", "item_id", "label", "source"],
    )

    terms = pd.read_csv(RAW_DIR / "terms.csv")
    items = pd.read_csv(
        RAW_DIR / "items.csv",
        usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    )

    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    eval_df = eval_scored.merge(terms, on="term_id", how="left")
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


def run_realistic_cv(mod, full_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 100)
    print("RUN V5 REALISTIC CV")
    print("=" * 100)

    if CV_RAW_PATH.exists() and CV_SUMMARY_PATH.exists():
        print("CV cache exists, loading.")
        return pd.read_csv(CV_RAW_PATH), pd.read_csv(CV_SUMMARY_PATH)

    terms = np.array(sorted(full_train["term_id"].astype(str).unique()))
    feature_cols = mod.FEATURE_COLS
    cat_features = mod.CAT_FEATURES
    cat_indices = [feature_cols.index(c) for c in cat_features]

    all_rows = []

    for fold_id, seed in enumerate(CV_FOLD_SEEDS, start=1):
        print("\n" + "=" * 100)
        print(f"V5 FOLD {fold_id}, seed={seed}")
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
        eval_df = build_realistic_eval_frame(mod, valid_terms)

        print(f"Train rows: {len(train_df):,}")
        print(f"Eval rows : {len(eval_df):,}")
        print("Eval label ratio:")
        print(eval_df["label"].value_counts(normalize=True))

        y_true = eval_df["label"].astype(int).to_numpy()

        for semw in SEMANTIC_WEIGHTS:
            print("\n" + "-" * 100)
            print(f"Training V5 CV fold={fold_id}, semantic_weight={semw}")
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

            del model, train_pool, tr, proba
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    print("\nV5 CV top 40:")
    print(summary.head(40).to_string(index=False))

    return raw, summary


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


def train_full_and_make_submissions(mod, full_train: pd.DataFrame, test: pd.DataFrame, cv_summary: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("TRAIN V5 FULL ENSEMBLE + MAKE SUBMISSIONS")
    print("=" * 100)

    best = cv_summary.iloc[0]
    best_semw = float(best["semantic_weight"])
    best_th = float(best["threshold"])

    thresholds = sorted(set([
        round(best_th - 0.010, 3),
        round(best_th - 0.005, 3),
        round(best_th, 3),
        round(best_th + 0.005, 3),
        round(best_th + 0.010, 3),
    ]))

    thresholds = [th for th in thresholds if 0.80 <= th <= 0.995]

    print(f"Best CV semantic_weight={best_semw}, threshold={best_th}")
    print(f"Submission thresholds: {thresholds}")

    feature_cols = mod.FEATURE_COLS
    cat_features = mod.CAT_FEATURES
    cat_indices = [feature_cols.index(c) for c in cat_features]

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
        print(f"Training V5 full model seed={seed}, semw={best_semw}")
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    proba_avg = proba_sum / float(len(FINAL_SEEDS))
    proba_cols["proba_avg"] = proba_avg

    out = pd.DataFrame(
        {
            "id": test["id"].values,
            "term_id": test["term_id"].values,
            "item_id": test["item_id"].values,
            "embedding_score": test["embedding_score"].values,
        }
    )

    for col, arr in proba_cols.items():
        out[col] = arr

    out.to_parquet(FINAL_PROBA_PATH, index=False)
    print("Saved final proba:", FINAL_PROBA_PATH)

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    rows = []

    for name, proba in proba_cols.items():
        for th in thresholds:
            row = make_submission(sample, test["id"], proba, th, name)
            rows.append(row)
            print(row)

    summary = pd.DataFrame(rows).sort_values(["name", "threshold"])
    summary.to_csv(FINAL_SUMMARY_PATH, index=False)

    print("\nFinal submission summary:")
    print(summary.to_string(index=False))

    return summary


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("28 V5 E5 EMBEDDING CV THRESHOLD")
    print("=" * 100)

    required = [
        RAW_DIR / "terms.csv",
        RAW_DIR / "items.csv",
        RAW_DIR / "training_pairs.csv",
        RAW_DIR / "submission_pairs.csv",
        RAW_DIR / "sample_submission.csv",
        TRAIN_PAIRS_PATH,
        CANDIDATE_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    build_embeddings()
    build_train_test_scores()

    mod = load_v4_module()
    mod = patch_v4_module_paths(mod)

    print("Building V5 train frame...")
    full_train = mod.build_train_frame()

    print("Building V5 test frame...")
    test = mod.build_test_frame()

    print(f"Full train rows: {len(full_train):,}")
    print(f"Test rows      : {len(test):,}")

    cv_raw, cv_summary = run_realistic_cv(mod, full_train)
    final_summary = train_full_and_make_submissions(mod, full_train, test, cv_summary)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "term_emb_path": str(TERM_EMB_PATH),
        "item_emb_path": str(ITEM_EMB_PATH),
        "train_score_path": str(TRAIN_SCORE_PATH),
        "test_score_path": str(TEST_SCORE_PATH),
        "cv_raw_path": str(CV_RAW_PATH),
        "cv_summary_path": str(CV_SUMMARY_PATH),
        "final_proba_path": str(FINAL_PROBA_PATH),
        "final_summary_path": str(FINAL_SUMMARY_PATH),
        "best_cv": cv_summary.head(10).to_dict(orient="records"),
        "final_submissions": final_summary.to_dict(orient="records"),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved report:", REPORT_PATH)
    print("\nBest V5 CV:")
    print(cv_summary.head(20).to_string(index=False))
    print("\nFinal submissions:")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
