"""Raw-data-only PU lexical baseline for 16 GB RAM / RTX 4060 Laptop.

Stages are cached independently:

    python scripts/219_train_raw_pu_lexical.py prepare
    python scripts/219_train_raw_pu_lexical.py features
    python scripts/219_train_raw_pu_lexical.py train
    python scripts/219_train_raw_pu_lexical.py all

Raw labels are positive-only. Generated negative labels are synthetic and are never
reported as ground-truth competition Macro-F1. The final threshold is estimated by
fitting a positive/negative score mixture to the unlabeled test score distribution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from scipy.optimize import nnls
from sklearn.feature_extraction.text import HashingVectorizer


ROOT = Path(".")
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports/experiments"
SUBMISSIONS = ROOT / "submissions/final_candidates_v219"

ITEM_CACHE = PROC / "v219_items_lexical.parquet"
TRAIN_PAIR_CACHE = PROC / "v219_train_pu_pairs.parquet"
TRAIN_FEATURES = PROC / "v219_train_features.parquet"
TEST_FEATURES = PROC / "v219_test_features.parquet"
OOF_SCORES = PROC / "v219_oof_scores.parquet"
TEST_SCORES = PROC / "v219_test_scores.parquet"
MODEL_PATH = MODELS / "v219_raw_pu_lexical.cbm"
REPORT_PATH = REPORTS / "v219_raw_pu_lexical.json"

SEED = 20260710
PAIR_CHUNK = int(os.environ.get("V219_PAIR_CHUNK", "50000"))
N_FOLDS = int(os.environ.get("V219_N_FOLDS", "3"))
ITERATIONS = int(os.environ.get("V219_ITERATIONS", "1000"))

TR_MAP = str.maketrans(
    "çğıöşüâîûÇĞİÖŞÜÂÎÛ",
    "cgiosuaiuCGIOSUAIU",
)

FEATURES = [
    "word_title",
    "char_title",
    "word_full",
    "char_full",
    "word_category",
    "char_category",
    "query_len",
    "title_len",
    "full_len",
    "query_tokens",
    "title_tokens",
    "length_ratio",
    "query_in_title",
    "title_in_query",
    "brand_in_query",
    "all_query_digits_in_item",
    "digit_conflict",
]


def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().translate(TR_MAP)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def root_category(value: object) -> str:
    text = norm(value)
    return text.split(" ", 1)[0] if text else "unknown"


def stable_fold(term_id: object) -> int:
    digest = hashlib.blake2b(str(term_id).encode("utf8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % N_FOLDS


def require_raw() -> None:
    names = ["items.csv", "terms.csv", "training_pairs.csv", "submission_pairs.csv", "sample_submission.csv"]
    missing = [str(RAW / name) for name in names if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError("Missing raw inputs:\n- " + "\n- ".join(missing))


def prepare_items() -> pd.DataFrame:
    if ITEM_CACHE.exists():
        print("loading", ITEM_CACHE, flush=True)
        return pd.read_parquet(ITEM_CACHE)

    print("reading items.csv", flush=True)
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    items = pd.read_csv(RAW / "items.csv", usecols=columns, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)
    for column in columns[1:]:
        items[f"{column}_n"] = items[column].map(norm)
    items["root"] = items["category"].map(root_category)
    items["full_n"] = (
        items["title_n"]
        + " "
        + items["title_n"]
        + " "
        + items["brand_n"]
        + " "
        + items["category_n"]
        + " "
        + items["gender_n"]
        + " "
        + items["age_group_n"]
        + " "
        + items["attributes_n"].str.slice(0, 350)
    ).str.strip()
    keep = ["item_id", "title_n", "category_n", "brand_n", "root", "full_n"]
    items = items[keep].copy()
    ITEM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(ITEM_CACHE, index=False, compression="zstd")
    print("saved", ITEM_CACHE, len(items), flush=True)
    return items


def sample_valid(
    rng: np.random.Generator,
    candidates: np.ndarray,
    term_id: str,
    positive_set: set[tuple[str, str]],
    positive_item: str,
) -> str | None:
    if len(candidates) == 0:
        return None
    for _ in range(50):
        item_id = str(candidates[int(rng.integers(0, len(candidates)))])
        if item_id != positive_item and (term_id, item_id) not in positive_set:
            return item_id
    return None


def prepare_pairs(items: pd.DataFrame) -> pd.DataFrame:
    if TRAIN_PAIR_CACHE.exists():
        print("loading", TRAIN_PAIR_CACHE, flush=True)
        return pd.read_parquet(TRAIN_PAIR_CACHE)

    positives = pd.read_csv(RAW / "training_pairs.csv", usecols=["id", "term_id", "item_id", "label"])
    positives["term_id"] = positives["term_id"].astype(str)
    positives["item_id"] = positives["item_id"].astype(str)
    if set(positives["label"].unique()) != {1}:
        raise ValueError("V219 expects positive-only raw training labels")

    item_root = dict(zip(items["item_id"], items["root"]))
    root_groups = {
        str(key): group["item_id"].astype(str).to_numpy()
        for key, group in items.groupby("root", sort=False)
    }
    all_items = items["item_id"].astype(str).to_numpy()
    positive_set = set(zip(positives["term_id"], positives["item_id"]))
    rng = np.random.default_rng(SEED)
    negative_rows: list[tuple[str, str, str, int, str, float]] = []

    print("sampling random + same-root negatives", flush=True)
    for row_number, row in enumerate(positives.itertuples(index=False), start=1):
        term_id = str(row.term_id)
        positive_item = str(row.item_id)
        random_item = sample_valid(rng, all_items, term_id, positive_set, positive_item)
        same_root = root_groups.get(item_root.get(positive_item, "unknown"), np.empty(0, dtype=object))
        root_item = sample_valid(rng, same_root, term_id, positive_set, positive_item)
        if random_item is not None:
            negative_rows.append((f"V219_R_{row_number}", term_id, random_item, 0, "easy_random", 1.0))
        if root_item is not None:
            # Same-root negatives are more informative but noisier, so they get less weight.
            negative_rows.append((f"V219_H_{row_number}", term_id, root_item, 0, "same_root", 0.70))
        if row_number % 50000 == 0:
            print("negative sampling", row_number, "/", len(positives), flush=True)

    positives = positives[["id", "term_id", "item_id", "label"]].copy()
    positives["negative_type"] = "positive"
    positives["sample_weight"] = np.float32(1.0)
    negatives = pd.DataFrame(
        negative_rows,
        columns=["id", "term_id", "item_id", "label", "negative_type", "sample_weight"],
    )
    frame = pd.concat([positives, negatives], ignore_index=True)
    frame["fold"] = frame["term_id"].map(stable_fold).astype(np.int8)
    frame.to_parquet(TRAIN_PAIR_CACHE, index=False, compression="zstd")
    print(frame[["label", "negative_type"]].value_counts(), flush=True)
    print("saved", TRAIN_PAIR_CACHE, len(frame), flush=True)
    return frame


def vectorizers() -> tuple[HashingVectorizer, HashingVectorizer]:
    word = HashingVectorizer(
        n_features=2**18,
        analyzer="word",
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    char = HashingVectorizer(
        n_features=2**19,
        analyzer="char_wb",
        ngram_range=(3, 5),
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    return word, char


def row_cosine(vectorizer: HashingVectorizer, left: pd.Series, right: pd.Series) -> np.ndarray:
    a = vectorizer.transform(left.fillna("").tolist())
    b = vectorizer.transform(right.fillna("").tolist())
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)


def digit_features(query: str, item: str) -> tuple[int, int]:
    query_digits = set(re.findall(r"\d+(?:[.,]\d+)?", query))
    item_digits = set(re.findall(r"\d+(?:[.,]\d+)?", item))
    if not query_digits:
        return 0, 0
    return int(query_digits <= item_digits), int(bool(query_digits) and bool(item_digits) and not query_digits <= item_digits)


def feature_chunk(
    pairs: pd.DataFrame,
    query_map: dict[str, str],
    items_indexed: pd.DataFrame,
    seen_items: set[str],
    word: HashingVectorizer,
    char: HashingVectorizer,
) -> pd.DataFrame:
    out = pairs.copy()
    out["term_id"] = out["term_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["query_n"] = out["term_id"].map(query_map).fillna("")
    meta = items_indexed.reindex(out["item_id"].to_numpy()).reset_index(drop=True)
    for column in ["title_n", "category_n", "brand_n", "full_n"]:
        out[column] = meta[column].fillna("").to_numpy()

    out["word_title"] = row_cosine(word, out["query_n"], out["title_n"])
    out["char_title"] = row_cosine(char, out["query_n"], out["title_n"])
    out["word_full"] = row_cosine(word, out["query_n"], out["full_n"])
    out["char_full"] = row_cosine(char, out["query_n"], out["full_n"])
    out["word_category"] = row_cosine(word, out["query_n"], out["category_n"])
    out["char_category"] = row_cosine(char, out["query_n"], out["category_n"])
    out["query_len"] = out["query_n"].str.len().astype(np.float32)
    out["title_len"] = out["title_n"].str.len().astype(np.float32)
    out["full_len"] = out["full_n"].str.len().astype(np.float32)
    out["query_tokens"] = out["query_n"].str.count(" ").add(1).where(out["query_n"].ne(""), 0).astype(np.float32)
    out["title_tokens"] = out["title_n"].str.count(" ").add(1).where(out["title_n"].ne(""), 0).astype(np.float32)
    out["length_ratio"] = (
        np.minimum(out["query_len"], out["title_len"]) / np.maximum(out["query_len"], out["title_len"]).clip(lower=1)
    ).astype(np.float32)
    out["query_in_title"] = np.fromiter(
        (int(bool(q) and q in t) for q, t in zip(out["query_n"], out["title_n"])), dtype=np.int8, count=len(out)
    )
    out["title_in_query"] = np.fromiter(
        (int(bool(t) and t in q) for q, t in zip(out["query_n"], out["title_n"])), dtype=np.int8, count=len(out)
    )
    out["brand_in_query"] = np.fromiter(
        (int(bool(b) and b in q) for q, b in zip(out["query_n"], out["brand_n"])), dtype=np.int8, count=len(out)
    )
    digits = [digit_features(q, t) for q, t in zip(out["query_n"], out["full_n"])]
    out["all_query_digits_in_item"] = np.fromiter((x[0] for x in digits), dtype=np.int8, count=len(out))
    out["digit_conflict"] = np.fromiter((x[1] for x in digits), dtype=np.int8, count=len(out))
    out["item_seen_positive"] = out["item_id"].isin(seen_items).astype(np.int8)

    identity = [column for column in ["id", "term_id", "item_id", "label", "fold", "negative_type", "sample_weight"] if column in out]
    return out[identity + FEATURES]


def write_feature_cache(
    source: Path,
    destination: Path,
    query_map: dict[str, str],
    items_indexed: pd.DataFrame,
    seen_items: set[str],
    is_train: bool,
) -> None:
    if destination.exists() and os.environ.get("V219_FORCE_FEATURES", "0") != "1":
        print("using cached", destination, flush=True)
        return

    temporary = destination.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    word, char = vectorizers()
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        if is_train:
            iterator = [pd.read_parquet(source)]
        else:
            iterator = pd.read_csv(source, chunksize=PAIR_CHUNK, dtype=str)
        for source_frame in iterator:
            for start in range(0, len(source_frame), PAIR_CHUNK):
                chunk = source_frame.iloc[start : start + PAIR_CHUNK].copy()
                features = feature_chunk(chunk, query_map, items_indexed, seen_items, word, char)
                table = pa.Table.from_pandas(features, preserve_index=False)
                if writer is None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                row_count += len(features)
                print(destination.name, row_count, flush=True)
                del chunk, features, table
                gc.collect()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"No feature rows were written for {source}")
    temporary.replace(destination)
    print("saved", destination, row_count, flush=True)


def build_features(items: pd.DataFrame, train_pairs: pd.DataFrame) -> None:
    terms = pd.read_csv(RAW / "terms.csv", usecols=["term_id", "query"])
    terms["term_id"] = terms["term_id"].astype(str)
    query_map = dict(zip(terms["term_id"], terms["query"].map(norm)))
    items_indexed = items.set_index("item_id", drop=True)
    seen_items = set(train_pairs.loc[train_pairs["label"].eq(1), "item_id"].astype(str))
    write_feature_cache(TRAIN_PAIR_CACHE, TRAIN_FEATURES, query_map, items_indexed, seen_items, True)
    write_feature_cache(RAW / "submission_pairs.csv", TEST_FEATURES, query_map, items_indexed, seen_items, False)


def catboost_params(iterations: int) -> dict:
    return {
        "loss_function": "Logloss",
        "iterations": iterations,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 8.0,
        "random_seed": SEED,
        "verbose": 100,
        "allow_writing_files": False,
        "task_type": "GPU",
        "devices": "0",
    }


def fit_model(x: pd.DataFrame, y: np.ndarray, weights: np.ndarray, iterations: int) -> CatBoostClassifier:
    params = catboost_params(iterations)
    model = CatBoostClassifier(**params)
    try:
        model.fit(x, y, sample_weight=weights)
    except Exception as exc:
        print("GPU fallback:", repr(exc), flush=True)
        params["task_type"] = "CPU"
        params.pop("devices", None)
        params["thread_count"] = max(1, (os.cpu_count() or 4) - 2)
        model = CatBoostClassifier(**params)
        model.fit(x, y, sample_weight=weights)
    return model


def histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=bins)[0].astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def macro_from_rates(prior: float, tpr: float, fpr: float) -> float:
    tp, fn = prior * tpr, prior * (1.0 - tpr)
    fp, tn = (1.0 - prior) * fpr, (1.0 - prior) * (1.0 - fpr)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    f1_neg = 2 * tn / max(2 * tn + fp + fn, 1e-12)
    return float((f1_pos + f1_neg) * 0.5)


def estimate_threshold(oof: pd.DataFrame, test_score: np.ndarray) -> dict:
    bins = np.linspace(0.0, 1.0, 501)
    positive_hist = histogram(oof.loc[oof["label"].eq(1), "score"].to_numpy(), bins)
    # Easy random negatives better approximate the dominant irrelevant test component.
    random_hist = histogram(oof.loc[oof["negative_type"].eq("easy_random"), "score"].to_numpy(), bins)
    test_hist = histogram(test_score, bins)
    matrix = np.stack([random_hist, positive_hist], axis=1)
    weights, _ = nnls(
        np.vstack([matrix, np.ones((1, 2)) * 10.0]),
        np.concatenate([test_hist, [10.0]]),
    )
    weights /= max(weights.sum(), 1e-12)
    prior = float(np.clip(weights[1], 0.03, 0.60))
    positive_tail = np.cumsum(positive_hist[::-1])[::-1]
    negative_tail = np.cumsum(random_hist[::-1])[::-1]
    candidates = []
    for index in range(len(positive_tail)):
        threshold = float(bins[index])
        tpr = float(positive_tail[index])
        fpr = float(negative_tail[index])
        candidates.append((macro_from_rates(prior, tpr, fpr), threshold, tpr, fpr))
    score, threshold, tpr, fpr = max(candidates)
    return {
        "estimated_positive_prior": prior,
        "threshold": threshold,
        "estimated_macro_f1": score,
        "estimated_tpr": tpr,
        "estimated_fpr": fpr,
        "mixture_rmse": float(np.sqrt(np.mean((matrix @ weights - test_hist) ** 2))),
    }


def train_and_submit() -> None:
    train = pd.read_parquet(TRAIN_FEATURES)
    train[FEATURES] = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = train["label"].astype(np.int8).to_numpy()
    weights = train["sample_weight"].astype(np.float32).to_numpy()
    folds = train["fold"].astype(np.int8).to_numpy()
    oof_score = np.empty(len(train), dtype=np.float32)
    best_iterations = []

    for fold in range(N_FOLDS):
        tr = folds != fold
        va = folds == fold
        model = fit_model(train.loc[tr, FEATURES], y[tr], weights[tr], ITERATIONS)
        oof_score[va] = model.predict_proba(train.loc[va, FEATURES])[:, 1].astype(np.float32)
        best_iteration = model.get_best_iteration()
        best_iterations.append(ITERATIONS if best_iteration is None or best_iteration <= 0 else best_iteration)
        print({"fold": fold, "train": int(tr.sum()), "valid": int(va.sum())}, flush=True)

    oof = train[["id", "term_id", "item_id", "label", "fold", "negative_type"]].copy()
    oof["score"] = oof_score
    OOF_SCORES.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OOF_SCORES, index=False, compression="zstd")

    final_iterations = max(300, int(np.median(best_iterations)))
    final_model = fit_model(train[FEATURES], y, weights, final_iterations)
    MODELS.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_PATH)
    del train
    gc.collect()

    test = pd.read_parquet(TEST_FEATURES)
    test[FEATURES] = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    test_score = np.empty(len(test), dtype=np.float32)
    for start in range(0, len(test), 250000):
        end = min(start + 250000, len(test))
        test_score[start:end] = final_model.predict_proba(test.iloc[start:end][FEATURES])[:, 1].astype(np.float32)
        print("predict", end, "/", len(test), flush=True)

    calibration = estimate_threshold(oof, test_score)
    threshold = calibration["threshold"]
    prediction = (test_score >= threshold).astype(np.int8)
    test_out = test[["id", "term_id", "item_id"]].copy()
    test_out["v219_score"] = test_score
    test_out.to_parquet(TEST_SCORES, index=False, compression="zstd")

    sample = pd.read_csv(RAW / "sample_submission.csv", usecols=["id"])
    if not sample["id"].astype(str).reset_index(drop=True).equals(test["id"].astype(str).reset_index(drop=True)):
        raise ValueError("sample/test ID alignment mismatch")
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    output = SUBMISSIONS / "FINAL_CANDIDATE_v219_pu_mixture.csv"
    pd.DataFrame({"id": sample["id"], "prediction": prediction}).to_csv(output, index=False)

    report = {
        "hardware_profile": "RTX 4060 Laptop / 16 GB RAM / 13th gen i7",
        "features": FEATURES,
        "folds": N_FOLDS,
        "iterations": final_iterations,
        "rows": {"oof": len(oof), "test": len(test)},
        "calibration": calibration,
        "submission_positive_ratio": float(prediction.mean()),
        "submission": str(output),
        "warning": (
            "Raw labels are positive-only. Negative labels are synthetic; estimated Macro-F1 and prior "
            "depend on the random-negative mixture assumption and are not ground-truth leaderboard metrics."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "features", "train", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_raw()
    if args.stage in {"prepare", "all"}:
        items = prepare_items()
        prepare_pairs(items)
        del items
        gc.collect()
    if args.stage in {"features", "all"}:
        items = prepare_items()
        pairs = prepare_pairs(items)
        build_features(items, pairs)
        del items, pairs
        gc.collect()
    if args.stage in {"train", "all"}:
        if not TRAIN_FEATURES.exists() or not TEST_FEATURES.exists():
            raise FileNotFoundError("Run the V219 features stage first")
        train_and_submit()


if __name__ == "__main__":
    main()
