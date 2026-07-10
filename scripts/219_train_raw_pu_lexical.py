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
from catboost import CatBoostClassifier, Pool
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

UNITS = {"gb", "tb", "mb", "mah", "w", "wh", "hz", "inch", "inc", "cm", "mm", "ml", "lt", "kg", "gr"}
ACCESSORY_TOKENS = {
    "kilif", "case", "kapak", "koruyucu", "kablo", "adapter", "adaptoru",
    "askisi", "stand", "tutucu", "filtre", "firca", "yedek", "kartus", "toner", "uc",
}
MAIN_DEVICE_TOKENS = {
    "telefon", "iphone", "tablet", "ipad", "laptop", "notebook", "bilgisayar", "monitor",
    "televizyon", "tv", "yazici", "printer", "supurge", "saat", "watch", "kulaklik",
}

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
    "query_has_specs",
    "all_query_digits_in_item",
    "digit_conflict",
    "spec_overlap_ratio",
    "spec_conflict",
    "unit_overlap_ratio",
    "unit_conflict",
    "query_main_item_accessory",
    "query_accessory_item_main",
]


def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().translate(TR_MAP)
    # Preserve decimal/model boundaries before punctuation cleanup: 12.5 -> 12p5.
    text = re.sub(r"(?<=\d)[.,](?=\d)", "p", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def root_category(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    # Category hierarchy separators must be handled before text normalization.
    first = re.split(r"[/|>]", str(value), maxsplit=1)[0]
    return norm(first) or "unknown"


def token_set(text: str) -> set[str]:
    return set(text.split()) if text else set()


def phrase_in_text(phrase: str, text: str) -> int:
    if not phrase or not text:
        return 0
    return int(f" {phrase} " in f" {text} ")


def spec_set(text: str) -> set[str]:
    # Product/model tokens: 128, 12p5, a55, s24, 205/55 fragments after normalization.
    return {token for token in text.split() if any(char.isdigit() for char in token)}


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return float(len(left & right) / len(left))


def stable_fold(term_id: object) -> int:
    digest = hashlib.blake2b(str(term_id).encode("utf8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % N_FOLDS


def require_raw() -> None:
    names = ["items.csv", "terms.csv", "training_pairs.csv", "submission_pairs.csv", "sample_submission.csv"]
    missing = [str(RAW / name) for name in names if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError("Missing raw inputs:\n- " + "\n- ".join(missing))


def prepare_items() -> pd.DataFrame:
    force = os.environ.get("V219_FORCE_PREPARE", "0") == "1"
    if ITEM_CACHE.exists() and not force:
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
    used_negatives: set[tuple[str, str]],
) -> str | None:
    if len(candidates) == 0:
        return None
    for _ in range(50):
        item_id = str(candidates[int(rng.integers(0, len(candidates)))])
        key = (term_id, item_id)
        if item_id != positive_item and key not in positive_set and key not in used_negatives:
            return item_id
    return None


def prepare_pairs(items: pd.DataFrame) -> pd.DataFrame:
    force = os.environ.get("V219_FORCE_PREPARE", "0") == "1"
    if TRAIN_PAIR_CACHE.exists() and not force:
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
    used_negatives: set[tuple[str, str]] = set()
    rng = np.random.default_rng(SEED)
    negative_rows: list[tuple[str, str, str, int, str, float]] = []

    print("sampling random + same-root negatives", flush=True)
    for row_number, row in enumerate(positives.itertuples(index=False), start=1):
        term_id = str(row.term_id)
        positive_item = str(row.item_id)
        random_item = sample_valid(rng, all_items, term_id, positive_set, positive_item, used_negatives)
        if random_item is not None:
            used_negatives.add((term_id, random_item))
        same_root = root_groups.get(item_root.get(positive_item, "unknown"), np.empty(0, dtype=object))
        root_item = sample_valid(rng, same_root, term_id, positive_set, positive_item, used_negatives)
        if root_item is not None:
            used_negatives.add((term_id, root_item))
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
    query_digits = set(re.findall(r"\d+(?:p\d+)?", query))
    item_digits = set(re.findall(r"\d+(?:p\d+)?", item))
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
        (phrase_in_text(b, q) for q, b in zip(out["query_n"], out["brand_n"])), dtype=np.int8, count=len(out)
    )
    digits = [digit_features(q, t) for q, t in zip(out["query_n"], out["full_n"])]
    out["all_query_digits_in_item"] = np.fromiter((x[0] for x in digits), dtype=np.int8, count=len(out))
    out["digit_conflict"] = np.fromiter((x[1] for x in digits), dtype=np.int8, count=len(out))
    query_specs = [spec_set(text) for text in out["query_n"]]
    item_specs = [spec_set(text) for text in out["full_n"]]
    out["query_has_specs"] = np.fromiter((int(bool(x)) for x in query_specs), dtype=np.int8, count=len(out))
    out["spec_overlap_ratio"] = np.fromiter(
        (overlap_ratio(q, i) for q, i in zip(query_specs, item_specs)), dtype=np.float32, count=len(out)
    )
    out["spec_conflict"] = np.fromiter(
        (int(bool(q) and bool(i) and not q <= i) for q, i in zip(query_specs, item_specs)),
        dtype=np.int8,
        count=len(out),
    )
    query_tokens = [token_set(text) for text in out["query_n"]]
    item_tokens = [token_set(text) for text in out["full_n"]]
    query_units = [tokens & UNITS for tokens in query_tokens]
    item_units = [tokens & UNITS for tokens in item_tokens]
    out["unit_overlap_ratio"] = np.fromiter(
        (overlap_ratio(q, i) for q, i in zip(query_units, item_units)), dtype=np.float32, count=len(out)
    )
    out["unit_conflict"] = np.fromiter(
        (int(bool(q) and bool(i) and not q <= i) for q, i in zip(query_units, item_units)),
        dtype=np.int8,
        count=len(out),
    )
    out["query_main_item_accessory"] = np.fromiter(
        (
            int(bool(q & MAIN_DEVICE_TOKENS) and not bool(q & ACCESSORY_TOKENS) and bool(i & ACCESSORY_TOKENS))
            for q, i in zip(query_tokens, item_tokens)
        ),
        dtype=np.int8,
        count=len(out),
    )
    out["query_accessory_item_main"] = np.fromiter(
        (
            int(bool(q & ACCESSORY_TOKENS) and bool(i & MAIN_DEVICE_TOKENS) and not bool(i & ACCESSORY_TOKENS))
            for q, i in zip(query_tokens, item_tokens)
        ),
        dtype=np.int8,
        count=len(out),
    )
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


def fit_model(
    x: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    valid_x: pd.DataFrame | None = None,
    valid_y: np.ndarray | None = None,
    valid_weights: np.ndarray | None = None,
) -> CatBoostClassifier:
    params = catboost_params(iterations)
    model = CatBoostClassifier(**params)
    fit_kwargs = {}
    if valid_x is not None and valid_y is not None:
        valid_pool = Pool(valid_x, label=valid_y, weight=valid_weights)
        fit_kwargs = {
            "eval_set": valid_pool,
            "use_best_model": True,
            "early_stopping_rounds": 100,
        }
    try:
        model.fit(x, y, sample_weight=weights, **fit_kwargs)
    except Exception as exc:
        print("GPU fallback:", repr(exc), flush=True)
        params["task_type"] = "CPU"
        params.pop("devices", None)
        params["thread_count"] = max(1, (os.cpu_count() or 4) - 2)
        model = CatBoostClassifier(**params)
        model.fit(x, y, sample_weight=weights, **fit_kwargs)
    return model


def predict_chunks(model: CatBoostClassifier, frame: pd.DataFrame, chunk_size: int = 250000) -> np.ndarray:
    output = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        end = min(start + chunk_size, len(frame))
        output[start:end] = model.predict_proba(frame.iloc[start:end][FEATURES])[:, 1].astype(np.float32)
    return output


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
    random_hist = histogram(oof.loc[oof["negative_type"].eq("easy_random"), "score"].to_numpy(), bins)
    test_hist = histogram(test_score, bins)
    # Current binary target accepts both fully and partially relevant products.
    # Same-root unlabeled rows may contain that partially relevant class, therefore
    # they must not define the negative component of the binary calibration.
    matrix = np.stack([random_hist, positive_hist], axis=1)
    weights, _ = nnls(
        np.vstack([matrix, np.ones((1, 2)) * 10.0]),
        np.concatenate([test_hist, [10.0]]),
    )
    weights /= max(weights.sum(), 1e-12)
    prior = float(np.clip(weights[1], 0.03, 0.80))
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
        "mixture_irrelevant_weight": float(weights[0]),
        "mixture_relevant_or_partial_weight_raw": float(weights[1]),
        "threshold": threshold,
        "estimated_macro_f1": score,
        "estimated_tpr": tpr,
        "estimated_fpr": fpr,
        "mixture_rmse": float(np.sqrt(np.mean((matrix @ weights - test_hist) ** 2))),
    }


def train_and_submit() -> None:
    hard_negative_mode = os.environ.get("V219_HARD_NEGATIVE", "0") == "1"
    run_tag = "v220_crossfit_hard_negative" if hard_negative_mode else "v219"
    oof_output = PROC / f"{run_tag}_oof_scores.parquet"
    test_score_output = PROC / f"{run_tag}_test_scores.parquet"
    model_output = MODELS / f"{run_tag}_raw_pu_lexical.cbm"
    report_output = REPORTS / f"{run_tag}_raw_pu_lexical.json"
    submission_dir = ROOT / f"submissions/final_candidates_{'v220' if hard_negative_mode else 'v219'}"

    train = pd.read_parquet(TRAIN_FEATURES)
    train[FEATURES] = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = train["label"].astype(np.int8).to_numpy(copy=True)
    # Arrow-backed parquet columns may expose a read-only NumPy view. We mutate
    # weights for the binary partial-relevance policy, so require an owned array.
    weights = train["sample_weight"].astype(np.float32).to_numpy(copy=True)
    # The raw labels only identify positives. Same-root samples are unlabeled and
    # can be partially relevant, which is positive in the current binary stage.
    # Keep only the lexically weakest tail as low-confidence irrelevant examples.
    same_root_mask = train["negative_type"].eq("same_root").to_numpy()
    lexical_proxy = (
        0.35 * train["char_full"].to_numpy(dtype=np.float32)
        + 0.30 * train["word_full"].to_numpy(dtype=np.float32)
        + 0.20 * train["char_title"].to_numpy(dtype=np.float32)
        + 0.15 * train["word_title"].to_numpy(dtype=np.float32)
    )
    same_root_cutoff = float(np.quantile(lexical_proxy[same_root_mask], 0.35))
    same_root_positive_cutoff = float(np.quantile(lexical_proxy[same_root_mask], 0.75))
    weak_same_root = same_root_mask & (lexical_proxy <= same_root_cutoff)
    partial_positive = same_root_mask & (lexical_proxy >= same_root_positive_cutoff)
    weights[same_root_mask] = np.float32(0.0)
    weights[weak_same_root] = np.float32(0.25)
    # The highest-similarity same-root tail is a conservative proxy for the
    # partially relevant class, which is positive in the current binary stage.
    y[partial_positive] = np.int8(1)
    weights[partial_positive] = np.float32(0.35)
    hard_negative_report = None
    if hard_negative_mode:
        if not OOF_SCORES.exists():
            raise FileNotFoundError(
                f"Hard-negative mode needs the baseline cross-fit scores: {OOF_SCORES}"
            )
        previous_oof = pd.read_parquet(OOF_SCORES, columns=["id", "score"])
        if len(previous_oof) != len(train) or not previous_oof["id"].astype(str).reset_index(drop=True).equals(
            train["id"].astype(str).reset_index(drop=True)
        ):
            raise ValueError("V219 OOF/train row alignment mismatch; refusing hard-negative mining")
        previous_score = previous_oof["score"].astype(np.float32).to_numpy()
        easy_mask = train["negative_type"].eq("easy_random").to_numpy()
        easy_cutoff = float(np.quantile(previous_score[easy_mask], 0.90))
        weak_cutoff = float(np.quantile(previous_score[weak_same_root], 0.90))
        hard_easy = easy_mask & (previous_score >= easy_cutoff)
        hard_weak_same_root = weak_same_root & (previous_score >= weak_cutoff)
        # Easy-random labels are the safest negatives. The weak same-root subset is
        # already the bottom lexical tail, so boosting only its OOF false positives
        # is conservative with respect to partially relevant products.
        weights[hard_easy] = np.float32(2.0)
        weights[hard_weak_same_root] = np.float32(0.75)
        hard_negative_report = {
            "source": str(OOF_SCORES),
            "selection": "previous term-grouped OOF false positives",
            "easy_random_quantile": 0.90,
            "easy_random_score_cutoff": easy_cutoff,
            "easy_random_rows": int(hard_easy.sum()),
            "easy_random_weight": 2.0,
            "weak_same_root_quantile": 0.90,
            "weak_same_root_score_cutoff": weak_cutoff,
            "weak_same_root_rows": int(hard_weak_same_root.sum()),
            "weak_same_root_weight": 0.75,
            "protected_partial_positive_rows": int(partial_positive.sum()),
            "protected_unlabeled_rows": int((same_root_mask & ~weak_same_root & ~partial_positive).sum()),
        }
    print(
        {
            "binary_target": "relevant_or_partially_relevant_vs_irrelevant",
            "same_root_unlabeled_zero_weight": int(
                (same_root_mask & ~weak_same_root & ~partial_positive).sum()
            ),
            "same_root_weak_negative": int(weak_same_root.sum()),
            "same_root_partial_positive": int(partial_positive.sum()),
            "same_root_proxy_cutoff": same_root_cutoff,
            "same_root_partial_positive_cutoff": same_root_positive_cutoff,
            "hard_negative_mode": hard_negative_mode,
            "hard_negative_report": hard_negative_report,
        },
        flush=True,
    )
    folds = train["fold"].astype(np.int8).to_numpy()
    oof_score = np.empty(len(train), dtype=np.float32)
    best_iterations = []
    test = pd.read_parquet(TEST_FEATURES)
    test[FEATURES] = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    fold_test_scores = []

    for fold in range(N_FOLDS):
        tr = folds != fold
        va = folds == fold
        model = fit_model(
            train.loc[tr, FEATURES],
            y[tr],
            weights[tr],
            ITERATIONS,
            valid_x=train.loc[va, FEATURES],
            valid_y=y[va],
            valid_weights=weights[va],
        )
        oof_score[va] = model.predict_proba(train.loc[va, FEATURES])[:, 1].astype(np.float32)
        fold_test_scores.append(predict_chunks(model, test))
        best_iteration = model.get_best_iteration()
        best_iterations.append(ITERATIONS if best_iteration is None or best_iteration <= 0 else best_iteration)
        print({"fold": fold, "train": int(tr.sum()), "valid": int(va.sum())}, flush=True)
        del model
        gc.collect()

    oof = train[["id", "term_id", "item_id", "label", "fold", "negative_type"]].copy()
    oof = oof.rename(columns={"label": "raw_label"})
    oof["label"] = y
    oof["training_weight"] = weights
    oof["partial_positive"] = partial_positive.astype(np.int8)
    oof["score"] = oof_score
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_output, index=False, compression="zstd")

    final_iterations = max(300, int(np.median(best_iterations)))
    final_model = fit_model(train[FEATURES], y, weights, final_iterations)
    MODELS.mkdir(parents=True, exist_ok=True)
    final_model.save_model(model_output)
    del train
    gc.collect()

    # OOF and test probabilities now come from models trained on the same fold
    # proportions, so the selected threshold is applied on a compatible scale.
    test_score = np.mean(np.stack(fold_test_scores), axis=0).astype(np.float32)
    del fold_test_scores
    gc.collect()

    calibration = estimate_threshold(oof, test_score)
    threshold = calibration["threshold"]
    prediction = (test_score >= threshold).astype(np.int8)
    test_out = test[["id", "term_id", "item_id"]].copy()
    test_out["v219_score"] = test_score
    test_out.to_parquet(test_score_output, index=False, compression="zstd")

    sample = pd.read_csv(RAW / "sample_submission.csv", usecols=["id"])
    if not sample["id"].astype(str).reset_index(drop=True).equals(test["id"].astype(str).reset_index(drop=True)):
        raise ValueError("sample/test ID alignment mismatch")
    submission_dir.mkdir(parents=True, exist_ok=True)
    output = submission_dir / f"FINAL_CANDIDATE_{run_tag}_pu_mixture.csv"
    pd.DataFrame({"id": sample["id"], "prediction": prediction}).to_csv(output, index=False)

    prior_outputs = {}
    estimated_prior = float(calibration["estimated_positive_prior"])
    for offset in (-0.03, 0.0, 0.03):
        target_prior = float(np.clip(estimated_prior + offset, 0.01, 0.80))
        prior_threshold = float(np.quantile(test_score, 1.0 - target_prior))
        prior_prediction = (test_score >= prior_threshold).astype(np.int8)
        tag = str(round(target_prior, 3)).replace(".", "p")
        prior_path = submission_dir / f"CANDIDATE_{run_tag}_prior_{tag}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": prior_prediction}).to_csv(prior_path, index=False)
        prior_outputs[tag] = {
            "path": str(prior_path),
            "target_prior": target_prior,
            "threshold": prior_threshold,
            "positive_ratio": float(prior_prediction.mean()),
        }

    report = {
        "hardware_profile": "RTX 4060 Laptop / 16 GB RAM / 13th gen i7",
        "run_tag": run_tag,
        "features": FEATURES,
        "folds": N_FOLDS,
        "iterations": final_iterations,
        "rows": {"oof": len(oof), "test": len(test)},
        "calibration": calibration,
        "submission_positive_ratio": float(prediction.mean()),
        "submission": str(output),
        "prior_candidates": prior_outputs,
        "test_prediction_source": "mean probability from term-grouped fold models",
        "hard_negative_mining": hard_negative_report,
        "binary_target": "relevant_or_partially_relevant_vs_irrelevant",
        "same_root_policy": {
            "interpretation": "unlabeled because partially relevant is positive in this stage",
            "weak_negative_quantile": 0.35,
            "weak_negative_weight": 0.25,
            "partial_positive_quantile": 0.75,
            "partial_positive_weight": 0.35,
            "lexical_proxy_cutoff": same_root_cutoff,
            "partial_positive_lexical_cutoff": same_root_positive_cutoff,
            "zero_weight_rows": int((same_root_mask & ~weak_same_root & ~partial_positive).sum()),
            "weak_negative_rows": int(weak_same_root.sum()),
            "partial_positive_rows": int(partial_positive.sum()),
        },
        "warning": (
            "Raw labels are positive-only. Negative labels are synthetic; estimated Macro-F1 and prior "
            "depend on the random-negative mixture assumption and are not ground-truth leaderboard metrics."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
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
