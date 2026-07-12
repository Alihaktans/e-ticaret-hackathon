"""Trendyol relevance — tek dosyalık, uçtan uca yarışma pipeline'ı.

Bu dosya proje içindeki başka hiçbir ``.py`` dosyasını import etmez. Altı veri
dosyasından başlayarak V219 PU lexical modeli, cross-fit hard-negative V220,
Trendyol embedding destekli V221 ve guarded-asymmetric V233 adayını üretir.

Gerekli girdiler:
    data/raw/items.csv
    data/raw/terms.csv
    data/raw/training_pairs.csv
    data/raw/submission_pairs.csv
    data/raw/sample_submission.csv
    submissions/reference/FINAL_CANDIDATE_v104_full_balanced.csv

Tam çalıştırma:
    python trendyol_single_file_pipeline.py all

Ara aşamalar ayrıca tek tek çalıştırılabilir. Parquet/NumPy cache'leri yalnızca
yeniden çalıştırmayı hızlandırır; sıfırdan çalışmak için zorunlu değildir.

Önemli bilimsel sınır: Ham train etiketleri yalnızca pozitiftir. Sentetik
negatiflerden hesaplanan offline metrikler yarışma ground-truth metriği değildir.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ağır ML bağımlılıkları isteğe bağlı import edilir. Böylece ``--help`` eksik
# ortamda bile çalışır; gerçek aşama başlamadan ``audit`` açık bir paket listesi
# verir. Tek dosya olmak, üçüncü taraf ML paketlerini dosyaya gömmek anlamına
# gelmez.
IMPORT_ERRORS: dict[str, str] = {}
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    pa = None
    pq = None
    IMPORT_ERRORS["pyarrow"] = repr(exc)
try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:
    CatBoostClassifier = None
    Pool = None
    IMPORT_ERRORS["catboost"] = repr(exc)
try:
    from scipy.optimize import nnls
except ImportError as exc:
    nnls = None
    IMPORT_ERRORS["scipy"] = repr(exc)
try:
    from sklearn.feature_extraction.text import HashingVectorizer
except ImportError as exc:
    HashingVectorizer = None
    IMPORT_ERRORS["scikit-learn"] = repr(exc)


# %% [markdown]
# 0. Pipeline haritası
# --------------------
# 1) Ham veri denetimi
# 2) Ürün normalizasyonu ve PU negatif örnekleme
# 3) Hashing-TF-IDF-benzeri lexical özellikler
# 4) V219 term-grouped OOF CatBoost
# 5) V220 cross-fit hard-negative CatBoost
# 6) V104/V219/V220 disagreement havuzu
# 7) Trendyol embedding ve V221 guarded swap
# 8) V221 add kuyruğunda kesin çatışma korumalı V233


# %% [markdown]
# 1. Ortak yollar, donanım ayarları ve özellik sözleşmesi


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
PAIR_CHUNK = int(os.environ.get("V219_PAIR_CHUNK", "25000"))
N_FOLDS = int(os.environ.get("V219_N_FOLDS", "3"))
ITERATIONS = int(os.environ.get("V219_ITERATIONS", "1200"))

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


# %% [markdown]
# 2. Metin normalizasyonu ve lexical yardımcılar


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


# %% [markdown]
# 3. Ham ürünleri hazırlama
# -------------------------
# Başlık, kategori, marka ve özellikler normalize edilerek tek ürün cache'ine
# yazılır. Cache yoksa ham items.csv'den kendisi üretir.


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


# %% [markdown]
# 4. Positive–Unlabeled eğitim çiftleri
# --------------------------------------
# Her gerçek pozitif için bir kolay rastgele ve bir aynı-kök aday oluşturulur.
# Aynı-kök örneklerin bir kısmı yarı alakalı olabileceği için daha sonra zayıf
# veya etiketsiz ağırlıklarla ele alınır.


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


# %% [markdown]
# 5. Lexical özellik üretimi
# --------------------------
# HashingVectorizer sabit bellekle word/character n-gram cosine benzerlikleri
# üretir. 3.36 milyon test satırı chunk'lar halinde yazılır.


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


# %% [markdown]
# 6. Term-grouped CatBoost ve cross-fit hard-negative


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


# %% [markdown]
# 7. V104/V219/V220 disagreement ve Trendyol embedding


ROOT = Path(".")
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
REPORTS = ROOT / "reports/experiments"
OUT_DIR = ROOT / "submissions/final_candidates_v221"

TEACHER = ROOT / "submissions/reference/FINAL_CANDIDATE_v104_full_balanced.csv"
V219_SCORE = PROC / "v219_test_scores.parquet"
V220_SCORE = PROC / "v220_crossfit_hard_negative_test_scores.parquet"
PAIRS = RAW / "submission_pairs.csv"
TERMS = RAW / "terms.csv"
ITEMS = RAW / "items.csv"

DISAGREEMENT = PROC / "v221_v104_v220_disagreement.parquet"
QUERY_IDS = PROC / "v221_query_ids.parquet"
ITEM_IDS = PROC / "v221_item_ids.parquet"
QUERY_EMB = PROC / "v221_query_embeddings_128.npy"
ITEM_EMB = PROC / "v221_item_embeddings_128.npy"
SCORED = PROC / "v221_embedding_guarded_disagreement.parquet"
REPORT = REPORTS / "v221_embedding_guarded_swaps.json"

MODEL_NAME = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
EMBED_DIM = 128
BATCH_SIZE = int(os.environ.get("V221_BATCH_SIZE", "32"))
BUDGETS = [500, 1000, 3000, 5000, 10000]


def assert_ids(left: pd.Series, right: pd.Series, label: str) -> None:
    if len(left) != len(right) or not left.astype(str).reset_index(drop=True).equals(
        right.astype(str).reset_index(drop=True)
    ):
        raise ValueError(f"{label} ID alignment mismatch")


def score_column(frame: pd.DataFrame) -> str:
    candidates = [column for column in frame.columns if column.endswith("_score")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one score column, found {candidates}")
    return candidates[0]


# %% [markdown]
# 7.1. V104 ile iki PU modelinin konsensüs uyuşmazlıkları


def prepare_disagreement() -> None:
    required = [TEACHER, V219_SCORE, V220_SCORE, PAIRS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V221 inputs:\n- " + "\n- ".join(missing))

    teacher = pd.read_csv(TEACHER, usecols=["id", "prediction"])
    if teacher["id"].duplicated().any() or teacher["prediction"].isna().any():
        raise ValueError("Teacher contains duplicate IDs or missing predictions")
    teacher["id"] = teacher["id"].astype(str)
    teacher["prediction"] = teacher["prediction"].astype(np.int8)
    teacher_ratio = float(teacher["prediction"].mean())

    v219 = pd.read_parquet(V219_SCORE)
    v220 = pd.read_parquet(V220_SCORE)
    v219_col = score_column(v219)
    v220_col = score_column(v220)
    assert_ids(teacher["id"], v219["id"], "teacher/V219")
    assert_ids(teacher["id"], v220["id"], "teacher/V220")

    p219 = v219[v219_col].astype(np.float32).to_numpy()
    p220 = v220[v220_col].astype(np.float32).to_numpy()
    threshold219 = float(np.quantile(p219, 1.0 - teacher_ratio))
    threshold220 = float(np.quantile(p220, 1.0 - teacher_ratio))
    pred219 = (p219 >= threshold219).astype(np.int8)
    pred220 = (p220 >= threshold220).astype(np.int8)
    teacher_pred = teacher["prediction"].to_numpy(np.int8)

    disagreement_mask = teacher_pred != pred220
    consensus_mask = pred219 == pred220
    selected = disagreement_mask & consensus_mask

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"], dtype=str)
    assert_ids(teacher["id"], pairs["id"], "teacher/pairs")
    out = pairs.loc[selected].copy()
    out["teacher"] = teacher_pred[selected]
    out["v219_pred"] = pred219[selected]
    out["v220_pred"] = pred220[selected]
    out["v219_score"] = p219[selected]
    out["v220_score"] = p220[selected]
    out["direction"] = np.where(out["teacher"].eq(0), "add", "remove")
    out.to_parquet(DISAGREEMENT, index=False, compression="zstd")

    summary = {
        "teacher_ratio": teacher_ratio,
        "threshold_v219_at_teacher_ratio": threshold219,
        "threshold_v220_at_teacher_ratio": threshold220,
        "all_disagreements": int(disagreement_mask.sum()),
        "student_consensus_disagreements": int(selected.sum()),
        "add_candidates": int(out["direction"].eq("add").sum()),
        "remove_candidates": int(out["direction"].eq("remove").sum()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    print("saved", DISAGREEMENT, flush=True)


def item_text(frame: pd.DataFrame) -> pd.Series:
    for column in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        frame[column] = frame[column].fillna("").astype(str)
    return (
        "urun: "
        + frame["title"]
        + ". kategori: "
        + frame["category"]
        + ". marka: "
        + frame["brand"]
        + ". cinsiyet: "
        + frame["gender"]
        + ". yas: "
        + frame["age_group"]
        + ". ozellikler: "
        + frame["attributes"].str.slice(0, 900)
    )


def save_embeddings(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float16, copy=False))
    temporary.replace(path)


# %% [markdown]
# 7.2. Yalnızca disagreement havuzunu embedding ile kodlama


def encode_disagreement() -> None:
    if not DISAGREEMENT.exists():
        raise FileNotFoundError(f"Run prepare first: {DISAGREEMENT}")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install the NLP dependencies: pip install -e \".[nlp]\"") from exc

    rows = pd.read_parquet(DISAGREEMENT, columns=["term_id", "item_id"])
    query_ids = pd.Series(rows["term_id"].astype(str).unique(), name="term_id")
    item_ids = pd.Series(rows["item_id"].astype(str).unique(), name="item_id")
    query_set = set(query_ids)
    item_set = set(item_ids)

    terms = pd.read_csv(TERMS, usecols=["term_id", "query"], dtype=str)
    terms = terms[terms["term_id"].isin(query_set)].copy()
    terms = terms.drop_duplicates("term_id").set_index("term_id").reindex(query_ids).reset_index()
    if terms["query"].isna().any():
        raise ValueError("Missing query text in V221 disagreement set")

    item_frames = []
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    for chunk in pd.read_csv(ITEMS, usecols=columns, dtype=str, chunksize=100000, low_memory=False):
        keep = chunk[chunk["item_id"].isin(item_set)]
        if not keep.empty:
            item_frames.append(keep)
    items = pd.concat(item_frames, ignore_index=True).drop_duplicates("item_id")
    items = items.set_index("item_id").reindex(item_ids).reset_index()
    if items["title"].isna().any():
        raise ValueError("Missing item metadata in V221 disagreement set")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": device, "queries": len(terms), "items": len(items), "batch": BATCH_SIZE}, flush=True)
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        truncate_dim=EMBED_DIM,
        device=device,
    )
    query_embeddings = model.encode(
        terms["query"].fillna("").tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    product_texts = item_text(items).tolist()
    item_embeddings = model.encode(
        product_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    pd.DataFrame({"term_id": query_ids}).to_parquet(QUERY_IDS, index=False)
    pd.DataFrame({"item_id": item_ids}).to_parquet(ITEM_IDS, index=False)
    save_embeddings(QUERY_EMB, query_embeddings)
    save_embeddings(ITEM_EMB, item_embeddings)
    print("saved embedding caches", flush=True)
    del model, query_embeddings, item_embeddings, items, terms, product_texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def percentile(values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.rank(method="average", pct=True, ascending=ascending).astype(np.float32)


# %% [markdown]
# 7.3. Embedding-guarded V221 adayları


def score_and_swap() -> None:
    required = [DISAGREEMENT, QUERY_IDS, ITEM_IDS, QUERY_EMB, ITEM_EMB, TEACHER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V221 caches:\n- " + "\n- ".join(missing))

    rows = pd.read_parquet(DISAGREEMENT)
    query_ids = pd.read_parquet(QUERY_IDS)["term_id"].astype(str)
    item_ids = pd.read_parquet(ITEM_IDS)["item_id"].astype(str)
    query_embeddings = np.load(QUERY_EMB).astype(np.float32)
    item_embeddings = np.load(ITEM_EMB).astype(np.float32)
    query_map = pd.Series(np.arange(len(query_ids), dtype=np.int32), index=query_ids)
    item_map = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)
    query_index = rows["term_id"].astype(str).map(query_map)
    item_index = rows["item_id"].astype(str).map(item_map)
    if query_index.isna().any() or item_index.isna().any():
        raise ValueError("Embedding ID mapping failed")
    semantic = np.sum(
        query_embeddings[query_index.to_numpy(np.int32)] * item_embeddings[item_index.to_numpy(np.int32)],
        axis=1,
    ).astype(np.float32)
    rows["trendyol_embedding"] = semantic

    add = rows[rows["direction"].eq("add")].copy()
    remove = rows[rows["direction"].eq("remove")].copy()
    add["priority"] = (
        0.55 * percentile(add["trendyol_embedding"])
        + 0.25 * percentile(add["v220_score"])
        + 0.20 * percentile(add["v219_score"])
    )
    remove["priority"] = (
        0.55 * percentile(remove["trendyol_embedding"], ascending=False)
        + 0.25 * percentile(remove["v220_score"], ascending=False)
        + 0.20 * percentile(remove["v219_score"], ascending=False)
    )
    # Require semantic evidence in the expected direction, not merely lexical consensus.
    add_semantic_cutoff = float(add["trendyol_embedding"].quantile(0.75))
    remove_semantic_cutoff = float(remove["trendyol_embedding"].quantile(0.25))
    add = add[add["trendyol_embedding"].ge(add_semantic_cutoff)].sort_values("priority", ascending=False)
    remove = remove[remove["trendyol_embedding"].le(remove_semantic_cutoff)].sort_values(
        "priority", ascending=False
    )
    rows.to_parquet(SCORED, index=False, compression="zstd")

    teacher = pd.read_csv(TEACHER, usecols=["id", "prediction"])
    teacher["id"] = teacher["id"].astype(str)
    teacher["prediction"] = teacher["prediction"].astype(np.int8)
    base_ratio = float(teacher["prediction"].mean())
    summaries = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for requested_budget in BUDGETS:
        budget = min(requested_budget, len(add), len(remove))
        if budget == 0:
            continue
        add_ids = set(add.head(budget)["id"].astype(str))
        remove_ids = set(remove.head(budget)["id"].astype(str))
        prediction = teacher["prediction"].to_numpy(copy=True)
        prediction[teacher["id"].isin(add_ids).to_numpy()] = 1
        prediction[teacher["id"].isin(remove_ids).to_numpy()] = 0
        path = OUT_DIR / f"FINAL_CANDIDATE_v221_embedding_guarded_swap_{budget}.csv"
        pd.DataFrame({"id": teacher["id"], "prediction": prediction}).to_csv(path, index=False)
        summaries.append(
            {
                "budget_each_direction": budget,
                "total_changes": int((prediction != teacher["prediction"].to_numpy()).sum()),
                "positive_ratio": float(prediction.mean()),
                "ratio_delta": float(prediction.mean() - base_ratio),
                "file": str(path),
            }
        )

    report = {
        "anchor": str(TEACHER),
        "anchor_positive_ratio": base_ratio,
        "model": MODEL_NAME,
        "embedding_dimension": EMBED_DIM,
        "disagreement_rows": len(rows),
        "eligible_add_rows": len(add),
        "eligible_remove_rows": len(remove),
        "add_semantic_cutoff_q75": add_semantic_cutoff,
        "remove_semantic_cutoff_q25": remove_semantic_cutoff,
        "outputs": summaries,
        "policy": "V104 anchor; V219/V220 consensus; Trendyol embedding guard; equal add/remove swaps",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


# %% [markdown]
# 8. V233 kesin çatışma korumalı genişleme


# %% [markdown]
# 8.1. Tasarım
# ----------
# - Anchor: FINAL_BEST_VERIFIED_0p832.csv
# - Yeni remove: yok
# - Kaynak: V221 V104/V219/V220 disagreement + Trendyol embedding
# - Genişleme: özgün add rank 1001..3000
# - Veto: model/grade/subject/bez no/formula stage/ölçü/cinsiyet/tek-çift
# - Marka veya genel alternatif tek başına veto değildir; yarı alakalı olabilir.


# %% [markdown]
# 8.2. Ayarlar


@dataclass(frozen=True)
class Config:
    # ``all`` komutu bu dosyayı V221 aşamasında kendisi üretir.
    anchor: Path = Path(
        "submissions/final_candidates_v221/FINAL_CANDIDATE_v221_embedding_guarded_swap_1000.csv"
    )
    v104: Path = Path("submissions/reference/FINAL_CANDIDATE_v104_full_balanced.csv")
    scored_disagreement: Path = Path("data/processed/v221_embedding_guarded_disagreement.parquet")
    terms: Path = Path("data/raw/terms.csv")
    items: Path = Path("data/raw/items.csv")
    output: Path = Path("submissions/FINAL_SINGLE_FILE_V233.csv")
    audit: Path = Path("reports/experiments/v233_guarded_asymmetric_add_audit.csv")
    report: Path = Path("reports/experiments/v233_guarded_asymmetric_add.json")
    verified_add_rank: int = int(os.environ.get("V233_VERIFIED_ADD_RANK", "1000"))
    expansion_add_rank: int = int(os.environ.get("V233_EXPANSION_ADD_RANK", "3000"))
    item_chunk: int = int(os.environ.get("V233_ITEM_CHUNK", "100000"))


CFG = Config()


# %% [markdown]
# 8.3. Metin normalizasyonu ve V221 önceliği


def normalize(value: object) -> str:
    text = str(value or "").lower().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def rebuild_v221_add_ranking(rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    add = rows[rows["direction"].eq("add")].copy()
    add["priority"] = (
        0.55 * percentile(add["trendyol_embedding"])
        + 0.25 * percentile(add["v220_score"])
        + 0.20 * percentile(add["v219_score"])
    )
    cutoff = float(add["trendyol_embedding"].quantile(0.75))
    add = add[add["trendyol_embedding"].ge(cutoff)].copy()
    add = add.sort_values(["priority", "id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    add["v221_add_rank"] = np.arange(1, len(add) + 1, dtype=np.int32)
    return add, cutoff


# %% [markdown]
# 8.4. Kesin uyuşmazlık çıkarıcıları
# ---------------------------------
# Kurallar yalnızca iki tarafta da açık bilgi bulunduğunda veto verir. Bilgi
# eksikliği veto değildir. Böylece marka alternatifi ve genel yarı-alaka korunur.


GRADE_RE = re.compile(r"(?<!\d)(1[0-2]|[1-9])\s*\.?\s*(?:sinif|sınıf)")
# ``normalize`` punctuationı boşluğa çevirdiği için 215/60 R17 aynı zamanda
# ``215 60 r17`` biçiminde yakalanır.
TIRE_RE = re.compile(r"(?<!\d)(\d{3})\s+(\d{2})\s*(?:r\s*)?(\d{2})(?!\d)")
JANT_RE = re.compile(r"(?<!\d)(1[0-9]|2[0-9])\s*(?:inc|inch|jant)(?!\w)")
PERSON_RE = re.compile(r"\b(tek|cift)\s+kisilik\b")
SUBJECTS = {
    "matematik", "turkce", "ingilizce", "fen", "fizik", "kimya", "biyoloji",
    "sosyal", "cografya", "tarih", "paragraf", "din", "inkilap",
}

DEVICE_PATTERNS = {
    "iphone": re.compile(r"\biphone\s*(\d{1,2})\s*(pro\s*max|promax|pro|plus|mini|e)?\b"),
    "ipad": re.compile(r"\bipad\s*(pro|air|mini)?\s*(\d{1,2}(?:\s*\.\s*\d)?)?\s*(?:nesil)?\b"),
    "samsung_tab": re.compile(r"\b(?:samsung\s+)?(?:galaxy\s+)?tab\s*([as]\s*\d{1,2})\s*(fe|plus|lite)?\b"),
    "samsung_phone": re.compile(r"\b(?:samsung\s+)?(?:galaxy\s+)?([asmz]\s*\d{1,3})\s*(fe|plus|ultra)?\b"),
    "redmi": re.compile(r"\b(?:xiaomi\s+)?redmi\s+(note\s+)?(\d{1,2})\s*(pro\s*plus|pro|plus|ultra|t)?\b"),
    "poco": re.compile(r"\bpoco\s*([xfc]\s*\d{1,2})\s*(pro|plus|ultra)?\b"),
    "oppo": re.compile(r"\boppo\s*([ar]\s*\d{1,3})\s*(pro|plus|5g|4g)?\b"),
    "vivo": re.compile(r"\bvivo\s*([yv]\s*\d{1,3})\s*(pro|plus|lite|5g|4g)?\b"),
    "honor_pad": re.compile(r"\bhonor\s+pad\s*([a-z]?\s*\d{1,2})\s*(pro|plus|lite)?\b"),
}


def canonical_match(match: re.Match) -> str:
    return " ".join(part.replace(" ", "") for part in match.groups() if part).strip()


def device_models(text: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for family, pattern in DEVICE_PATTERNS.items():
        values = {canonical_match(match) for match in pattern.finditer(text)}
        if values:
            out[family] = values
    return out


def explicit_device_conflict(query: str, item: str) -> str | None:
    q_models = device_models(query)
    i_models = device_models(item)
    for family in sorted(set(q_models) & set(i_models)):
        if q_models[family].isdisjoint(i_models[family]):
            return f"device_model_mismatch:{family}"
    return None


def extract_grades(text: str) -> set[int]:
    return {int(value) for value in GRADE_RE.findall(text)}


def extract_subjects(text: str) -> set[str]:
    tokens = set(text.split())
    return tokens & SUBJECTS


def diaper_numbers(text: str) -> set[int]:
    if not re.search(r"\b(?:bebek\s+bezi|kulot\s+bez|bez)\b", text):
        return set()
    values = set()
    for pattern in [
        r"\b(?:bebek\s+bezi|kulot\s+bez)\s*(?:no|numara)?\s*([1-8])(?!\d|\s*['’]?\s*li\b)",
        r"\b([1-8])\s*(?:numara|no|beden)\b",
        r"\b(?:numara|no|beden)\s*([1-8])\b",
    ]:
        values.update(int(v) for v in re.findall(pattern, text))
    return values


def formula_stages(text: str) -> set[int]:
    if not re.search(r"\b(?:aptamil|bebelac|hipp|devam\s+sutu|bebek\s+sutu)\b", text):
        return set()
    values = set()
    for pattern in [
        r"\b(?:aptamil|bebelac|hipp)\D{0,12}([1-4])(?!\d)",
        r"\b(?:no|numara)\s*([1-4])\b",
        r"\b([1-4])\s*(?:no|numara)\b",
    ]:
        values.update(int(v) for v in re.findall(pattern, text))
    return values


def genders(text: str) -> set[str]:
    out = set()
    if re.search(r"\b(?:kadin|bayan|kiz)\b", text):
        out.add("female")
    if re.search(r"\b(?:erkek|bay)\b", text):
        out.add("male")
    if "unisex" in text:
        out.add("unisex")
    return out


def disjoint_nonempty(left: set, right: set) -> bool:
    return bool(left and right and left.isdisjoint(right))


def hard_conflict_reasons(query_raw: object, title_raw: object, category_raw: object, attributes_raw: object) -> list[str]:
    query = normalize(query_raw)
    item = normalize(f"{title_raw} {category_raw} {attributes_raw}")
    reasons: list[str] = []

    model_reason = explicit_device_conflict(query, item)
    if model_reason:
        reasons.append(model_reason)

    q_grade, i_grade = extract_grades(query), extract_grades(item)
    if disjoint_nonempty(q_grade, i_grade):
        reasons.append("class_grade_mismatch")
    q_subject, i_subject = extract_subjects(query), extract_subjects(item)
    if q_grade and i_grade and not q_grade.isdisjoint(i_grade) and disjoint_nonempty(q_subject, i_subject):
        reasons.append("education_subject_mismatch")

    if disjoint_nonempty(diaper_numbers(query), diaper_numbers(item)):
        reasons.append("diaper_number_mismatch")
    if disjoint_nonempty(formula_stages(query), formula_stages(item)):
        reasons.append("formula_stage_mismatch")

    q_tire, i_tire = set(TIRE_RE.findall(query)), set(TIRE_RE.findall(item))
    if disjoint_nonempty(q_tire, i_tire):
        reasons.append("tire_size_mismatch")
    if ("jant" in query or "bisiklet" in query) and disjoint_nonempty(
        set(JANT_RE.findall(query)), set(JANT_RE.findall(item))
    ):
        reasons.append("wheel_size_mismatch")

    q_person, i_person = set(PERSON_RE.findall(query)), set(PERSON_RE.findall(item))
    if disjoint_nonempty(q_person, i_person):
        reasons.append("person_count_mismatch")

    q_gender, i_gender = genders(query), genders(item)
    q_explicit = q_gender - {"unisex"}
    i_explicit = i_gender - {"unisex"}
    if "unisex" not in i_gender and disjoint_nonempty(q_explicit, i_explicit):
        reasons.append("opposite_gender_mismatch")
    return sorted(set(reasons))


# %% [markdown]
# 8.5. Aday metadatası


def load_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    terms = pd.read_csv(CFG.terms, usecols=["term_id", "query"], dtype=str).drop_duplicates("term_id")
    needed_items = set(candidates["item_id"].astype(str))
    item_frames = []
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    for chunk in pd.read_csv(CFG.items, usecols=columns, dtype=str, chunksize=CFG.item_chunk, low_memory=False):
        hit = chunk[chunk["item_id"].isin(needed_items)]
        if not hit.empty:
            item_frames.append(hit)
    if not item_frames:
        raise ValueError("No V233 item metadata found")
    items = pd.concat(item_frames, ignore_index=True).drop_duplicates("item_id")
    out = candidates.merge(terms, on="term_id", how="left", validate="many_to_one")
    out = out.merge(items, on="item_id", how="left", validate="many_to_one")
    if out["query"].isna().any() or out["title"].isna().any():
        raise ValueError("Missing query or item metadata in V233 pool")
    return out


# %% [markdown]
# 8.6. Guard uygulanması ve submission


def build() -> dict:
    required = [CFG.anchor, CFG.v104, CFG.scored_disagreement, CFG.terms, CFG.items]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V233 inputs:\n- " + "\n- ".join(missing))
    if CFG.expansion_add_rank <= CFG.verified_add_rank:
        raise ValueError("V233_EXPANSION_ADD_RANK must exceed V233_VERIFIED_ADD_RANK")

    rows = pd.read_parquet(CFG.scored_disagreement)
    needed = {"id", "term_id", "item_id", "direction", "v219_score", "v220_score", "trendyol_embedding"}
    if not needed <= set(rows):
        raise ValueError(f"Scored disagreement missing {sorted(needed - set(rows))}")
    for column in ["id", "term_id", "item_id"]:
        rows[column] = rows[column].astype(str)
    ranked_add, semantic_cutoff = rebuild_v221_add_ranking(rows)
    tail = ranked_add[
        ranked_add["v221_add_rank"].gt(CFG.verified_add_rank)
        & ranked_add["v221_add_rank"].le(CFG.expansion_add_rank)
    ].copy()
    if tail.empty:
        raise ValueError("V233 expansion tail is empty")
    enriched = load_metadata(tail)
    enriched["guard_reasons"] = enriched.apply(
        lambda row: "|".join(hard_conflict_reasons(row["query"], row["title"], row["category"], row["attributes"])),
        axis=1,
    )
    enriched["guarded"] = enriched["guard_reasons"].ne("")
    eligible = enriched[~enriched["guarded"]].copy()

    anchor = pd.read_csv(CFG.anchor, dtype={"id": str})
    v104 = pd.read_csv(CFG.v104, dtype={"id": str})
    if not anchor["id"].equals(v104["id"]):
        raise ValueError("V221 anchor and V104 ID order mismatch")
    old = anchor["prediction"].to_numpy(np.int8)
    teacher = v104["prediction"].to_numpy(np.int8)
    add_ids = set(eligible["id"])
    add_mask = anchor["id"].isin(add_ids).to_numpy() & (old == 0)
    prediction = old.copy()
    prediction[add_mask] = 1
    if ((old == 1) & (prediction == 0)).any():
        raise RuntimeError("V233 invariant violated: a positive was removed")

    CFG.output.parent.mkdir(parents=True, exist_ok=True)
    CFG.audit.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": anchor["id"], "prediction": prediction}).to_csv(CFG.output, index=False)
    audit_columns = [
        "id", "term_id", "item_id", "v221_add_rank", "priority", "trendyol_embedding",
        "v219_score", "v220_score", "guarded", "guard_reasons", "query", "title",
        "category", "brand", "gender", "age_group", "attributes",
    ]
    enriched[audit_columns].to_csv(CFG.audit, index=False, encoding="utf-8-sig")

    guard_counts = (
        enriched.loc[enriched["guarded"], "guard_reasons"]
        .str.split("|").explode().value_counts().to_dict()
    )
    report = {
        "run": "v233_guarded_asymmetric_v221_expansion",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "anchor": str(CFG.anchor),
        "anchor_changes_vs_v104": int((old != teacher).sum()),
        "v221_semantic_cutoff_q75": semantic_cutoff,
        "ranked_add_rows": len(ranked_add),
        "expansion_tail_rows": len(tail),
        "guarded_rows": int(enriched["guarded"].sum()),
        "eligible_new_add_rows": int(add_mask.sum()),
        "guard_reason_counts": {str(k): int(v) for k, v in guard_counts.items()},
        "new_remove_rows": int(((old == 1) & (prediction == 0)).sum()),
        "changes_vs_v221": int((old != prediction).sum()),
        "changes_vs_v104": int((teacher != prediction).sum()),
        "anchor_positive_ratio": float(old.mean()),
        "output_positive_ratio": float(prediction.mean()),
        "positive_ratio_delta": float(prediction.mean() - old.mean()),
        "submission": str(CFG.output),
        "audit": str(CFG.audit),
        "policy": "V221 anchor; add ranks 1001..3000; hard conflicts vetoed; no new removals",
        "warning": "V221's +0.001 public improvement validates only its first 1000 equal-direction swaps; tail expansion remains a challenger.",
    }
    CFG.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# %% [markdown]
# 9. Tek dosyalık orkestratör
# ---------------------------
# Her komut yukarıdaki aynı dosyada tanımlı fonksiyonları çağırır. Başka script,
# ``src`` paketi veya hazır processed cache zorunlu değildir. ``all`` aşamaları
# doğru sırayla çalıştırır; mevcut cache'ler varsa zaman kazanmak için kullanılır.


PIPELINE_STAGES = [
    "audit",
    "prepare",
    "features",
    "train-baseline",
    "train-hard-negative",
    "embedding",
    "v221",
    "v233",
    "all",
]


def dependency_versions() -> dict[str, str]:
    packages = [
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "catboost",
        "torch", "sentence-transformers", "transformers",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def audit_single_file_inputs() -> dict:
    require_raw()
    if not TEACHER.exists():
        raise FileNotFoundError(
            "Missing V104 anchor:\n- " + str(TEACHER)
            + "\nThe pipeline intentionally treats this verified rulepack submission as input."
        )
    sample = pd.read_csv(RAW / "sample_submission.csv", usecols=["id"], dtype=str)
    pairs = pd.read_csv(RAW / "submission_pairs.csv", usecols=["id"], dtype=str)
    teacher = pd.read_csv(TEACHER, usecols=["id", "prediction"], dtype={"id": str})
    if not sample["id"].equals(pairs["id"]):
        raise ValueError("sample_submission/submission_pairs ID order mismatch")
    if not sample["id"].equals(teacher["id"]):
        raise ValueError("sample_submission/V104 anchor ID order mismatch")
    if teacher["id"].duplicated().any() or not teacher["prediction"].isin([0, 1]).all():
        raise ValueError("V104 anchor must contain unique IDs and binary predictions")
    versions = dependency_versions()
    required_packages = [
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "catboost",
        "torch", "sentence-transformers", "transformers",
    ]
    missing = [
        name for name in required_packages
        if versions[name] == "MISSING" or name in IMPORT_ERRORS
    ]
    if missing:
        raise RuntimeError(
            "Missing/unimportable Python packages: " + ", ".join(missing)
            + "\nInstall the supplied requirements.txt before running the pipeline."
        )
    import torch
    hardware = {
        "python": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary = {
        "single_python_file": True,
        "external_project_modules": False,
        "raw_inputs": [str(RAW / name) for name in [
            "items.csv", "terms.csv", "training_pairs.csv", "submission_pairs.csv", "sample_submission.csv"
        ]],
        "anchor": str(TEACHER),
        "submission_rows": len(sample),
        "anchor_positive_ratio": float(teacher["prediction"].mean()),
        "dependency_versions": versions,
        "hardware": hardware,
        "embedding_model": MODEL_NAME,
        "embedding_note": "Downloaded on first use unless already present in the Hugging Face cache.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_prepare_stage() -> None:
    items = prepare_items()
    prepare_pairs(items)
    del items
    release_accelerator_memory()


def run_feature_stage() -> None:
    items = prepare_items()
    pairs = prepare_pairs(items)
    build_features(items, pairs)
    del items, pairs
    release_accelerator_memory()


def require_feature_caches() -> None:
    missing = [str(path) for path in [TRAIN_FEATURES, TEST_FEATURES] if not path.exists()]
    if missing:
        raise FileNotFoundError("Run the features stage first:\n- " + "\n- ".join(missing))


def run_baseline_stage() -> None:
    require_feature_caches()
    previous = os.environ.get("V219_HARD_NEGATIVE")
    os.environ["V219_HARD_NEGATIVE"] = "0"
    try:
        train_and_submit()
    finally:
        if previous is None:
            os.environ.pop("V219_HARD_NEGATIVE", None)
        else:
            os.environ["V219_HARD_NEGATIVE"] = previous
    release_accelerator_memory()


def run_hard_negative_stage() -> None:
    require_feature_caches()
    if not OOF_SCORES.exists():
        raise FileNotFoundError("Run train-baseline first: " + str(OOF_SCORES))
    previous = os.environ.get("V219_HARD_NEGATIVE")
    os.environ["V219_HARD_NEGATIVE"] = "1"
    try:
        train_and_submit()
    finally:
        if previous is None:
            os.environ.pop("V219_HARD_NEGATIVE", None)
        else:
            os.environ["V219_HARD_NEGATIVE"] = previous
    release_accelerator_memory()


def run_embedding_stage() -> None:
    prepare_disagreement()
    encode_disagreement()
    release_accelerator_memory()


def validate_final_submission(path: Path) -> dict:
    sample = pd.read_csv(RAW / "sample_submission.csv", usecols=["id"], dtype=str)
    output = pd.read_csv(path, dtype={"id": str})
    if list(output.columns) != ["id", "prediction"]:
        raise ValueError(f"Unexpected submission columns: {output.columns.tolist()}")
    if not sample["id"].equals(output["id"]):
        raise ValueError("Final submission ID order mismatch")
    if output["id"].duplicated().any() or output["prediction"].isna().any():
        raise ValueError("Final submission contains duplicate IDs or null predictions")
    if not output["prediction"].isin([0, 1]).all():
        raise ValueError("Final submission predictions must be binary")
    summary = {
        "path": str(path),
        "rows": len(output),
        "positives": int(output["prediction"].sum()),
        "positive_ratio": float(output["prediction"].mean()),
        "duplicates": 0,
        "nulls": 0,
    }
    print(json.dumps({"final_submission": summary}, indent=2, ensure_ascii=False), flush=True)
    return summary


def run_stage(stage: str) -> None:
    if stage == "audit":
        audit_single_file_inputs()
    elif stage == "prepare":
        audit_single_file_inputs()
        run_prepare_stage()
    elif stage == "features":
        audit_single_file_inputs()
        run_feature_stage()
    elif stage == "train-baseline":
        audit_single_file_inputs()
        run_baseline_stage()
    elif stage == "train-hard-negative":
        audit_single_file_inputs()
        run_hard_negative_stage()
    elif stage == "embedding":
        audit_single_file_inputs()
        run_embedding_stage()
    elif stage == "v221":
        audit_single_file_inputs()
        score_and_swap()
    elif stage == "v233":
        audit_single_file_inputs()
        build()
        validate_final_submission(CFG.output)
    elif stage == "all":
        audit_single_file_inputs()
        run_prepare_stage()
        run_feature_stage()
        run_baseline_stage()
        run_hard_negative_stage()
        run_embedding_stage()
        score_and_swap()
        build()
        validate_final_submission(CFG.output)
    else:
        raise KeyError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw CSV + V104 anchor -> V219 -> V220 -> V221 -> guarded V233"
    )
    parser.add_argument("stage", choices=PIPELINE_STAGES)
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
