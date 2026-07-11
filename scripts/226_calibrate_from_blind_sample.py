"""V226 — Kör örneklemden bağımsız olasılık ve cardinality kalibrasyonu.

Normal bir ``.py`` dosyasıdır; ``# %%`` başlıkları gerektiğinde hücre olarak
çalıştırılmasını sağlar. Dosya yukarıdan aşağı veri seçimi, kör etiketleme,
kalibrasyon, doğrulama ve submission akışını açıklar.

Akış:
    python scripts/226_calibrate_from_blind_sample.py build-audit
    # reports/manual_review/v226_blind_calibration.csv içindeki human_label doldurulur
    python scripts/226_calibrate_from_blind_sample.py calibrate
"""

# %% [markdown]
# 0. Yöntem
# ---------
# Test satırları consensus-score dilimi ve sorgu aday sayısı katmanlarına ayrılır.
# Her katmandan örnek alınır; inclusion_probability anahtar dosyasında saklanır.
# Etiketleme dosyasında model skoru, eski prediction ve yön bilgisi bulunmaz.
# Isotonic regression inverse-probability ağırlıklarıyla gerçek test dağılımına
# kalibre edilir. Son olarak her sorgu için sum(P(relevant)) kadar pozitif seçilir.

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


# %% [markdown]
# 1. Ayarlar ve dosya sözleşmesi


@dataclass(frozen=True)
class Config:
    score_cache: Path = Path("data/processed/v225_calibrated_test_scores.parquet")
    pairs: Path = Path("data/raw/submission_pairs.csv")
    terms: Path = Path("data/raw/terms.csv")
    items: Path = Path("data/raw/items.csv")
    sample: Path = Path("data/raw/sample_submission.csv")
    blind: Path = Path("reports/manual_review/v226_blind_calibration.csv")
    key: Path = Path("data/processed/v226_blind_calibration_key.parquet")
    calibrated_scores: Path = Path("data/processed/v226_calibrated_scores.parquet")
    submission: Path = Path("submissions/final_candidates_v226/FINAL_CANDIDATE_v226_blind_calibrated.csv")
    report: Path = Path("reports/experiments/v226_blind_calibration.json")
    seed: int = int(os.environ.get("V226_SEED", "20260711"))
    target_rows: int = int(os.environ.get("V226_AUDIT_ROWS", "1000"))
    score_bins: int = int(os.environ.get("V226_SCORE_BINS", "10"))
    cv_folds: int = int(os.environ.get("V226_CV_FOLDS", "5"))
    bootstrap_rounds: int = int(os.environ.get("V226_BOOTSTRAP", "1000"))
    item_chunk: int = int(os.environ.get("V226_ITEM_CHUNK", "100000"))


CFG = Config()
EPS = 1e-6


# %% [markdown]
# 2. Girdi denetimi
# -----------------
# V225 score cache yalnızca skor kaynağıdır; V225'in prior adayları kullanılmaz.


def audit_inputs(require_labels: bool = False) -> None:
    required = [CFG.score_cache, CFG.pairs, CFG.terms, CFG.items, CFG.sample]
    if require_labels:
        required += [CFG.blind, CFG.key]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V226 inputs:\n- " + "\n- ".join(missing))
    print(json.dumps({"required_found": [str(p) for p in required]}, indent=2, ensure_ascii=False), flush=True)


def load_population() -> pd.DataFrame:
    score = pd.read_parquet(CFG.score_cache)
    needed = {"id", "term_id", "consensus_logit"}
    if not needed <= set(score.columns):
        raise ValueError(f"Score cache must contain {sorted(needed)}; got {score.columns.tolist()}")
    score = score[["id", "term_id", "consensus_logit"]].copy()
    score["id"] = score["id"].astype(str)
    score["term_id"] = score["term_id"].astype(str)
    sample = pd.read_csv(CFG.sample, usecols=["id"], dtype=str)
    if not score["id"].reset_index(drop=True).equals(sample["id"]):
        raise ValueError("Score cache and sample_submission ID order mismatch")
    if score["id"].duplicated().any() or not np.isfinite(score["consensus_logit"]).all():
        raise ValueError("Invalid ID or consensus score")
    return score


# %% [markdown]
# 3. Temsili tabakalı kör örneklem
# ---------------------------------
# score_decile model güvenini, query_size_tier sorgunun aday havuzu büyüklüğünü
# temsil eder. Katman başına yaklaşık eşit örnek, uç skorların kaybolmasını önler.
# inclusion_probability daha sonra bu eşit olmayan seçimi nüfusa geri ağırlıklar.


def add_strata(population: pd.DataFrame) -> pd.DataFrame:
    frame = population.copy()
    frame["score_bin"] = pd.qcut(
        frame["consensus_logit"], q=CFG.score_bins, labels=False, duplicates="drop"
    ).astype(np.int8)
    query_size = frame.groupby("term_id", sort=False)["id"].transform("size")
    frame["query_size"] = query_size.astype(np.int32)
    frame["query_size_tier"] = pd.cut(
        query_size,
        bins=[0, 100, 110, 200, 500, np.inf],
        labels=["le100", "101_110", "111_200", "201_500", "gt500"],
        include_lowest=True,
    ).astype(str)
    frame["stratum"] = frame["score_bin"].astype(str) + "__" + frame["query_size_tier"]
    return frame


def stratified_sample(population: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(CFG.seed)
    sizes = population["stratum"].value_counts().sort_index()
    active = sizes.index.tolist()
    base_take = max(1, CFG.target_rows // len(active))
    allocation = {s: min(base_take, int(sizes[s])) for s in active}
    remaining = CFG.target_rows - sum(allocation.values())
    # Kalan bütçeyi kapasitesi olan büyük katmanlara deterministik dağıt. Böylece
    # her satırın inclusion_probability değeri tam olarak take / population olur.
    expandable = sorted(active, key=lambda s: (-int(sizes[s]), s))
    while remaining > 0:
        progressed = False
        for stratum in expandable:
            if allocation[stratum] < int(sizes[stratum]):
                allocation[stratum] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    chosen_parts = []
    for stratum in active:
        pool = population[population["stratum"].eq(stratum)]
        take = allocation[stratum]
        index = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        selected = pool.loc[index].copy()
        selected["stratum_population"] = len(pool)
        selected["stratum_sample"] = take
        selected["inclusion_probability"] = take / len(pool)
        chosen_parts.append(selected)
    selected = pd.concat(chosen_parts, ignore_index=True)
    return selected.drop_duplicates("id").reset_index(drop=True)


def load_item_metadata(item_ids: set[str]) -> pd.DataFrame:
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    parts = []
    for chunk in pd.read_csv(CFG.items, usecols=columns, dtype=str, chunksize=CFG.item_chunk, low_memory=False):
        keep = chunk[chunk["item_id"].isin(item_ids)]
        if not keep.empty:
            parts.append(keep)
    if not parts:
        raise RuntimeError("No item metadata found")
    return pd.concat(parts, ignore_index=True).drop_duplicates("item_id")


def build_audit() -> None:
    audit_inputs()
    population = add_strata(load_population())
    selected = stratified_sample(population)
    pairs = pd.read_csv(CFG.pairs, usecols=["id", "term_id", "item_id"], dtype=str)
    selected = selected.merge(pairs, on=["id", "term_id"], how="left", validate="one_to_one")
    if selected["item_id"].isna().any():
        raise ValueError("Selected IDs could not be mapped to item_id")
    selected["audit_id"] = [f"V226_{i:04d}" for i in range(1, len(selected) + 1)]

    key_columns = [
        "audit_id", "id", "term_id", "item_id", "consensus_logit", "score_bin",
        "query_size", "query_size_tier", "stratum", "stratum_population",
        "stratum_sample", "inclusion_probability",
    ]
    CFG.key.parent.mkdir(parents=True, exist_ok=True)
    selected[key_columns].to_parquet(CFG.key, index=False, compression="zstd")

    terms = pd.read_csv(CFG.terms, usecols=["term_id", "query"], dtype=str).drop_duplicates("term_id")
    items = load_item_metadata(set(selected["item_id"]))
    blind = selected[["audit_id", "term_id", "item_id"]].merge(terms, on="term_id", validate="many_to_one")
    blind = blind.merge(items, on="item_id", validate="many_to_one")
    blind = blind[["audit_id", "query", "title", "category", "brand", "gender", "age_group", "attributes"]]
    blind["attributes"] = blind["attributes"].fillna("").str.slice(0, 700)
    blind["human_label"] = ""
    blind["notes"] = ""
    blind = blind.sample(frac=1.0, random_state=CFG.seed).reset_index(drop=True)
    CFG.blind.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(CFG.blind, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "audit_rows": len(blind),
        "strata": int(selected["stratum"].nunique()),
        "blind_file": str(CFG.blind),
        "key_file": str(CFG.key),
        "label_definition": {"1": "relevant or partially relevant", "0": "irrelevant", "?": "uncertain"},
    }, indent=2, ensure_ascii=False), flush=True)


# %% [markdown]
# 4. Etiket okuma ve ağırlıklı doğrulama
# --------------------------------------
# Sadece 0/1 etiketleri kalibrasyona girer. ``?`` belirsiz kabul edilir. Ağırlık
# 1/inclusion_probability olup seçilmiş örneklemi test nüfusuna geri taşır.


def normalize_label(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "1.0", "alakalı", "alakali", "yarı alakalı", "yari alakali", "relevant"}:
        return 1.0
    if text in {"0", "0.0", "alakasız", "alakasiz", "irrelevant"}:
        return 0.0
    if text in {"", "?", "uncertain", "emin değilim", "emin degilim"}:
        return np.nan
    raise ValueError(f"Unknown human_label: {value!r}")


def load_labels() -> pd.DataFrame:
    blind = pd.read_csv(CFG.blind, dtype=str)
    key = pd.read_parquet(CFG.key)
    if blind["audit_id"].duplicated().any():
        raise ValueError("Duplicate audit_id")
    blind["label"] = blind["human_label"].map(normalize_label)
    labeled = key.merge(blind[["audit_id", "label"]], on="audit_id", validate="one_to_one")
    labeled = labeled[labeled["label"].notna()].copy()
    if len(labeled) < 100 or labeled["label"].nunique() != 2:
        raise ValueError("Need at least 100 labeled rows containing both 0 and 1")
    labeled["label"] = labeled["label"].astype(np.int8)
    labeled["weight"] = (1.0 / labeled["inclusion_probability"].clip(lower=EPS)).astype(np.float64)
    labeled["weight"] /= labeled["weight"].mean()
    return labeled


def cross_validated_calibration(labeled: pd.DataFrame) -> dict:
    folds = min(CFG.cv_folds, labeled["term_id"].nunique())
    splitter = GroupKFold(n_splits=folds)
    prediction = np.full(len(labeled), np.nan, dtype=np.float64)
    x = labeled["consensus_logit"].to_numpy(np.float64)
    y = labeled["label"].to_numpy(np.int8)
    w = labeled["weight"].to_numpy(np.float64)
    groups = labeled["term_id"].astype(str).to_numpy()
    for train_index, valid_index in splitter.split(x, y, groups):
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(x[train_index], y[train_index], sample_weight=w[train_index])
        prediction[valid_index] = model.predict(x[valid_index])
    clipped = np.clip(prediction, EPS, 1.0 - EPS)
    return {
        "folds": folds,
        "weighted_brier": float(np.average((prediction - y) ** 2, weights=w)),
        "weighted_logloss": float(log_loss(y, clipped, sample_weight=w, labels=[0, 1])),
        "weighted_auc": float(roc_auc_score(y, prediction, sample_weight=w)),
    }


def weighted_prior_interval(labeled: pd.DataFrame) -> dict:
    rng = np.random.default_rng(CFG.seed + 1)
    y = labeled["label"].to_numpy(np.float64)
    w = labeled["weight"].to_numpy(np.float64)
    estimate = float(np.average(y, weights=w))
    values = np.empty(CFG.bootstrap_rounds, dtype=np.float64)
    for i in range(CFG.bootstrap_rounds):
        index = rng.integers(0, len(labeled), size=len(labeled))
        values[i] = np.average(y[index], weights=w[index])
    return {
        "estimate": estimate,
        "bootstrap_low_95": float(np.quantile(values, 0.025)),
        "bootstrap_high_95": float(np.quantile(values, 0.975)),
    }


# %% [markdown]
# 5. Tüm testin kalibrasyonu ve bağımsız cardinality


def decode(population: pd.DataFrame, probability: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    work = population[["id", "term_id"]].copy()
    work["probability"] = probability.astype(np.float32)
    stats = work.groupby("term_id", sort=False)["probability"].agg(["size", "sum", "mean", "max"])
    stats = stats.rename(columns={"size": "candidate_count", "sum": "expected_positive"})
    stats["positive_count"] = np.floor(stats["expected_positive"] + 0.5).astype(int)
    stats["positive_count"] = stats[["positive_count", "candidate_count"]].min(axis=1).astype(int)
    work["k"] = work["term_id"].map(stats["positive_count"])
    ordered = work.sort_values(["term_id", "probability", "id"], ascending=[True, False, True], kind="stable")
    ordered["position"] = ordered.groupby("term_id", sort=False).cumcount()
    ordered["prediction"] = (ordered["position"] < ordered["k"]).astype(np.int8)
    prediction = ordered.set_index("id")["prediction"].reindex(work["id"]).to_numpy(np.int8)
    return prediction, stats.reset_index()


def calibrate() -> None:
    audit_inputs(require_labels=True)
    labeled = load_labels()
    validation = cross_validated_calibration(labeled)
    prior = weighted_prior_interval(labeled)
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(labeled["consensus_logit"], labeled["label"], sample_weight=labeled["weight"])

    population = load_population()
    probability = model.predict(population["consensus_logit"].to_numpy(np.float64)).astype(np.float32)
    prediction, query_stats = decode(population, probability)
    CFG.calibrated_scores.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": population["id"], "term_id": population["term_id"],
        "consensus_logit": population["consensus_logit"], "calibrated_probability": probability,
    }).to_parquet(CFG.calibrated_scores, index=False, compression="zstd")
    CFG.submission.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": population["id"], "prediction": prediction}).to_csv(CFG.submission, index=False)

    count_desc = query_stats["positive_count"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    report = {
        "run": "v226_blind_calibrated_adaptive_cardinality",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "labeled_rows": len(labeled),
        "weighted_positive_prior": prior,
        "cross_validated_calibration": validation,
        "output_positive_ratio": float(prediction.mean()),
        "positive_rows": int(prediction.sum()),
        "zero_positive_queries": int(query_stats["positive_count"].eq(0).sum()),
        "query_positive_count": {k: float(v) for k, v in count_desc.to_dict().items()},
        "submission": str(CFG.submission),
        "method": "stratified blind labels + inverse-probability weighted isotonic calibration + per-query expected count",
        "warning": "Human labels and finite audit size remain sources of uncertainty; use the bootstrap interval.",
    }
    CFG.report.parent.mkdir(parents=True, exist_ok=True)
    CFG.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


# %% [markdown]
# 6. Komut satırı


def main() -> None:
    parser = argparse.ArgumentParser(description="V226 blind calibration pipeline")
    parser.add_argument("stage", choices=["audit", "build-audit", "calibrate"])
    args = parser.parse_args()
    if args.stage == "audit":
        audit_inputs(require_labels=False)
    elif args.stage == "build-audit":
        build_audit()
    else:
        calibrate()


if __name__ == "__main__":
    main()
