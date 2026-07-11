"""V225 — Açıklanabilir, sorguya göre değişen pozitif sayısı üretimi.

Bu dosya normal bir Python scriptidir. ``# %%`` başlıkları VS Code ve Spyder'da
istenirse hücre gibi çalıştırılmasını sağlar. Dosya aynı zamanda yukarıdan aşağı
okunduğunda veri -> kalibrasyon -> sorgu cardinality -> submission akışını anlatır.

V104 kullanılmaz. Her sorgudaki pozitif sayısı, kalibre edilmiş satır
olasılıklarının toplamından hesaplanır.

Kullanım:
    python scripts/225_build_adaptive_cardinality.py audit
    python scripts/225_build_adaptive_cardinality.py build
    python scripts/225_build_adaptive_cardinality.py all
"""

# %% [markdown]
# 0. Tasarım özeti
# ----------------
# 1) V219 ve varsa V220 olasılıklarını okur.
# 2) Modelleri logit uzayında ortalayarak consensus skor üretir.
# 3) Global pozitif prior'ına ulaşan tek bir intercept öğrenir.
# 4) Her term_id için beklenen pozitif sayısını sum(probability) ile hesaplar.
# 5) En yüksek olasılıklı round(expected_count) ürünü pozitif seçer.
# Böylece pozitif sayısı V104'e veya sabit add/remove bütçesine bağlı değildir.

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# %% [markdown]
# 1. Deney ayarları
# -----------------
# Bütün yollar ve değiştirilebilir parametreler tek yerde tutulur. Ortam
# değişkenleriyle deney yapılabilir; kaynak kodu değiştirmek gerekmez.


@dataclass(frozen=True)
class Config:
    root: Path = Path(".")
    v219_scores: Path = Path("data/processed/v219_test_scores.parquet")
    v220_scores: Path = Path("data/processed/v220_crossfit_hard_negative_test_scores.parquet")
    pairs: Path = Path("data/raw/submission_pairs.csv")
    sample: Path = Path("data/raw/sample_submission.csv")
    v219_report: Path = Path("reports/experiments/v219_raw_pu_lexical.json")
    output_dir: Path = Path("submissions/final_candidates_v225")
    report: Path = Path("reports/experiments/v225_adaptive_cardinality.json")
    score_cache: Path = Path("data/processed/v225_calibrated_test_scores.parquet")
    chunk_size: int = int(os.environ.get("V225_CHUNK", "500000"))
    # Prior kesin gerçek değildir; üç komşu değer ayrı aday olarak üretilir.
    prior_delta: float = float(os.environ.get("V225_PRIOR_DELTA", "0.02"))
    # Query expected count'u round etmeden önce uygulanan minimum güven.
    min_expected_count: float = float(os.environ.get("V225_MIN_EXPECTED", "0.35"))


CFG = Config()
EPS = np.float32(1e-6)


# %% [markdown]
# 2. Küçük matematik yardımcıları
# --------------------------------
# Olasılıkları logit uzayında birleştirmek, 0.9 ile 0.6 arasındaki farkı lineer
# ortalamadan daha doğru temsil eder. Intercept ise sıralamayı bozmadan yalnızca
# global pozitif oranını kalibre eder.


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def fit_intercept(raw_logit: np.ndarray, target_prior: float) -> float:
    """Binary search ile mean(sigmoid(raw_logit + b)) == target_prior çözer."""
    if not 0.0 < target_prior < 1.0:
        raise ValueError(f"target_prior must be in (0, 1), got {target_prior}")
    low, high = -30.0, 30.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if float(sigmoid(raw_logit + middle).mean()) < target_prior:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


# %% [markdown]
# 3. Girdi denetimi ve şema keşfi
# --------------------------------
# Script sabit bir skor sütunu adına güvenmez. ID dışındaki sayısal skor sütununu
# bulur, ID sırasını sample_submission ile doğrular ve sessiz merge hatalarını önler.


def audit_inputs() -> dict:
    required = [CFG.v219_scores, CFG.pairs, CFG.sample]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V225 inputs:\n- " + "\n- ".join(missing))
    summary = {
        "required_found": [str(path) for path in required],
        "v220_available": CFG.v220_scores.exists(),
        "v219_report_available": CFG.v219_report.exists(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def score_column(frame: pd.DataFrame) -> str:
    preferred = ["v219_score", "v220_score", "score", "prediction", "probability"]
    for name in preferred:
        if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name]):
            return name
    numeric = [c for c in frame.columns if c != "id" and pd.api.types.is_numeric_dtype(frame[c])]
    if len(numeric) != 1:
        raise ValueError(f"Cannot identify one score column: {frame.columns.tolist()}")
    return numeric[0]


def read_score(path: Path, expected_ids: pd.Series) -> tuple[np.ndarray, str]:
    frame = pd.read_parquet(path)
    column = score_column(frame)
    if "id" in frame:
        ids = frame["id"].astype(str).reset_index(drop=True)
        if not ids.equals(expected_ids):
            raise ValueError(f"ID order mismatch in {path}")
    elif len(frame) != len(expected_ids):
        raise ValueError(f"Row count mismatch in {path}: {len(frame)}")
    values = frame[column].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite score in {path}")
    return values, column


# %% [markdown]
# 4. Model konsensüsü
# -------------------
# V219 ana sınıflandırıcıdır. V220 mevcutsa iki model logit uzayında eşit ağırlıkla
# birleştirilir. V224 ranker leaderboard'da 0.830 aldığı için bilerek kullanılmaz.


def load_consensus() -> tuple[pd.DataFrame, np.ndarray, dict]:
    pairs = pd.read_csv(CFG.pairs, usecols=["id", "term_id"], dtype=str)
    sample = pd.read_csv(CFG.sample, usecols=["id"], dtype=str)
    if not pairs["id"].equals(sample["id"]):
        raise ValueError("submission_pairs and sample_submission ID order mismatch")

    v219, col219 = read_score(CFG.v219_scores, sample["id"])
    logits = [logit(v219)]
    sources = [{"path": str(CFG.v219_scores), "column": col219, "weight": 1.0}]
    if CFG.v220_scores.exists():
        v220, col220 = read_score(CFG.v220_scores, sample["id"])
        logits.append(logit(v220))
        sources.append({"path": str(CFG.v220_scores), "column": col220, "weight": 1.0})
    consensus_logit = np.mean(np.vstack(logits), axis=0)
    return pairs, consensus_logit, {"sources": sources}


# %% [markdown]
# 5. Global PU prior
# ------------------
# Öncelik ortam değişkenindedir. Yoksa V219 raporundaki estimated_positive_prior
# kullanılır. Bu değer de sentetik-negatif varsayımına bağlı olduğu için merkez ve
# ±delta olmak üzere üç aday üretilir.


def load_central_prior() -> float:
    explicit = os.environ.get("V225_PRIOR")
    if explicit is not None:
        return float(explicit)
    if CFG.v219_report.exists():
        report = json.loads(CFG.v219_report.read_text(encoding="utf8"))
        calibration = report.get("calibration", {})
        if "estimated_positive_prior" in calibration:
            return float(calibration["estimated_positive_prior"])
    raise ValueError("Set V225_PRIOR or provide v219 report with estimated_positive_prior")


# %% [markdown]
# 6. Sorguya özgü pozitif sayısı
# --------------------------------
# Her query için k = round(sum(calibrated_probability)) hesaplanır. Bu, modelin o
# sorguda beklediği pozitif ürün adedidir. V104 sayıları okunmaz ve add/remove
# eşitliği zorlanmaz.


def query_adaptive_decode(frame: pd.DataFrame, probability: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    work = frame.copy()
    work["probability"] = probability.astype(np.float32)
    stats = work.groupby("term_id", sort=False)["probability"].agg(["size", "sum", "mean", "max"]).rename(
        columns={"size": "candidate_count", "sum": "expected_positive", "mean": "mean_probability", "max": "max_probability"}
    )
    stats["positive_count"] = np.floor(stats["expected_positive"] + 0.5).astype(int)
    stats.loc[stats["expected_positive"] < CFG.min_expected_count, "positive_count"] = 0
    stats["positive_count"] = stats[["positive_count", "candidate_count"]].min(axis=1).astype(int)

    work["k"] = work["term_id"].map(stats["positive_count"])
    ordered = work.sort_values(["term_id", "probability", "id"], ascending=[True, False, True], kind="stable")
    ordered["position"] = ordered.groupby("term_id", sort=False).cumcount()
    ordered["prediction"] = (ordered["position"] < ordered["k"]).astype(np.int8)
    prediction = ordered.set_index("id")["prediction"].reindex(work["id"]).to_numpy(np.int8)
    return prediction, stats.reset_index()


# %% [markdown]
# 7. Submission üretimi ve doğrulama
# -----------------------------------
# Her prior ayrı dosyadır. Satır sırası, null, duplicate, binary değer ve sorgu
# cardinality dağılımı rapora yazılır.


def build_candidates() -> dict:
    audit_inputs()
    pairs, consensus_logit, consensus_info = load_consensus()
    central = load_central_prior()
    priors = sorted({max(0.01, central - CFG.prior_delta), central, min(0.99, central + CFG.prior_delta)})
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    CFG.report.parent.mkdir(parents=True, exist_ok=True)
    CFG.score_cache.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    cache = pairs.copy()
    cache["consensus_logit"] = consensus_logit.astype(np.float32)
    for prior in priors:
        intercept = fit_intercept(consensus_logit, prior)
        probability = sigmoid(consensus_logit + intercept).astype(np.float32)
        prediction, query_stats = query_adaptive_decode(pairs, probability)
        tag = f"{prior:.3f}".replace(".", "p")
        path = CFG.output_dir / f"FINAL_CANDIDATE_v225_adaptive_prior_{tag}.csv"
        pd.DataFrame({"id": pairs["id"], "prediction": prediction}).to_csv(path, index=False)
        cache[f"probability_prior_{tag}"] = probability
        count_description = query_stats["positive_count"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        candidates.append({
            "target_prior": prior,
            "intercept": intercept,
            "output_positive_ratio": float(prediction.mean()),
            "positive_rows": int(prediction.sum()),
            "zero_positive_queries": int(query_stats["positive_count"].eq(0).sum()),
            "query_positive_count": {k: float(v) for k, v in count_description.to_dict().items()},
            "path": str(path),
        })
    cache.to_parquet(CFG.score_cache, index=False, compression="zstd")
    report = {
        "run": "v225_adaptive_cardinality",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "method": "logit consensus -> global PU intercept -> per-query expected positive count -> top-k",
        "independent_of_v104_counts": True,
        "consensus": consensus_info,
        "central_prior": central,
        "candidates": candidates,
        "warning": "The prior and probabilities rely on PU/synthetic-negative assumptions; validate with a fresh blind audit.",
    }
    CFG.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


# %% [markdown]
# 8. Komut satırı girişi
# ----------------------
# Fonksiyonlar notebook hücrelerinden tek tek çağrılabilir. CLI kullanıldığında
# audit yalnızca girdileri kontrol eder; build/all adayları üretir.


def main() -> None:
    parser = argparse.ArgumentParser(description="V225 adaptive-cardinality pipeline")
    parser.add_argument("stage", choices=["audit", "build", "all"])
    args = parser.parse_args()
    if args.stage == "audit":
        audit_inputs()
    else:
        build_candidates()


if __name__ == "__main__":
    main()
