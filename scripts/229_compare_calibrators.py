"""V229 — IPW ve olasılık kalibratörlerini karşılaştır.

Bu script submission üretmez. Amaç, V226 isotonic basamak probleminden sonra
Platt, Beta ve bagged-isotonic yöntemlerini aynı group-CV protokolünde ölçmek ve
tam test için pürüzsüz olasılık cache'leri üretmektir.

    python scripts/229_compare_calibrators.py compare

Dosya ``# %%`` bölümleriyle yukarıdan aşağı okunabilir ve gerektiğinde hücre gibi
çalıştırılabilir.
"""

# %% [markdown]
# 0. Deney protokolü
# ------------------
# - Etiket kaynağı: V226 kör audit
# - Split: term_id bazlı GroupKFold
# - Ağırlık: raw IPW, p95-trim IPW, p99-trim IPW
# - Kalibratör: Platt, Beta, bagged Isotonic
# - Seçim: weighted logloss, ardından Brier ve ECE
# - AUC yalnızca ranking sinyalini raporlar; kalibratör seçiminde ana ölçüt değildir.

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


# %% [markdown]
# 1. Ayarlar


@dataclass(frozen=True)
class Config:
    blind: Path = Path("reports/manual_review/v226_blind_calibration.csv")
    key: Path = Path("data/processed/v226_blind_calibration_key.parquet")
    population: Path = Path("data/processed/v225_calibrated_test_scores.parquet")
    output_dir: Path = Path("data/processed")
    report: Path = Path("reports/experiments/v229_calibration_comparison.json")
    fold_report: Path = Path("reports/experiments/v229_calibration_folds.csv")
    seed: int = int(os.environ.get("V229_SEED", "20260711"))
    folds: int = int(os.environ.get("V229_FOLDS", "5"))
    isotonic_bags: int = int(os.environ.get("V229_ISOTONIC_BAGS", "15"))
    ece_bins: int = int(os.environ.get("V229_ECE_BINS", "15"))


CFG = Config()
EPS = 1e-6


# %% [markdown]
# 2. Veri yükleme ve etiket sözleşmesi


def normalize_label(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "1.0", "alakalı", "alakali", "yarı alakalı", "yari alakali", "relevant"}:
        return 1.0
    if text in {"0", "0.0", "alakasız", "alakasiz", "irrelevant"}:
        return 0.0
    if text in {"", "?", "uncertain"}:
        return np.nan
    raise ValueError(f"Unknown label: {value!r}")


def require_inputs() -> None:
    missing = [str(p) for p in [CFG.blind, CFG.key, CFG.population] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing V229 inputs:\n- " + "\n- ".join(missing))


def load_labeled() -> pd.DataFrame:
    blind = pd.read_csv(CFG.blind, dtype=str)
    key = pd.read_parquet(CFG.key)
    blind["label"] = blind["human_label"].map(normalize_label)
    frame = key.merge(blind[["audit_id", "label"]], on="audit_id", validate="one_to_one")
    frame = frame[frame["label"].notna()].copy()
    frame["label"] = frame["label"].astype(np.int8)
    frame["raw_ipw"] = 1.0 / frame["inclusion_probability"].clip(lower=EPS)
    if len(frame) < 100 or frame["label"].nunique() != 2:
        raise ValueError("Need at least 100 labels containing both classes")
    return frame


def load_population() -> pd.DataFrame:
    frame = pd.read_parquet(CFG.population)
    needed = {"id", "term_id", "consensus_logit"}
    if not needed <= set(frame.columns):
        raise ValueError(f"Population cache missing {sorted(needed - set(frame.columns))}")
    return frame[["id", "term_id", "consensus_logit"]].copy()


# %% [markdown]
# 3. IPW tanıları ve trimming
# ---------------------------
# Normalizasyon model sonucunu değiştirmez ancak ağırlık ölçeğini okunabilir yapar.
# Trimming, az örneklenen tabakaların tek başına kalibrasyonu sürüklemesini önler.


def normalized_weight(raw: pd.Series, quantile: float | None) -> np.ndarray:
    weight = raw.to_numpy(np.float64).copy()
    if quantile is not None:
        cap = float(np.quantile(weight, quantile))
        weight = np.minimum(weight, cap)
    return weight / weight.mean()


def weight_schemes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "raw_ipw": normalized_weight(frame["raw_ipw"], None),
        "trim_p95": normalized_weight(frame["raw_ipw"], 0.95),
        "trim_p99": normalized_weight(frame["raw_ipw"], 0.99),
    }


def effective_sample_size(weight: np.ndarray) -> float:
    return float(weight.sum() ** 2 / np.square(weight).sum())


# %% [markdown]
# 4. Kalibratörler
# ----------------
# Platt: sigmoid(a*x+b)
# Beta: sigmoid(a*log(p)+b*(-log(1-p))+c)
# Bagged isotonic: bootstrap isotonic tahminlerinin ortalaması


def sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def beta_features(raw_logit: np.ndarray) -> np.ndarray:
    probability = np.clip(sigmoid(raw_logit), EPS, 1.0 - EPS)
    return np.column_stack([np.log(probability), -np.log1p(-probability)])


def logistic_model() -> LogisticRegression:
    return LogisticRegression(C=100.0, solver="lbfgs", max_iter=2000, random_state=CFG.seed)


def predict_platt(x_train, y_train, w_train, x_valid) -> np.ndarray:
    model = logistic_model()
    model.fit(np.asarray(x_train).reshape(-1, 1), y_train, sample_weight=w_train)
    return model.predict_proba(np.asarray(x_valid).reshape(-1, 1))[:, 1]


def predict_beta(x_train, y_train, w_train, x_valid) -> np.ndarray:
    model = logistic_model()
    model.fit(beta_features(np.asarray(x_train)), y_train, sample_weight=w_train)
    return model.predict_proba(beta_features(np.asarray(x_valid)))[:, 1]


def predict_bagged_isotonic(x_train, y_train, w_train, x_valid, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prediction = np.zeros(len(x_valid), dtype=np.float64)
    for _ in range(CFG.isotonic_bags):
        index = rng.integers(0, len(x_train), size=len(x_train))
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(np.asarray(x_train)[index], y_train[index], sample_weight=w_train[index])
        prediction += model.predict(np.asarray(x_valid)) / CFG.isotonic_bags
    return prediction


CALIBRATORS = {
    "platt": predict_platt,
    "beta": predict_beta,
    "bagged_isotonic": predict_bagged_isotonic,
}


# %% [markdown]
# 5. Weighted metrikler


def weighted_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = w.sum()
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if not mask.any():
            continue
        mass = w[mask].sum()
        observed = np.average(y[mask], weights=w[mask])
        predicted = np.average(p[mask], weights=w[mask])
        error += mass / total * abs(observed - predicted)
    return float(error)


def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    clipped = np.clip(p, EPS, 1.0 - EPS)
    return {
        "logloss": float(log_loss(y, clipped, sample_weight=w, labels=[0, 1])),
        "brier": float(np.average(np.square(p - y), weights=w)),
        "ece": weighted_ece(y, p, w, CFG.ece_bins),
        "auc": float(roc_auc_score(y, p, sample_weight=w)),
        "predicted_prior": float(np.average(p, weights=w)),
        "observed_prior": float(np.average(y, weights=w)),
    }


# %% [markdown]
# 6. GroupKFold karşılaştırması


def cross_validate(frame: pd.DataFrame, schemes: dict[str, np.ndarray]) -> tuple[pd.DataFrame, list[dict]]:
    x = frame["consensus_logit"].to_numpy(np.float64)
    y = frame["label"].to_numpy(np.int8)
    groups = frame["term_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(CFG.folds, frame["term_id"].nunique()))
    fold_rows = []
    summaries = []
    for scheme_name, weight in schemes.items():
        for calibrator_name, function in CALIBRATORS.items():
            oof = np.full(len(frame), np.nan, dtype=np.float64)
            for fold, (train_index, valid_index) in enumerate(splitter.split(x, y, groups)):
                kwargs = {}
                if calibrator_name == "bagged_isotonic":
                    kwargs["seed"] = CFG.seed + fold
                prediction = function(
                    x[train_index], y[train_index], weight[train_index], x[valid_index], **kwargs
                )
                oof[valid_index] = prediction
                row = {"weight_scheme": scheme_name, "calibrator": calibrator_name, "fold": fold}
                row.update(metrics(y[valid_index], prediction, weight[valid_index]))
                fold_rows.append(row)
            summary = {"weight_scheme": scheme_name, "calibrator": calibrator_name}
            summary.update(metrics(y, oof, weight))
            summary["effective_sample_size"] = effective_sample_size(weight)
            summaries.append(summary)
    return pd.DataFrame(fold_rows), summaries


# %% [markdown]
# 7. Tam veri modeli ve test cache'leri
# -------------------------------------
# Her kalibratör kendi CV sonucunda en iyi logloss veren ağırlık şemasıyla fit edilir.


def fit_full_predictions(
    frame: pd.DataFrame,
    population: pd.DataFrame,
    schemes: dict[str, np.ndarray],
    summaries: list[dict],
) -> list[dict]:
    x = frame["consensus_logit"].to_numpy(np.float64)
    y = frame["label"].to_numpy(np.int8)
    test_x = population["consensus_logit"].to_numpy(np.float64)
    outputs = []
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    for calibrator_name in CALIBRATORS:
        candidates = [s for s in summaries if s["calibrator"] == calibrator_name]
        best = min(candidates, key=lambda s: (s["logloss"], s["brier"], s["ece"]))
        weight = schemes[best["weight_scheme"]]
        if calibrator_name == "platt":
            probability = predict_platt(x, y, weight, test_x)
        elif calibrator_name == "beta":
            probability = predict_beta(x, y, weight, test_x)
        else:
            probability = predict_bagged_isotonic(x, y, weight, test_x, CFG.seed + 100)
        probability = np.clip(probability, 0.0, 1.0).astype(np.float32)
        path = CFG.output_dir / f"v229_{calibrator_name}_scores.parquet"
        pd.DataFrame({
            "id": population["id"].astype(str),
            "term_id": population["term_id"].astype(str),
            "calibrated_probability": probability,
        }).to_parquet(path, index=False, compression="zstd")
        counts = pd.Series(probability).value_counts()
        outputs.append({
            "calibrator": calibrator_name,
            "weight_scheme": best["weight_scheme"],
            "cv_logloss": best["logloss"],
            "cv_brier": best["brier"],
            "cv_ece": best["ece"],
            "mean_probability": float(probability.mean()),
            "unique_probabilities": int(pd.Series(probability).nunique()),
            "largest_plateau": int(counts.iloc[0]),
            "largest_plateau_probability": float(counts.index[0]),
            "path": str(path),
        })
    return outputs


# %% [markdown]
# 8. Ana çalışma


def compare() -> None:
    require_inputs()
    frame = load_labeled()
    population = load_population()
    schemes = weight_schemes(frame)
    fold_frame, summaries = cross_validate(frame, schemes)
    outputs = fit_full_predictions(frame, population, schemes, summaries)
    CFG.fold_report.parent.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(CFG.fold_report, index=False)
    raw = frame["raw_ipw"].to_numpy(np.float64)
    report = {
        "run": "v229_calibration_comparison",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "labeled_rows": len(frame),
        "ipw_diagnostics": {
            "raw_ess": effective_sample_size(raw),
            "raw_ess_ratio": effective_sample_size(raw) / len(raw),
            "max_over_median": float(np.max(raw) / np.median(raw)),
            "quantiles": {str(q): float(np.quantile(raw, q)) for q in [0.5, 0.9, 0.95, 0.99, 1.0]},
        },
        "cv_summaries": sorted(summaries, key=lambda s: (s["logloss"], s["brier"])),
        "full_test_outputs": outputs,
        "selection_rule": "minimum group-CV weighted logloss, then Brier, then ECE",
        "submission_generated": False,
    }
    CFG.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="V229 calibration comparison")
    parser.add_argument("stage", choices=["compare"])
    args = parser.parse_args()
    if args.stage == "compare":
        compare()


if __name__ == "__main__":
    main()
