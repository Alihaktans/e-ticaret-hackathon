"""V231 — V221 ve V230 karar politikalarını leakage olmadan karşılaştır.

V229'daki Platt modeli bu kez term_id bazlı GroupKFold ile cross-fit edilir.
Her audit satırının olasılığı, o satırın etiketini hiç görmemiş bir modelden
gelir. V221 ve farklı belirsizlik marjları aynı 1.000 satır, aynı raw-IPW
ağırlıkları ve aynı fold'lar üzerinde ölçülür.

Bu script submission üretmez. Önce karar politikasını seçmek için kanıt üretir.

Kullanım:
    python scripts/231_evaluate_crossfit_decoder_policies.py evaluate
"""

# %% [markdown]
# 0. Değerlendirme sözleşmesi
# ---------------------------
# - Etiket: V226 kör audit (1 = ilgili veya kısmen ilgili, 0 = ilgisiz)
# - Split: term_id GroupKFold
# - Kalibrasyon: raw-IPW Platt
# - Ana metrik: weighted Macro-F1
# - Belirsizlik: term_id cluster bootstrap
# - Politika: Platt güçlü ise yeni karar, gri bölgede V221

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


# %% [markdown]
# 1. Ayarlar


@dataclass(frozen=True)
class Config:
    blind: Path = Path("reports/manual_review/v226_blind_calibration.csv")
    key: Path = Path("data/processed/v226_blind_calibration_key.parquet")
    anchor: Path = Path("submissions/FINAL_BEST_VERIFIED_0p832.csv")
    report: Path = Path("reports/experiments/v231_crossfit_decoder_policies.json")
    row_report: Path = Path("reports/experiments/v231_crossfit_decoder_rows.parquet")
    fold_report: Path = Path("reports/experiments/v231_crossfit_decoder_folds.csv")
    folds: int = int(os.environ.get("V231_FOLDS", "5"))
    bootstrap_rounds: int = int(os.environ.get("V231_BOOTSTRAP", "2000"))
    seed: int = int(os.environ.get("V231_SEED", "20260711"))
    margins: str = os.environ.get("V231_MARGINS", "0.00,0.05,0.10,0.15,0.20")


CFG = Config()
EPS = 1e-6


# %% [markdown]
# 2. Veri yükleme


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


def parse_margins() -> list[float]:
    values = sorted({float(x.strip()) for x in CFG.margins.split(",") if x.strip()})
    if not values or any(not 0.0 <= x < 0.5 for x in values):
        raise ValueError("V231_MARGINS must contain values in [0, 0.5)")
    return values


def load_frame() -> pd.DataFrame:
    required = [CFG.blind, CFG.key, CFG.anchor]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V231 inputs:\n- " + "\n- ".join(missing))

    blind = pd.read_csv(CFG.blind, dtype=str)
    key = pd.read_parquet(CFG.key)
    anchor = pd.read_csv(CFG.anchor, dtype={"id": str}, usecols=["id", "prediction"])
    blind["label"] = blind["human_label"].map(normalize_label)
    frame = key.merge(blind[["audit_id", "label"]], on="audit_id", validate="one_to_one")
    frame = frame[frame["label"].notna()].copy()
    frame = frame.merge(anchor.rename(columns={"prediction": "anchor"}), on="id", validate="one_to_one")
    frame["label"] = frame["label"].astype(np.int8)
    frame["anchor"] = frame["anchor"].astype(np.int8)
    frame["weight"] = 1.0 / frame["inclusion_probability"].clip(lower=EPS)
    frame["weight"] /= frame["weight"].mean()

    needed = {"audit_id", "id", "term_id", "consensus_logit", "label", "anchor", "weight"}
    if not needed <= set(frame):
        raise ValueError(f"Missing columns: {sorted(needed - set(frame))}")
    if len(frame) < 100 or frame["label"].nunique() != 2:
        raise ValueError("Need at least 100 labeled rows containing both classes")
    if not np.isin(frame["anchor"], [0, 1]).all():
        raise ValueError("Anchor must be binary")
    return frame.reset_index(drop=True)


# %% [markdown]
# 3. Cross-fitted Platt olasılığı
# --------------------------------
# Fold içindeki audit etiketleri model eğitiminde kullanılmaz. Böylece politika
# kıyaslaması, full-data kalibratörünün eğitim performansına dayanmaz.


def new_platt() -> LogisticRegression:
    return LogisticRegression(C=100.0, solver="lbfgs", max_iter=2000, random_state=CFG.seed)


def crossfit_platt(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame["consensus_logit"].to_numpy(np.float64).reshape(-1, 1)
    y = frame["label"].to_numpy(np.int8)
    w = frame["weight"].to_numpy(np.float64)
    groups = frame["term_id"].astype(str).to_numpy()
    n_splits = min(CFG.folds, frame["term_id"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    probability = np.full(len(frame), np.nan, dtype=np.float64)
    fold_id = np.full(len(frame), -1, dtype=np.int16)
    for fold, (train_index, valid_index) in enumerate(splitter.split(x, y, groups)):
        model = new_platt()
        model.fit(x[train_index], y[train_index], sample_weight=w[train_index])
        probability[valid_index] = model.predict_proba(x[valid_index])[:, 1]
        fold_id[valid_index] = fold
    if not np.isfinite(probability).all() or (fold_id < 0).any():
        raise RuntimeError("Cross-fit prediction is incomplete")
    return probability, fold_id


# %% [markdown]
# 4. Ağırlıklı metrikler


def weighted_confusion(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    return {
        "tn": float(weight[(y == 0) & (pred == 0)].sum()),
        "fp": float(weight[(y == 0) & (pred == 1)].sum()),
        "fn": float(weight[(y == 1) & (pred == 0)].sum()),
        "tp": float(weight[(y == 1) & (pred == 1)].sum()),
    }


def macro_f1(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    c = weighted_confusion(y, pred, weight)
    f1_positive = 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], EPS)
    f1_negative = 2 * c["tn"] / max(2 * c["tn"] + c["fp"] + c["fn"], EPS)
    return float((f1_positive + f1_negative) / 2.0)


def decode(probability: np.ndarray, anchor: np.ndarray, margin: float) -> np.ndarray:
    prediction = anchor.copy()
    prediction[probability >= 0.5 + margin] = 1
    prediction[probability <= 0.5 - margin] = 0
    return prediction


def policy_metrics(y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weight: np.ndarray) -> dict:
    changed = pred != anchor
    add = (anchor == 0) & (pred == 1)
    remove = (anchor == 1) & (pred == 0)
    add_weight = weight[add].sum()
    remove_weight = weight[remove].sum()
    return {
        "weighted_macro_f1": macro_f1(y, pred, weight),
        "weighted_accuracy": float(np.average(pred == y, weights=weight)),
        "audit_changed_rows": int(changed.sum()),
        "weighted_change_mass": float(weight[changed].sum() / weight.sum()),
        "audit_add_rows": int(add.sum()),
        "audit_remove_rows": int(remove.sum()),
        "add_precision": None if add_weight <= 0 else float(weight[add & (y == 1)].sum() / add_weight),
        "remove_precision": None if remove_weight <= 0 else float(weight[remove & (y == 0)].sum() / remove_weight),
    }


# %% [markdown]
# 5. Term-cluster bootstrap
# -------------------------
# Aynı sorgudaki satırlar bağımsız kabul edilmez. Her bootstrap turunda satır
# yerine term_id kümeleri yeniden örneklenir ve Macro-F1 farkı hesaplanır.


def bootstrap_delta(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    baseline: np.ndarray,
) -> dict:
    rng = np.random.default_rng(CFG.seed)
    y = frame["label"].to_numpy(np.int8)
    w = frame["weight"].to_numpy(np.float64)
    group_indices = [g.index.to_numpy() for _, g in frame.groupby("term_id", sort=False)]
    n_groups = len(group_indices)
    values = np.empty(CFG.bootstrap_rounds, dtype=np.float64)
    for iteration in range(CFG.bootstrap_rounds):
        sampled = rng.integers(0, n_groups, size=n_groups)
        index = np.concatenate([group_indices[i] for i in sampled])
        values[iteration] = (
            macro_f1(y[index], prediction[index], w[index])
            - macro_f1(y[index], baseline[index], w[index])
        )
    return {
        "delta_vs_v221": float(macro_f1(y, prediction, w) - macro_f1(y, baseline, w)),
        "bootstrap_low_95": float(np.quantile(values, 0.025)),
        "bootstrap_high_95": float(np.quantile(values, 0.975)),
        "probability_delta_positive": float(np.mean(values > 0)),
    }


# %% [markdown]
# 6. Politika karşılaştırması ve rapor


def evaluate() -> dict:
    frame = load_frame()
    probability, fold_id = crossfit_platt(frame)
    frame["crossfit_platt_probability"] = probability
    frame["fold"] = fold_id

    y = frame["label"].to_numpy(np.int8)
    w = frame["weight"].to_numpy(np.float64)
    anchor = frame["anchor"].to_numpy(np.int8)
    baseline_metrics = policy_metrics(y, anchor, anchor, w)

    fold_rows = []
    policies = []
    policy_predictions: dict[str, np.ndarray] = {"v221_anchor": anchor}
    for margin in parse_margins():
        name = "platt_direct" if margin == 0 else f"platt_margin_{margin:.2f}".replace(".", "p")
        prediction = decode(probability, anchor, margin)
        policy_predictions[name] = prediction
        result = {"policy": name, "margin": margin}
        result.update(policy_metrics(y, prediction, anchor, w))
        result.update(bootstrap_delta(frame, prediction, anchor))
        policies.append(result)
        for fold in sorted(np.unique(fold_id)):
            mask = fold_id == fold
            fold_rows.append({
                "policy": name,
                "margin": margin,
                "fold": int(fold),
                "rows": int(mask.sum()),
                "weighted_macro_f1": macro_f1(y[mask], prediction[mask], w[mask]),
                "anchor_macro_f1": macro_f1(y[mask], anchor[mask], w[mask]),
            })

    for name, prediction in policy_predictions.items():
        frame[f"prediction_{name}"] = prediction

    clipped = np.clip(probability, EPS, 1.0 - EPS)
    calibration = {
        "weighted_logloss": float(log_loss(y, clipped, sample_weight=w, labels=[0, 1])),
        "weighted_brier": float(np.average((probability - y) ** 2, weights=w)),
        "weighted_auc": float(roc_auc_score(y, probability, sample_weight=w)),
        "weighted_observed_prior": float(np.average(y, weights=w)),
        "weighted_predicted_prior": float(np.average(probability, weights=w)),
    }

    eligible = [p for p in policies if p["bootstrap_low_95"] > 0]
    selected = max(eligible, key=lambda p: p["weighted_macro_f1"])["policy"] if eligible else None
    report = {
        "run": "v231_crossfit_decoder_policy_evaluation",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "labeled_rows": len(frame),
        "unique_terms": int(frame["term_id"].nunique()),
        "effective_sample_size": float(w.sum() ** 2 / np.square(w).sum()),
        "crossfit_platt": calibration,
        "v221_anchor": baseline_metrics,
        "policies": policies,
        "selection_rule": "highest Macro-F1 among policies whose cluster-bootstrap 95% delta lower bound is above zero",
        "selected_policy": selected,
        "warning": "Audit labels are AI-assisted judgments; statistical confidence does not turn them into competition ground truth.",
    }

    CFG.report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CFG.row_report, index=False, compression="zstd")
    pd.DataFrame(fold_rows).to_csv(CFG.fold_report, index=False)
    CFG.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# %% [markdown]
# 7. Komut satırı


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate"])
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()
