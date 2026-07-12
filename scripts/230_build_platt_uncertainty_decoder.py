"""V230 — Pürüzsüz Platt olasılıklarıyla belirsizlik-duyarlı karar üretimi.

Bu sürüm sabit swap bütçesi veya V104/V221 sorgu cardinality'si kullanmaz.
V221 yalnızca Platt olasılığının kararsız olduğu bölgede bir anchor'dır:

* p >= 0.5 + margin  -> alakalı
* p <= 0.5 - margin  -> alakasız
* aradaki bölge      -> doğrulanmış V221 kararını koru

Margin büyüdükçe yeni modele geçmek için daha güçlü kanıt gerekir. Her margin
için değişiklik sayısı model tarafından kendiliğinden belirlenir; add/remove
eşitliği ve pozitif oran kısıtı yoktur.

Kullanım:
    python scripts/230_build_platt_uncertainty_decoder.py audit
    python scripts/230_build_platt_uncertainty_decoder.py build
    python scripts/230_build_platt_uncertainty_decoder.py all
"""

# %% [markdown]
# 0. Deney akışı
# ----------------
# 1) V229 raw-IPW Platt olasılıklarını ve V221 anchor'ını doğrula.
# 2) Doğrudan Platt ve dört belirsizlik marjı için karar üret.
# 3) Hiçbir adayın pozitif sayısını veya swap sayısını önceden sabitleme.
# 4) Satır ve query seviyesindeki değişimleri ayrıntılı raporla.

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# %% [markdown]
# 1. Merkezi ayarlar


@dataclass(frozen=True)
class Config:
    platt_scores: Path = Path("data/processed/v229_platt_scores.parquet")
    anchor: Path = Path("submissions/FINAL_BEST_VERIFIED_0p832.csv")
    pairs: Path = Path("data/raw/submission_pairs.csv")
    sample: Path = Path("data/raw/sample_submission.csv")
    output_dir: Path = Path("submissions/final_candidates_v230")
    report: Path = Path("reports/experiments/v230_platt_uncertainty_decoder.json")
    margins: str = os.environ.get("V230_MARGINS", "0.05,0.10,0.15,0.20")


CFG = Config()


# %% [markdown]
# 2. Girdi denetimi
# -----------------
# ID sırası özellikle kontrol edilir. Böylece 3.36 milyon satırda sessiz bir
# sıra kayması yanlış submission üretemez.


def parse_margins() -> list[float]:
    values = sorted({float(x.strip()) for x in CFG.margins.split(",") if x.strip()})
    if not values or any(not 0.0 <= x < 0.5 for x in values):
        raise ValueError("V230_MARGINS values must be in [0, 0.5)")
    return values


def probability_column(frame: pd.DataFrame) -> str:
    preferred = ["calibrated_probability", "probability", "platt_probability", "score"]
    for name in preferred:
        if name in frame and pd.api.types.is_numeric_dtype(frame[name]):
            return name
    numeric = [c for c in frame if c != "id" and pd.api.types.is_numeric_dtype(frame[c])]
    if len(numeric) != 1:
        raise ValueError(f"Cannot identify Platt probability column: {frame.columns.tolist()}")
    return numeric[0]


def load_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    required = [CFG.platt_scores, CFG.anchor, CFG.pairs, CFG.sample]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V230 inputs:\n- " + "\n- ".join(missing))

    pairs = pd.read_csv(CFG.pairs, usecols=["id", "term_id"], dtype=str)
    sample = pd.read_csv(CFG.sample, usecols=["id"], dtype=str)
    anchor = pd.read_csv(CFG.anchor, dtype={"id": str})
    scores = pd.read_parquet(CFG.platt_scores)
    column = probability_column(scores)

    if not pairs["id"].equals(sample["id"]):
        raise ValueError("submission_pairs and sample_submission ID order mismatch")
    if not pairs["id"].equals(anchor["id"]):
        raise ValueError("V221 anchor ID order mismatch")
    if "id" in scores and not pairs["id"].equals(scores["id"].astype(str).reset_index(drop=True)):
        raise ValueError("V229 Platt score ID order mismatch")
    if len(scores) != len(pairs):
        raise ValueError("V229 Platt score row count mismatch")

    old = anchor["prediction"].to_numpy(np.int8)
    probability = scores[column].to_numpy(np.float64)
    if not np.isin(old, [0, 1]).all():
        raise ValueError("Anchor prediction must be binary")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Platt probabilities must be finite and in [0, 1]")
    return pairs, old, probability, column


def audit_inputs() -> dict:
    pairs, old, probability, column = load_inputs()
    result = {
        "rows": len(pairs),
        "queries": int(pairs["term_id"].nunique()),
        "anchor_positive_ratio": float(old.mean()),
        "platt_probability_column": column,
        "platt_mean_probability": float(probability.mean()),
        "platt_unique_probabilities": int(pd.Series(probability).nunique()),
        "margins": parse_margins(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# %% [markdown]
# 3. Belirsizlik-duyarlı decoder
# ------------------------------
# Güvenli bölgedeki Platt kararı doğrudan uygulanır. Sadece 0.5 çevresindeki
# gri bölgede V221 korunur. Bu bir "top-N swap" değildir: kaç satırın değişeceği
# olasılık dağılımından çıkar.


def decode(probability: np.ndarray, anchor: np.ndarray, margin: float) -> np.ndarray:
    prediction = anchor.copy()
    prediction[probability >= 0.5 + margin] = 1
    prediction[probability <= 0.5 - margin] = 0
    return prediction


def summarize_candidate(
    pairs: pd.DataFrame,
    old: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    margin: float,
    path: Path,
) -> dict:
    changed = prediction != old
    query_frame = pd.DataFrame({
        "term_id": pairs["term_id"],
        "anchor": old,
        "prediction": prediction,
    })
    query_counts = query_frame.groupby("term_id", sort=False)[["anchor", "prediction"]].sum()
    delta = query_counts["prediction"] - query_counts["anchor"]
    uncertain = np.abs(probability - 0.5) < margin
    return {
        "margin": margin,
        "decision_rule": f"Platt outside [{0.5-margin:.2f}, {0.5+margin:.2f}]; V221 inside",
        "changes_vs_v221": int(changed.sum()),
        "change_ratio": float(changed.mean()),
        "adds": int(((old == 0) & (prediction == 1)).sum()),
        "removes": int(((old == 1) & (prediction == 0)).sum()),
        "positive_rows": int(prediction.sum()),
        "positive_ratio": float(prediction.mean()),
        "anchor_preserved_uncertain_rows": int(uncertain.sum()),
        "queries_with_cardinality_change": int(delta.ne(0).sum()),
        "query_delta_abs_sum": int(delta.abs().sum()),
        "query_delta_min": int(delta.min()),
        "query_delta_max": int(delta.max()),
        "path": str(path),
    }


# %% [markdown]
# 4. Adayların yazılması
# ----------------------
# Direct-Platt bilimsel referanstır; agresif olabilir. Margin adayları ise aynı
# pürüzsüz kalibrasyonu kontrollü biçimde kullanır. Script kazanan ilan etmez;
# gerçek leaderboard etiketi olmadan böyle bir iddia güvenilir olmaz.


def build() -> dict:
    audit = audit_inputs()
    pairs, old, probability, column = load_inputs()
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    CFG.report.parent.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    configurations = [("direct", 0.0)] + [
        (f"margin_{margin:.2f}".replace(".", "p"), margin) for margin in parse_margins()
    ]
    for tag, margin in configurations:
        prediction = decode(probability, old, margin)
        path = CFG.output_dir / f"FINAL_CANDIDATE_v230_platt_{tag}.csv"
        output = pd.DataFrame({"id": pairs["id"], "prediction": prediction})
        if output["id"].duplicated().any() or output["prediction"].isna().any():
            raise ValueError(f"Invalid output generated for margin={margin}")
        output.to_csv(path, index=False)
        candidates.append(summarize_candidate(pairs, old, prediction, probability, margin, path))

    report = {
        "run": "v230_platt_uncertainty_decoder",
        "config": {key: str(value) for key, value in asdict(CFG).items()},
        "calibrator": "raw-IPW Platt selected by grouped-CV weighted logloss",
        "fixed_swap_budget": False,
        "fixed_positive_ratio": False,
        "fixed_query_cardinality": False,
        "audit": audit,
        "candidates": candidates,
        "recommendation": "Use margin candidates for fresh blind audit; direct is an aggressive reference.",
        "warning": "V226 labels are AI-assisted audit judgments, not competition ground truth.",
    }
    CFG.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# %% [markdown]
# 5. Komut satırı girişi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "build", "all"])
    args = parser.parse_args()
    if args.command == "audit":
        audit_inputs()
    else:
        build()


if __name__ == "__main__":
    main()
