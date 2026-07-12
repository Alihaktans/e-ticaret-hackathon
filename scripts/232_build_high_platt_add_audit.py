"""V232 — V221 negatifleri için bağımsız yüksek-Platt add audit'i.

Bu script submission üretmez. V221'in 0 dediği fakat raw-IPW Platt modelinin
yüksek olasılık verdiği satırları beş skor bandından kör olarak örnekler.
V226'da kullanılan ID'ler dışlanır; böylece değerlendirme örnekleri yenidir.

Kullanım:
    python scripts/232_build_high_platt_add_audit.py build
    # reports/manual_review/v232_high_platt_add_audit.csv içindeki human_label
    # sütununu 0/1 ile doldur.
    python scripts/232_build_high_platt_add_audit.py score
"""

# %% [markdown]
# 0. Deney tasarımı
# -----------------
# Hedef popülasyon: V221=0 ve Platt p>=0.70 olan satırlar.
# Skor bantları: [0.70,0.80), [0.80,0.90), [0.90,0.95), [0.95,0.98), [0.98,1].
# Her banttan basit rastgele örnek alınır ve inclusion_probability anahtar
# dosyasında saklanır. Kör CSV skor, bant ve V221 kararını göstermez.

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# %% [markdown]
# 1. Ayarlar


@dataclass(frozen=True)
class Config:
    scores: Path = Path("data/processed/v229_platt_scores.parquet")
    anchor: Path = Path("submissions/FINAL_BEST_VERIFIED_0p832.csv")
    pairs: Path = Path("data/raw/submission_pairs.csv")
    terms: Path = Path("data/raw/terms.csv")
    items: Path = Path("data/raw/items.csv")
    previous_key: Path = Path("data/processed/v226_blind_calibration_key.parquet")
    blind: Path = Path("reports/manual_review/v232_high_platt_add_audit.csv")
    key: Path = Path("data/processed/v232_high_platt_add_audit_key.parquet")
    report: Path = Path("reports/experiments/v232_high_platt_add_audit.json")
    rows_per_band: int = int(os.environ.get("V232_ROWS_PER_BAND", "200"))
    bootstrap_rounds: int = int(os.environ.get("V232_BOOTSTRAP", "5000"))
    seed: int = int(os.environ.get("V232_SEED", "20260712"))


CFG = Config()
BANDS = [
    ("p070_080", 0.70, 0.80, False),
    ("p080_090", 0.80, 0.90, False),
    ("p090_095", 0.90, 0.95, False),
    ("p095_098", 0.95, 0.98, False),
    ("p098_100", 0.98, 1.00, True),
]
EPS = 1e-9


# %% [markdown]
# 2. Veri denetimi ve skor sütunu


def probability_column(frame: pd.DataFrame) -> str:
    for name in ["calibrated_probability", "probability", "platt_probability", "score"]:
        if name in frame and pd.api.types.is_numeric_dtype(frame[name]):
            return name
    numeric = [c for c in frame if c != "id" and pd.api.types.is_numeric_dtype(frame[c])]
    if len(numeric) != 1:
        raise ValueError(f"Cannot identify probability column: {frame.columns.tolist()}")
    return numeric[0]


def require_inputs(for_score: bool = False) -> None:
    required = [CFG.scores, CFG.anchor, CFG.pairs, CFG.terms, CFG.items]
    if for_score:
        required.extend([CFG.blind, CFG.key])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V232 inputs:\n- " + "\n- ".join(missing))


def load_population() -> pd.DataFrame:
    pairs = pd.read_csv(CFG.pairs, usecols=["id", "term_id", "item_id"], dtype=str)
    anchor = pd.read_csv(CFG.anchor, usecols=["id", "prediction"], dtype={"id": str})
    scores = pd.read_parquet(CFG.scores)
    column = probability_column(scores)
    if not pairs["id"].equals(anchor["id"]):
        raise ValueError("Anchor ID order mismatch")
    if "id" in scores and not pairs["id"].equals(scores["id"].astype(str).reset_index(drop=True)):
        raise ValueError("Platt score ID order mismatch")
    if len(scores) != len(pairs):
        raise ValueError("Platt score row count mismatch")
    probability = scores[column].to_numpy(np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid Platt probabilities")
    pairs["anchor"] = anchor["prediction"].to_numpy(np.int8)
    pairs["probability"] = probability
    return pairs


# %% [markdown]
# 3. Bağımsız tabakalı örnekleme
# -----------------------------
# Önceki V226 ID'leri dışarıda bırakılır. Her bandın dahil edilme olasılığı
# take / eligible_population olarak kaydedilir; score aşamasında IPW uygulanır.


def assign_band(probability: pd.Series) -> pd.Series:
    result = pd.Series("", index=probability.index, dtype="object")
    for name, low, high, inclusive_high in BANDS:
        mask = probability.ge(low) & (probability.le(high) if inclusive_high else probability.lt(high))
        result.loc[mask] = name
    return result


def load_selected_item_metadata(item_ids: set[str]) -> pd.DataFrame:
    chunks = []
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    for chunk in pd.read_csv(CFG.items, usecols=columns, dtype=str, chunksize=100_000):
        hit = chunk[chunk["item_id"].isin(item_ids)]
        if not hit.empty:
            chunks.append(hit)
    if not chunks:
        raise ValueError("No item metadata matched audit rows")
    return pd.concat(chunks, ignore_index=True).drop_duplicates("item_id")


def build() -> dict:
    require_inputs()
    population = load_population()
    excluded: set[str] = set()
    if CFG.previous_key.exists():
        excluded = set(pd.read_parquet(CFG.previous_key, columns=["id"])["id"].astype(str))
    eligible = population[(population["anchor"] == 0) & population["probability"].ge(0.70)].copy()
    eligible = eligible[~eligible["id"].isin(excluded)].copy()
    eligible["score_band"] = assign_band(eligible["probability"])
    if eligible["score_band"].eq("").any():
        raise RuntimeError("Some eligible rows did not receive a score band")

    rng = np.random.default_rng(CFG.seed)
    selections = []
    band_report = []
    for name, low, high, _ in BANDS:
        pool = eligible[eligible["score_band"].eq(name)]
        take = min(CFG.rows_per_band, len(pool))
        if take == 0:
            band_report.append({"score_band": name, "low": low, "high": high, "population": 0, "sample": 0})
            continue
        indices = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        selected = pool.loc[indices].copy()
        selected["band_population"] = len(pool)
        selected["band_sample"] = take
        selected["inclusion_probability"] = take / len(pool)
        selections.append(selected)
        band_report.append({
            "score_band": name,
            "low": low,
            "high": high,
            "population": len(pool),
            "sample": take,
            "inclusion_probability": take / len(pool),
        })
    if not selections:
        raise ValueError("No eligible high-Platt V221-negative rows")
    selected = pd.concat(selections, ignore_index=True)
    selected["audit_id"] = [f"V232_{i:04d}" for i in range(1, len(selected) + 1)]

    key_columns = [
        "audit_id", "id", "term_id", "item_id", "anchor", "probability",
        "score_band", "band_population", "band_sample", "inclusion_probability",
    ]
    CFG.key.parent.mkdir(parents=True, exist_ok=True)
    selected[key_columns].to_parquet(CFG.key, index=False, compression="zstd")

    terms = pd.read_csv(CFG.terms, usecols=["term_id", "query"], dtype=str).drop_duplicates("term_id")
    items = load_selected_item_metadata(set(selected["item_id"]))
    blind = selected[["audit_id", "term_id", "item_id"]].merge(terms, on="term_id", validate="many_to_one")
    blind = blind.merge(items, on="item_id", validate="many_to_one")
    blind["attributes"] = blind["attributes"].fillna("").str.slice(0, 700)
    blind["human_label"] = ""
    blind["notes"] = ""
    blind = blind[[
        "audit_id", "query", "title", "category", "brand", "gender",
        "age_group", "attributes", "human_label", "notes",
    ]].sample(frac=1.0, random_state=CFG.seed).reset_index(drop=True)
    CFG.blind.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(CFG.blind, index=False, encoding="utf-8-sig")

    result = {
        "run": "v232_high_platt_add_audit_build",
        "audit_rows": len(blind),
        "previous_v226_ids_excluded": len(excluded),
        "bands": band_report,
        "blind_file": str(CFG.blind),
        "key_file": str(CFG.key),
        "label_definition": {"1": "relevant or partially relevant", "0": "irrelevant", "?": "uncertain"},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# %% [markdown]
# 4. Etiket okuma


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


def load_labeled() -> pd.DataFrame:
    blind = pd.read_csv(CFG.blind, dtype=str)
    # Excel/CSV round-trips can leave one or more UTF-8 BOM characters on the
    # first header. Normalize all headers before selecting audit_id.
    blind.columns = blind.columns.map(lambda name: str(name).lstrip("\ufeff").strip())
    key = pd.read_parquet(CFG.key)
    blind["label"] = blind["human_label"].map(normalize_label)
    frame = key.merge(blind[["audit_id", "label"]], on="audit_id", validate="one_to_one")
    frame = frame[frame["label"].notna()].copy()
    frame["label"] = frame["label"].astype(np.int8)
    frame["weight"] = 1.0 / frame["inclusion_probability"].clip(lower=EPS)
    if len(frame) < 100 or frame["label"].nunique() != 2:
        raise ValueError("Need at least 100 labeled rows containing both classes")
    return frame.reset_index(drop=True)


# %% [markdown]
# 5. Precision ve term-cluster bootstrap
# --------------------------------------
# Add-only politikasının net kazancı 2*precision-1 ile ilişkilidir. Eşik ancak
# precision güven aralığının alt sınırı 0.5'i geçtiğinde güvenilir sayılır.


def weighted_precision(frame: pd.DataFrame) -> float:
    return float(np.average(frame["label"], weights=frame["weight"]))


def cluster_bootstrap_precision(frame: pd.DataFrame, seed_offset: int) -> dict:
    groups = [g.index.to_numpy() for _, g in frame.groupby("term_id", sort=False)]
    rng = np.random.default_rng(CFG.seed + seed_offset)
    values = np.empty(CFG.bootstrap_rounds, dtype=np.float64)
    for iteration in range(CFG.bootstrap_rounds):
        sampled = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[i] for i in sampled])
        values[iteration] = weighted_precision(frame.loc[index])
    return {
        "bootstrap_low_95": float(np.quantile(values, 0.025)),
        "bootstrap_high_95": float(np.quantile(values, 0.975)),
        "probability_precision_above_0p5": float(np.mean(values > 0.5)),
    }


def summarize_slice(frame: pd.DataFrame, name: str, seed_offset: int) -> dict:
    weight = frame["weight"].to_numpy(np.float64)
    result = {
        "slice": name,
        "labeled_rows": len(frame),
        "unique_terms": int(frame["term_id"].nunique()),
        "effective_sample_size": float(weight.sum() ** 2 / np.square(weight).sum()),
        "weighted_add_precision": weighted_precision(frame),
        "estimated_net_gain_per_add": float(2 * weighted_precision(frame) - 1),
    }
    result.update(cluster_bootstrap_precision(frame, seed_offset))
    return result


def score() -> dict:
    require_inputs(for_score=True)
    frame = load_labeled()
    by_band = []
    for index, (name, _, _, _) in enumerate(BANDS):
        part = frame[frame["score_band"].eq(name)]
        if not part.empty:
            by_band.append(summarize_slice(part, name, index + 1))

    thresholds = []
    for index, threshold in enumerate([0.70, 0.80, 0.90, 0.95, 0.98]):
        part = frame[frame["probability"].ge(threshold)]
        if not part.empty:
            row = summarize_slice(part, f"p>={threshold:.2f}", 100 + index)
            row["threshold"] = threshold
            row["eligible"] = bool(row["labeled_rows"] >= 100 and row["bootstrap_low_95"] > 0.5)
            thresholds.append(row)

    eligible = [row for row in thresholds if row["eligible"]]
    # En düşük güvenilir eşik en fazla doğrulanmış düzeltmeyi kapsar.
    selected = min(eligible, key=lambda row: row["threshold"])["threshold"] if eligible else None
    result = {
        "run": "v232_high_platt_add_audit_score",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "labeled_rows": len(frame),
        "by_band": by_band,
        "cumulative_thresholds": thresholds,
        "selection_rule": "lowest threshold with >=100 labels and term-cluster bootstrap 95% precision lower bound >0.5",
        "selected_add_threshold": selected,
        "submission_generated": False,
        "warning": "Labels remain audit judgments, not competition ground truth.",
    }
    CFG.report.parent.mkdir(parents=True, exist_ok=True)
    CFG.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# %% [markdown]
# 6. Komut satırı


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "score"])
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        score()


if __name__ == "__main__":
    main()
