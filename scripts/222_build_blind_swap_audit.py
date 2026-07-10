"""Build and score a blind human audit for V221 swap budgets.

Build:
    python scripts/222_build_blind_swap_audit.py build

Fill ``human_label`` with 1 (relevant or partially relevant), 0 (irrelevant),
or ? (uncertain), then score:
    python scripts/222_build_blind_swap_audit.py score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
REPORTS = ROOT / "reports/experiments"
MANUAL = ROOT / "reports/manual_review"

SCORED = PROC / "v221_embedding_guarded_disagreement.parquet"
TERMS = RAW / "terms.csv"
ITEMS = RAW / "items.csv"

BLIND = MANUAL / "v222_blind_swap_audit.csv"
KEY = PROC / "v222_blind_swap_audit_key.parquet"
REPORT = REPORTS / "v222_blind_swap_audit_report.json"

SEED = 20260710
SAMPLES_PER_DIRECTION_TIER = 50
TIERS = [
    (1, 500, "rank_00001_00500"),
    (501, 1000, "rank_00501_01000"),
    (1001, 3000, "rank_01001_03000"),
    (3001, 5000, "rank_03001_05000"),
    (5001, 10000, "rank_05001_10000"),
    (10001, 20000, "rank_10001_20000"),
]
BUDGETS = [500, 1000, 3000, 5000, 10000, 20000]


def percentile(values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.rank(method="average", pct=True, ascending=ascending).astype(np.float32)


def add_priorities(rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for direction, frame in rows.groupby("direction", sort=False):
        frame = frame.copy()
        ascending = direction == "add"
        if direction == "add":
            frame["priority"] = (
                0.55 * percentile(frame["trendyol_embedding"])
                + 0.25 * percentile(frame["v220_score"])
                + 0.20 * percentile(frame["v219_score"])
            )
            frame["semantic_guard"] = frame["trendyol_embedding"].ge(
                frame["trendyol_embedding"].quantile(0.75)
            )
        else:
            frame["priority"] = (
                0.55 * percentile(frame["trendyol_embedding"], ascending=False)
                + 0.25 * percentile(frame["v220_score"], ascending=False)
                + 0.20 * percentile(frame["v219_score"], ascending=False)
            )
            frame["semantic_guard"] = frame["trendyol_embedding"].le(
                frame["trendyol_embedding"].quantile(0.25)
            )
        frame = frame[frame["semantic_guard"]].sort_values("priority", ascending=False).reset_index(drop=True)
        frame["swap_rank"] = np.arange(1, len(frame) + 1, dtype=np.int32)
        frame["expected_label"] = np.int8(1 if ascending else 0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_item_metadata(item_ids: set[str]) -> pd.DataFrame:
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    frames = []
    for chunk in pd.read_csv(ITEMS, usecols=columns, dtype=str, chunksize=100000, low_memory=False):
        selected = chunk[chunk["item_id"].isin(item_ids)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("No item metadata found for audit rows")
    return pd.concat(frames, ignore_index=True).drop_duplicates("item_id")


def build_audit() -> None:
    if not SCORED.exists():
        raise FileNotFoundError(f"Run V221 swap first: {SCORED}")
    rows = add_priorities(pd.read_parquet(SCORED))
    rng = np.random.default_rng(SEED)
    samples = []
    for direction in ("add", "remove"):
        direction_rows = rows[rows["direction"].eq(direction)]
        for lower, upper, tier_name in TIERS:
            pool = direction_rows[direction_rows["swap_rank"].between(lower, upper)].copy()
            take = min(SAMPLES_PER_DIRECTION_TIER, len(pool))
            if take == 0:
                continue
            chosen = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            sample = pool.loc[chosen].copy()
            sample["tier"] = tier_name
            samples.append(sample)
    audit = pd.concat(samples, ignore_index=True)
    audit["audit_id"] = [f"AUDIT_{index:04d}" for index in range(1, len(audit) + 1)]

    terms = pd.read_csv(TERMS, usecols=["term_id", "query"], dtype=str).drop_duplicates("term_id")
    items = load_item_metadata(set(audit["item_id"].astype(str)))
    audit = audit.merge(terms, on="term_id", how="left", validate="many_to_one")
    audit = audit.merge(items, on="item_id", how="left", validate="many_to_one")
    if audit["query"].isna().any() or audit["title"].isna().any():
        raise ValueError("Missing query or item text in blind audit")

    key_columns = [
        "audit_id", "id", "term_id", "item_id", "direction", "tier", "swap_rank", "expected_label",
        "teacher", "v219_pred", "v220_pred", "v219_score", "v220_score", "trendyol_embedding", "priority",
    ]
    KEY.parent.mkdir(parents=True, exist_ok=True)
    audit[key_columns].to_parquet(KEY, index=False, compression="zstd")

    blind = audit[["audit_id", "query", "title", "category", "brand", "gender", "age_group", "attributes"]].copy()
    blind["attributes"] = blind["attributes"].fillna("").astype(str).str.slice(0, 700)
    blind["human_label"] = ""
    blind["notes"] = ""
    blind = blind.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    BLIND.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(BLIND, index=False, encoding="utf-8-sig")

    summary = {
        "rows": len(blind),
        "samples_per_direction_tier": SAMPLES_PER_DIRECTION_TIER,
        "directions": audit["direction"].value_counts().to_dict(),
        "tiers": audit["tier"].value_counts().to_dict(),
        "blind_file": str(BLIND),
        "key_file": str(KEY),
        "label_definition": {"1": "relevant or partially relevant", "0": "irrelevant", "?": "uncertain"},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def normalized_label(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "relevant", "alakali", "yarı alakalı", "yari alakali"}:
        return 1
    if text in {"0", "0.0", "irrelevant", "alakasiz", "alakasız"}:
        return 0
    if text in {"", "?", "emin degilim", "emin değilim", "uncertain"}:
        return None
    raise ValueError(f"Unknown human_label value: {value!r}")


def score_audit() -> None:
    if not BLIND.exists() or not KEY.exists():
        raise FileNotFoundError("Run V222 build first")
    blind = pd.read_csv(BLIND, dtype=str)
    key = pd.read_parquet(KEY)
    if blind["audit_id"].duplicated().any():
        raise ValueError("Duplicate audit IDs")
    labels = blind[["audit_id", "human_label"]].copy()
    labels["label"] = labels["human_label"].map(normalized_label)
    scored = key.merge(labels[["audit_id", "label"]], on="audit_id", how="left", validate="one_to_one")
    labeled = scored[scored["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(np.int8)
    labeled["correct_swap"] = labeled["label"].eq(labeled["expected_label"])

    by_tier = []
    for (direction, tier), frame in labeled.groupby(["direction", "tier"], sort=True):
        by_tier.append(
            {
                "direction": direction,
                "tier": tier,
                "labeled": len(frame),
                "swap_precision": float(frame["correct_swap"].mean()),
                "net_gain_per_swap": float((2 * frame["correct_swap"].astype(int) - 1).mean()),
            }
        )

    budgets = []
    for budget in BUDGETS:
        frame = labeled[labeled["swap_rank"].le(budget)]
        directions = {}
        for direction, group in frame.groupby("direction"):
            directions[direction] = {
                "labeled": len(group),
                "precision": float(group["correct_swap"].mean()),
            }
        budgets.append(
            {
                "budget_each_direction": budget,
                "labeled": len(frame),
                "swap_precision": float(frame["correct_swap"].mean()) if len(frame) else None,
                "estimated_net_gain_per_change": (
                    float((2 * frame["correct_swap"].astype(int) - 1).mean()) if len(frame) else None
                ),
                "directions": directions,
            }
        )

    report = {
        "audit_rows": len(scored),
        "labeled_rows": len(labeled),
        "unlabeled_or_uncertain_rows": int(len(scored) - len(labeled)),
        "by_tier": by_tier,
        "budget_estimates": budgets,
        "interpretation": "Positive net gain requires swap precision above 0.5 in both directions.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["build", "score"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "build":
        build_audit()
    else:
        score_audit()


if __name__ == "__main__":
    main()
