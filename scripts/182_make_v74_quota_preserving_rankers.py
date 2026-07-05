"""Build quota-preserving rankers after the v71/v72 public-LB ablation.

The important experimental fact is that changing the per-query positive budget
(v72) was destructive, while re-ranking inside the v22 budget (v71) improved
public Macro-F1.  This script therefore never learns or changes query budgets.
It tests genuinely different row scorers under exactly the same budget and
adds conservative anchor bonuses to control how far a candidate moves from v22.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
FEATURES = ROOT / "data/processed/v72_anchorless_trendlex_features.parquet"
BASE = ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv"
V71 = ROOT / "submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv"
OUT = ROOT / "submissions/final_candidates_v74"
REPORT = ROOT / "reports/experiments/v74_quota_preserving_report.json"
SUMMARY = ROOT / "reports/manual_review/v74_quota_preserving_summary.csv"


def short(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:8]


def top_by_budget(term: pd.Series, score: np.ndarray, budgets: pd.Series) -> np.ndarray:
    w = pd.DataFrame({"term_id": term, "score": score, "row": np.arange(len(term))})
    rank = w.groupby("term_id", sort=False)["score"].rank(method="first", ascending=False)
    return (rank.to_numpy() <= w["term_id"].map(budgets).to_numpy()).astype(np.int8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    f = pd.read_parquet(FEATURES)
    f["id"] = f["id"].astype(str)
    f["term_id"] = f["term_id"].astype(str)
    base_df = pd.read_csv(BASE)
    v71_df = pd.read_csv(V71)
    base_df["id"] = base_df["id"].astype(str)
    v71_df["id"] = v71_df["id"].astype(str)
    if not f["id"].equals(base_df["id"]) or not f["id"].equals(v71_df["id"]):
        raise RuntimeError("feature/submission row alignment failed")

    base = base_df["prediction"].to_numpy(np.int8)
    v71 = v71_df["prediction"].to_numpy(np.int8)
    budgets = pd.Series(base, index=f["term_id"]).groupby(level=0).sum()

    trend = f["trendyol_qpct"].to_numpy(np.float32)
    lex = f["lex_qpct"].to_numpy(np.float32)
    v69 = f["v69_qpct"].to_numpy(np.float32)
    poison = f["v69_poison_penalty"].to_numpy(np.float32)

    # Independent score families.  v72_manual is the source-balanced optimum
    # found on v29/v33 swap audits, now used only for within-query ordering.
    scores = {
        "trendlex_50_50": .50 * trend + .50 * lex,
        "trendlex_35_65": .35 * trend + .65 * lex,
        "trendlex_25_60_v69_15": .25 * trend + .60 * lex + .15 * v69 - .15 * poison,
        "v72_manual": .2174 * trend + .5217 * lex + .2609 * v69 - .30 * poison,
        "robust_equal3": (trend + lex + v69) / 3.0 - .20 * poison,
    }

    variants: dict[str, np.ndarray] = {}
    # Exact scorer variants.
    for name, score in scores.items():
        variants[name] = top_by_budget(f["term_id"], score, budgets)

    # Conservative paths between the known 0.80 anchor and independent rankers.
    # A binary bonus is deliberately transparent and preserves every query quota.
    for family in ["trendlex_35_65", "v72_manual", "robust_equal3"]:
        score = scores[family]
        for bonus in [.02, .05, .08, .12]:
            variants[f"{family}_v22bonus{str(bonus).replace('.', 'p')}"] = top_by_budget(
                f["term_id"], score + bonus * base, budgets
            )

    rows = []
    for name, pred in variants.items():
        # Hard invariants: same global count AND same count for every query.
        got = pd.Series(pred, index=f["term_id"]).groupby(level=0).sum()
        if int(pred.sum()) != int(base.sum()) or not got.equals(budgets):
            raise RuntimeError(f"quota invariant failed: {name}")
        path = OUT / f"FINAL_CANDIDATE_v74_{name}_{short(name)}.csv"
        pd.DataFrame({"id": f["id"], "prediction": pred}).to_csv(path, index=False)
        changed_base = pred != base
        changed_v71 = pred != v71
        rows.append({
            "variant": name,
            "file": str(path),
            "positives": int(pred.sum()),
            "positive_ratio": float(pred.mean()),
            "changed_rows_vs_v22": int(changed_base.sum()),
            "changed_terms_vs_v22": int(f.loc[changed_base, "term_id"].nunique()),
            "changed_rows_vs_v71": int(changed_v71.sum()),
            "jaccard_vs_v22": float(np.logical_and(pred == 1, base == 1).sum() / np.logical_or(pred == 1, base == 1).sum()),
            "jaccard_vs_v71": float(np.logical_and(pred == 1, v71 == 1).sum() / np.logical_or(pred == 1, v71 == 1).sum()),
            "mean_poison_selected": float(poison[pred == 1].mean()),
        })

    summary = pd.DataFrame(rows).sort_values(["changed_rows_vs_v71", "changed_rows_vs_v22"])
    summary.to_csv(SUMMARY, index=False)
    report = {
        "premise": "v71 public=0.81 with v22 per-query budgets; v72 global ratio0.32 public=0.63",
        "base": str(BASE),
        "known_best": str(V71),
        "rows": len(f),
        "terms": int(f["term_id"].nunique()),
        "fixed_positives": int(base.sum()),
        "fixed_positive_ratio": float(base.mean()),
        "candidates": rows,
        "warning": "No hidden labels were used. Internal distances are diagnostics, not Macro-F1 estimates.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(summary.to_string(index=False))
    print(REPORT)


if __name__ == "__main__":
    main()
