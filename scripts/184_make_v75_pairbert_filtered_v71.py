"""Filter v71's quota-preserving swaps with two source-holdout pairwise BERTs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
BASE = ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv"
V71 = ROOT / "submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv"
FEATURES = ROOT / "data/processed/v72_anchorless_trendlex_features.parquet"
CE = ROOT / "data/processed/v75_boundary_pairbert_scores.parquet"
OUT = ROOT / "submissions/final_candidates_v75"
SUMMARY = ROOT / "reports/manual_review/v75_pairbert_filtered_summary.csv"
PAIRS_OUT = ROOT / "reports/manual_review/v75_pairbert_swap_pairs.parquet"
REPORT = ROOT / "reports/experiments/v75_pairbert_filtered_report.json"


def short(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()[:8]


def build_pairs(frame: pd.DataFrame, base: np.ndarray, v71: np.ndarray) -> pd.DataFrame:
    rows = []
    trendlex = frame["trendlex_score"].to_numpy(np.float32)
    for term_id, idx in frame.groupby("term_id", sort=False).indices.items():
        idx = np.asarray(idx)
        add = idx[(base[idx] == 0) & (v71[idx] == 1)]
        drop = idx[(base[idx] == 1) & (v71[idx] == 0)]
        if len(add) != len(drop):
            raise RuntimeError(f"query quota mismatch: {term_id}")
        # Match strongest proposed additions to weakest displaced positives.
        add = add[np.argsort(-trendlex[add], kind="stable")]
        drop = drop[np.argsort(trendlex[drop], kind="stable")]
        for rank, (a, d) in enumerate(zip(add, drop), start=1):
            rows.append({"term_id": term_id, "pair_rank": rank, "add_row": int(a), "drop_row": int(d)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    f = pd.read_parquet(FEATURES)
    f["id"] = f["id"].astype(str); f["term_id"] = f["term_id"].astype(str)
    f["trendlex_score"] = .50 * f["trendyol_qpct"].astype(np.float32) + .50 * f["lex_qpct"].astype(np.float32)
    base_df = pd.read_csv(BASE); v71_df = pd.read_csv(V71)
    base_df["id"] = base_df["id"].astype(str); v71_df["id"] = v71_df["id"].astype(str)
    if not f["id"].equals(base_df["id"]) or not f["id"].equals(v71_df["id"]):
        raise RuntimeError("row alignment failed")
    base = base_df["prediction"].to_numpy(np.int8); v71 = v71_df["prediction"].to_numpy(np.int8)

    ce = pd.read_parquet(CE, columns=["id", "v75_v29hold_score", "v75_v33hold_score"])
    ce["id"] = ce["id"].astype(str)
    f = f.merge(ce, on="id", how="left", validate="one_to_one")
    boundary = base != v71
    if f.loc[boundary, ["v75_v29hold_score", "v75_v33hold_score"]].isna().any().any():
        raise RuntimeError("missing CE score on boundary")

    pairs = build_pairs(f, base, v71)
    for model in ["v29hold", "v33hold"]:
        score = f[f"v75_{model}_score"].to_numpy(np.float32)
        pairs[f"margin_{model}"] = score[pairs["add_row"]] - score[pairs["drop_row"]]
    tl = f["trendlex_score"].to_numpy(np.float32)
    pairs["margin_trendlex"] = tl[pairs["add_row"]] - tl[pairs["drop_row"]]
    pairs["margin_min_ce"] = pairs[["margin_v29hold", "margin_v33hold"]].min(axis=1)
    pairs["margin_mean_ce"] = pairs[["margin_v29hold", "margin_v33hold"]].mean(axis=1)
    pairs["models_agree"] = np.sign(pairs["margin_v29hold"]) == np.sign(pairs["margin_v33hold"])
    pairs.to_parquet(PAIRS_OUT, index=False)

    masks = {
        "ce_both_positive": (pairs.margin_v29hold > 0) & (pairs.margin_v33hold > 0),
        "ce_mean_positive": pairs.margin_mean_ce > 0,
        "ce_both_nonnegative": (pairs.margin_v29hold >= 0) & (pairs.margin_v33hold >= 0),
        "ce_both_margin0p25": (pairs.margin_v29hold > .25) & (pairs.margin_v33hold > .25),
        "ce_both_margin0p75": (pairs.margin_v29hold > .75) & (pairs.margin_v33hold > .75),
        "ce_both_margin1p5": (pairs.margin_v29hold > 1.5) & (pairs.margin_v33hold > 1.5),
    }

    base_budget = pd.Series(base, index=f["term_id"]).groupby(level=0).sum()
    rows = []
    for name, mask in masks.items():
        pred = base.copy()
        accepted = pairs.loc[mask]
        pred[accepted.add_row.to_numpy()] = 1
        pred[accepted.drop_row.to_numpy()] = 0
        got = pd.Series(pred, index=f["term_id"]).groupby(level=0).sum()
        if not got.equals(base_budget):
            raise RuntimeError(f"quota invariant failed: {name}")
        path = OUT / f"FINAL_CANDIDATE_v75_{name}_{short(name)}.csv"
        pd.DataFrame({"id": f["id"], "prediction": pred}).to_csv(path, index=False)
        rows.append({
            "variant": name, "file": str(path), "accepted_swaps": int(mask.sum()),
            "changed_rows_vs_v22": int((pred != base).sum()),
            "changed_rows_vs_v71": int((pred != v71).sum()),
            "changed_terms_vs_v22": int(f.loc[pred != base, "term_id"].nunique()),
            "positives": int(pred.sum()), "positive_ratio": float(pred.mean()),
            "mean_ce_margin": float(accepted.margin_mean_ce.mean()) if len(accepted) else np.nan,
            "min_ce_margin_mean": float(accepted.margin_min_ce.mean()) if len(accepted) else np.nan,
            "mean_trendlex_margin": float(accepted.margin_trendlex.mean()) if len(accepted) else np.nan,
        })

    summary = pd.DataFrame(rows).sort_values("accepted_swaps", ascending=False)
    summary.to_csv(SUMMARY, index=False)
    report = {
        "boundary_rows": int(boundary.sum()), "proposed_v71_swaps": int(len(pairs)),
        "model_sign_agreement": float(pairs.models_agree.mean()),
        "both_support_v71_rate": float(((pairs.margin_v29hold > 0) & (pairs.margin_v33hold > 0)).mean()),
        "both_reject_v71_rate": float(((pairs.margin_v29hold < 0) & (pairs.margin_v33hold < 0)).mean()),
        "margin_correlation": float(pairs[["margin_v29hold", "margin_v33hold"]].corr().iloc[0, 1]),
        "candidates": rows,
        "warning": "Pairwise models were selected using separate manual-source holdouts; those labels are diagnostics, not hidden truth.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
