"""Create independent and consensus candidates from the audited Trendyol signal."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(".")
FEATURES = ROOT / "data/processed/v70_trendyol_embedding/v70_compact_rank_features.parquet"
BASE_PATH = ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv"
OUT = ROOT / "submissions"
REPORT = ROOT / "reports/experiments/v71_trendyol_candidate_summary.json"


def save_candidate(ids: pd.Series, pred: np.ndarray, name: str, base: np.ndarray, terms: pd.Series) -> dict:
    path = OUT / name
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(path, index=False)
    changed = pred != base
    return {
        "file": str(path), "positive_ratio": float(pred.mean()), "positives": int(pred.sum()),
        "changed_rows_vs_v22": int(changed.sum()), "changed_terms_vs_v22": int(terms[changed].nunique()),
        "jaccard_positive_vs_v22": float(np.logical_and(pred == 1, base == 1).sum() / np.logical_or(pred == 1, base == 1).sum()),
    }


def top_by_query(frame: pd.DataFrame, score: np.ndarray, budgets: pd.Series) -> np.ndarray:
    work = pd.DataFrame({"term_id": frame["term_id"], "score": score, "row": np.arange(len(frame))})
    work["rank"] = work.groupby("term_id", sort=False)["score"].rank(method="first", ascending=False)
    limit = work["term_id"].map(budgets).to_numpy()
    return (work["rank"].to_numpy() <= limit).astype(np.int8)


def consensus_swaps(frame: pd.DataFrame, base: np.ndarray, trend: np.ndarray, lex: np.ndarray) -> list[tuple]:
    # Within every query, pair the strongest proposed addition with the weakest
    # current positive. Keep only swaps improved by BOTH independent signals.
    blend = .5 * trend + .5 * lex
    proposals = []
    groups = frame.groupby("term_id", sort=False).indices
    for term_id, idx in groups.items():
        idx = np.asarray(idx)
        pos = idx[base[idx] == 1]
        neg = idx[base[idx] == 0]
        if not len(pos) or not len(neg):
            continue
        add = neg[np.argsort(-blend[neg])]
        drop = pos[np.argsort(blend[pos])]
        for ai, di in zip(add, drop):
            dt = float(trend[ai] - trend[di])
            dl = float(lex[ai] - lex[di])
            if dt > 0 and dl > 0:
                # Conservative confidence: the weaker of the two improvements.
                proposals.append((min(dt, dl), .5 * (dt + dl), int(ai), int(di), str(term_id)))
    proposals.sort(reverse=True)
    return proposals


def apply_swaps(base: np.ndarray, proposals: list[tuple], budget: int) -> np.ndarray:
    pred = base.copy()
    for _, _, add, drop, _ in proposals[:budget]:
        pred[add] = 1
        pred[drop] = 0
    return pred


def main() -> None:
    f = pd.read_parquet(FEATURES)
    base_df = pd.read_csv(BASE_PATH)
    if not np.array_equal(f["id"].astype(str).to_numpy(), base_df["id"].astype(str).to_numpy()):
        raise RuntimeError("V22 and compact feature rows are not aligned")
    base = base_df["prediction"].to_numpy(np.int8)
    trend = f["trendyol_qpct"].to_numpy(np.float32)
    lex = f["lex_qpct"].to_numpy(np.float32)
    blend = .5 * trend + .5 * lex
    budgets = pd.Series(base, index=f["term_id"]).groupby(level=0).sum()

    trend_budget = top_by_query(f, trend, budgets)
    blend_budget = top_by_query(f, blend, budgets)
    target_n = int(base.sum())
    global_trend = np.zeros(len(f), dtype=np.int8)
    global_trend[np.argpartition(f["trendyol_score"].to_numpy(), -target_n)[-target_n:]] = 1
    vote = ((base + trend_budget + blend_budget) >= 2).astype(np.int8)
    proposals = consensus_swaps(f, base, trend, lex)

    candidates = {
        "FINAL_CANDIDATE_v71_trendyol_global_same_v22_ratio.csv": global_trend,
        "FINAL_CANDIDATE_v71_trendyol_query_v22_budget_full.csv": trend_budget,
        "FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv": blend_budget,
        "FINAL_CANDIDATE_v71_vote_v22_trend_trendlex_2of3.csv": vote,
        "FINAL_CANDIDATE_v71_v22_consensus_swap25k.csv": apply_swaps(base, proposals, 25_000),
        "FINAL_CANDIDATE_v71_v22_consensus_swap50k.csv": apply_swaps(base, proposals, 50_000),
    }
    rows = [save_candidate(f["id"], pred, name, base, f["term_id"]) for name, pred in candidates.items()]
    proposal_summary = {
        "eligible_consensus_swaps": len(proposals),
        "top_min_margin": float(proposals[0][0]) if proposals else None,
        "margin_at_25k": float(proposals[min(24_999, len(proposals)-1)][0]) if proposals else None,
        "margin_at_50k": float(proposals[min(49_999, len(proposals)-1)][0]) if proposals else None,
    }
    result = {"base": str(BASE_PATH), "base_positive_ratio": float(base.mean()),
              "proposal_summary": proposal_summary, "candidates": rows,
              "warning": "Candidate files are not submissions and have not been evaluated on hidden Kaggle truth."}
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
