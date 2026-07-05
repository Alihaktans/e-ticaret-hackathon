"""Filter v71 swaps with targeted intent-error model and create audits."""
from pathlib import Path
import hashlib
import json
import runpy

import numpy as np
import pandas as pd


ROOT = Path(".")
BASE = ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv"
V71 = ROOT / "submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv"
FEATURES = ROOT / "data/processed/v72_anchorless_trendlex_features.parquet"
CE = ROOT / "data/processed/v75_boundary_pairbert_scores.parquet"
OUT_DIR = ROOT / "submissions/final_candidates_v89"
REPORT = ROOT / "reports/experiments/v89_intent_filtered_report.json"
AUDIT = ROOT / "reports/manual_review/v89_intent_veto_audit.csv"


def main():
    helper = runpy.run_path(str(ROOT / "scripts/184_make_v75_pairbert_filtered_v71.py"))
    f = pd.read_parquet(FEATURES)
    f["id"] = f["id"].astype(str); f["term_id"] = f["term_id"].astype(str)
    f["trendlex_score"] = .5 * f["trendyol_qpct"].astype(np.float32) + .5 * f["lex_qpct"].astype(np.float32)
    bdf = pd.read_csv(BASE); vdf = pd.read_csv(V71)
    bdf["id"] = bdf["id"].astype(str); vdf["id"] = vdf["id"].astype(str)
    if not f.id.equals(bdf.id) or not f.id.equals(vdf.id): raise RuntimeError("alignment")
    base = bdf.prediction.to_numpy(np.int8); v71 = vdf.prediction.to_numpy(np.int8)

    ce = pd.read_parquet(CE)
    ce["id"] = ce["id"].astype(str)
    score_cols = [c for c in ["v75_v29hold_score", "v75_v33hold_score", "v88_intent_score", "v90_intent_score"] if c in ce]
    f = f.merge(ce[["id"] + score_cols], on="id", how="left", validate="one_to_one")
    pairs = helper["build_pairs"](f, base, v71)
    for col in score_cols:
        s = f[col].to_numpy(np.float32)
        pairs["margin_" + col] = s[pairs.add_row] - s[pairs.drop_row]
    tl = f.trendlex_score.to_numpy(np.float32)
    pairs["margin_trendlex"] = tl[pairs.add_row] - tl[pairs.drop_row]
    m = pairs["margin_v88_intent_score"]
    m2 = pairs.get("margin_v90_intent_score")
    old_both = (pairs.get("margin_v75_v29hold_score", 1) > 0) & (pairs.get("margin_v75_v33hold_score", 1) > 0)
    masks = {
        "intent_gt0": m > 0,
        "intent_gt0p5": m > .5,
        "intent_gt1": m > 1,
        "intent_gt2": m > 2,
        "intent_gt3": m > 3,
        "intent_gt0_oldboth": (m > 0) & old_both,
        "intent_gt1_oldboth": (m > 1) & old_both,
    }
    if m2 is not None:
        masks.update({
            "dual_intent_gt0": (m > 0) & (m2 > 0),
            "dual_intent_gt0p5": (m > .5) & (m2 > .5),
            "dual_intent_gt1": (m > 1) & (m2 > 1),
            "dual_intent_gt2": (m > 2) & (m2 > 2),
            "dual_intent_gt0_oldboth": (m > 0) & (m2 > 0) & old_both,
            # Conservative production guards: keep the public-best v71 swap
            # unless both independently initialized intent models strongly veto it.
            "v71_dual_veto_below0": ~((m < 0) & (m2 < 0)),
            "v71_dual_veto_below_m1": ~((m < -1) & (m2 < -1)),
            "v71_dual_veto_below_m2": ~((m < -2) & (m2 < -2)),
            "v71_dual_veto_below_m3": ~((m < -3) & (m2 < -3)),
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True); REPORT.parent.mkdir(parents=True, exist_ok=True); AUDIT.parent.mkdir(parents=True, exist_ok=True)
    budget = pd.Series(base, index=f.term_id).groupby(level=0).sum()
    rows = []
    for name, mask in masks.items():
        pred = base.copy(); a = pairs.loc[mask]
        pred[a.add_row.to_numpy()] = 1; pred[a.drop_row.to_numpy()] = 0
        got = pd.Series(pred, index=f.term_id).groupby(level=0).sum()
        if not got.equals(budget): raise RuntimeError("quota invariant " + name)
        path = OUT_DIR / f"FINAL_CANDIDATE_v89_{name}.csv"
        pd.DataFrame({"id": f.id, "prediction": pred}).to_csv(path, index=False)
        rows.append({"variant": name, "file": str(path), "accepted_swaps": int(mask.sum()),
                     "changed_vs_v22": int((pred != base).sum()), "changed_vs_v71": int((pred != v71).sum()),
                     "mean_intent_margin": float(m[mask].mean()) if mask.any() else None,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    # Human-readable strongest veto/support pairs.
    boundary = ce.set_index("id")
    def enrich(part, bucket):
        out = part.copy(); out["bucket"] = bucket
        for side in ["add", "drop"]:
            ids = f.id.iloc[out[f"{side}_row"].to_numpy()].to_numpy()
            out[f"id_{side}"] = ids
            out[f"text_{side}"] = [boundary.at[i, "item_text"] for i in ids]
        out["query"] = [boundary.at[i, "query_text"] for i in out.id_add]
        return out
    audit = pd.concat([enrich(pairs.nsmallest(150, "margin_v88_intent_score"), "strong_veto"),
                       enrich(pairs.nlargest(150, "margin_v88_intent_score"), "strong_support")], ignore_index=True)
    audit.to_csv(AUDIT, index=False)
    report = {"model": "v88 targeted intent BERTurk", "proposed_swaps": len(pairs),
              "intent_support_rate": float((m > 0).mean()), "intent_margin_summary": m.describe().to_dict(),
              "oldboth_rate": float(old_both.mean()),
              "dual_margin_correlation": float(np.corrcoef(m, m2)[0, 1]) if m2 is not None else None,
              "dual_support_rate": float(((m > 0) & (m2 > 0)).mean()) if m2 is not None else None,
              "candidates": rows,
              "target_error_families": ["category_bait", "brand_mismatch", "numeric_mismatch", "gender_mismatch", "head_noun_mismatch"]}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
