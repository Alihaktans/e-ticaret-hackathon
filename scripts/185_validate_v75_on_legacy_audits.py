"""Evaluate v75 candidates on legacy audits not used by pair-BERT training."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(".")
LABELS = {
    "random_blind": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
}
CANDIDATES = {
    "v22_public080": ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
    "v71_public081": ROOT / "submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv",
    "v74_lex65": ROOT / "submissions/final_candidates_v74/FINAL_CANDIDATE_v74_trendlex_35_65_94f52908.csv",
}
for p in sorted((ROOT / "submissions/final_candidates_v75").glob("*.csv")):
    CANDIDATES[p.stem.replace("FINAL_CANDIDATE_v75_", "v75_")] = p
for p in sorted((ROOT / "submissions/final_candidates_v78").glob("*.csv")):
    CANDIDATES[p.stem.replace("FINAL_CANDIDATE_v78_", "v78_")] = p

OUT = ROOT / "reports/experiments/v75_legacy_audit_validation.csv"
REPORT = ROOT / "reports/experiments/v75_legacy_audit_validation.json"


def main():
    pred_maps = {}
    for name, path in CANDIDATES.items():
        d = pd.read_csv(path, dtype={"id": "string"})
        pred_maps[name] = pd.Series(d.prediction.to_numpy(np.int8), index=d.id.astype(str))

    rows = []
    for source, path in LABELS.items():
        lab = pd.read_csv(path, dtype={"id": "string"})
        y = pd.to_numeric(lab.assistant_label, errors="coerce")
        recheck = pd.to_numeric(lab.get("needs_recheck", 0), errors="coerce").fillna(1)
        conf = lab.get("assistant_confidence", "").astype(str).str.lower()
        keep = y.isin([0, 1]) & recheck.eq(0) & conf.isin(["high", "medium"])
        lab = lab.loc[keep].copy(); y = y.loc[keep].astype(np.int8).to_numpy()
        predictions = {name: m.reindex(lab.id.astype(str)).to_numpy() for name, m in pred_maps.items()}
        if any(pd.isna(p).any() for p in predictions.values()):
            raise RuntimeError(f"missing audit ids: {source}")
        base = predictions["v22_public080"].astype(np.int8)
        for name, pred in predictions.items():
            pred = pred.astype(np.int8)
            changed = pred != base
            rows.append({
                "source": source, "variant": name, "n": len(y),
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "accuracy": float((y == pred).mean()), "pred_ratio": float(pred.mean()),
                "changed_vs_v22_n": int(changed.sum()),
                "changed_vs_v22_accuracy": float((y[changed] == pred[changed]).mean()) if changed.any() else np.nan,
                "v22_accuracy_on_changed": float((y[changed] == base[changed]).mean()) if changed.any() else np.nan,
                "net_correct_vs_v22": int((y == pred).sum() - (y == base).sum()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pivot = out.pivot(index="variant", columns="source", values="macro_f1")
    pivot["mean"] = pivot.mean(axis=1); pivot["min"] = pivot.min(axis=1)
    net = out.groupby("variant").net_correct_vs_v22.sum().rename("net_correct_sum")
    summary = pivot.join(net).sort_values(["mean", "min"], ascending=False)
    REPORT.write_text(json.dumps({"labels_are_hidden_truth": False, "summary": summary.reset_index().to_dict("records")}, indent=2), encoding="utf8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
