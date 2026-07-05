"""Stress-test v85 on fixed-100 candidate pools built from untouched queries."""
from pathlib import Path
import json
import runpy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(".")
V85 = runpy.run_path(str(ROOT / "scripts/195_train_v82_independent_classifier.py"))
OUT = ROOT / "reports/experiments/v85_fixed100_validation.json"


def metrics(y, s):
    rows = []
    for th in np.linspace(0.05, 0.95, 181):
        p = (s >= th).astype(np.int8)
        rows.append({"threshold": float(th), "macro_f1": float(f1_score(y, p, average="macro")),
                     "pred_ratio": float(p.mean())})
    return {"auc": float(roc_auc_score(y, s)), "best": max(rows, key=lambda x: x["macro_f1"]),
            "at_v85_threshold": next(x for x in rows if abs(x["threshold"] - 0.505) < 1e-8)}


def main():
    hold = set(V85["HOLDOUT"].read_text(encoding="utf8").splitlines())
    train = pd.read_parquet(V85["TRAIN"])
    train["term_id"] = train["term_id"].astype(str); train["item_id"] = train["item_id"].astype(str)
    train = V85["add_history_and_safe_ranks"](train, True)
    report = json.loads(V85["OUT_REPORT"].read_text(encoding="utf8"))
    features = report["features"]
    tr = train[~train.term_id.isin(hold)]
    xtr = tr[features].replace([np.inf, -np.inf], np.nan).fillna(-1).astype(np.float32)
    model = CatBoostClassifier(loss_function="Logloss", iterations=1400, learning_rate=.04, depth=8,
        l2_leaf_reg=10, random_strength=.7, bootstrap_type="Bernoulli", subsample=.85,
        random_seed=20260704, verbose=200, allow_writing_files=False, task_type="GPU", devices="0")
    model.fit(xtr, tr.label.astype(np.int8))

    pool = pd.read_parquet(ROOT / "data/processed/v21_train_features.parquet")
    pool["term_id"] = pool["term_id"].astype(str); pool["item_id"] = pool["item_id"].astype(str)
    pool = pool[pool.term_id.isin(hold)].copy()
    selected = []
    for _, g in pool.groupby("term_id", sort=False):
        pos = g[g.label.astype(int) == 1]
        if len(pos) >= 100:
            continue
        neg = g[g.label.astype(int) == 0].sort_values(
            ["weighted_overlap", "total_cov", "title_cov"], ascending=False).head(100 - len(pos))
        if len(pos) + len(neg) == 100:
            selected.append(pd.concat([pos, neg], ignore_index=True))
    fixed = pd.concat(selected, ignore_index=True)
    # Drop stale group features. add_history_and_safe_ranks recomputes only the
    # safe v85 ranks used by the model.
    fixed = V85["add_history_and_safe_ranks"](fixed, True)
    x = fixed[features].replace([np.inf, -np.inf], np.nan).fillna(-1).astype(np.float32)
    s = model.predict_proba(x)[:, 1]
    result = {"queries": int(fixed.term_id.nunique()), "rows": int(len(fixed)),
              "true_ratio": float(fixed.label.mean()), **metrics(fixed.label.to_numpy(np.int8), s)}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
