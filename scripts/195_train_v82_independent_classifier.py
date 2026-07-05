"""Train/evaluate an anchor-free classifier on disjoint queries, then score test."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(".")
TRAIN = ROOT / "data/processed/v82_v34_features.parquet"
TEST = ROOT / "data/processed/v21_test_features.parquet"
HOLDOUT = ROOT / "models/v76_trendyol_contrastive/holdout_terms.txt"
MODEL = ROOT / "models/v85_independent_relative_history_classifier.cbm"
OUT_SCORE = ROOT / "data/processed/v85_independent_test_scores.parquet"
OUT_REPORT = ROOT / "reports/experiments/v85_independent_relative_history_classifier.json"
OUT_DIR = ROOT / "submissions/final_candidates_v85"

RANK_SIGNALS = ["weighted_overlap", "total_cov", "title_cov", "brand_cov",
                "category_cov", "total_jacc", "loo_item_history",
                "loo_brand_history", "loo_root_history"]


def root_category(x):
    s = "" if pd.isna(x) else str(x)
    for sep in (">", "/", "|"):
        if sep in s:
            return s.split(sep)[0].strip().lower()
    return s.strip().lower()


def add_history_and_safe_ranks(frame, is_train):
    out = frame.copy()
    if is_train:
        items = pd.read_csv(ROOT / "data/raw/items.csv", usecols=lambda c: c in {"item_id", "brand", "category"}, low_memory=False)
        items["item_id"] = items["item_id"].astype(str)
        items["brand_key_v84"] = items.get("brand", "").fillna("").astype(str).str.lower()
        items["root_key_v84"] = items.get("category", "").map(root_category)
        out = out.merge(items[["item_id", "brand_key_v84", "root_key_v84"]], on="item_id", how="left", validate="many_to_one")
        pos = out[out["label"].astype(int) == 1]
        brand_q = pos.groupby(["term_id", "brand_key_v84"]).size().rename("own_brand_pos").reset_index()
        root_q = pos.groupby(["term_id", "root_key_v84"]).size().rename("own_root_pos").reset_index()
        out = out.merge(brand_q, on=["term_id", "brand_key_v84"], how="left")
        out = out.merge(root_q, on=["term_id", "root_key_v84"], how="left")
        out["loo_item_history"] = np.log1p(np.maximum(0, np.expm1(out["item_train_pos_log"].astype(float)) - out["label"].astype(float)))
        out["loo_brand_history"] = np.log1p(np.maximum(0, np.expm1(out["brand_train_pos_log"].astype(float)) - out["own_brand_pos"].fillna(0)))
        out["loo_root_history"] = np.log1p(np.maximum(0, np.expm1(out["root_train_pos_log"].astype(float)) - out["own_root_pos"].fillna(0)))
        out = out.drop(columns=["brand_key_v84", "root_key_v84", "own_brand_pos", "own_root_pos"])
    else:
        out["loo_item_history"] = out["item_train_pos_log"].astype(float)
        out["loo_brand_history"] = out["brand_train_pos_log"].astype(float)
        out["loo_root_history"] = out["root_train_pos_log"].astype(float)

    n = out.groupby("term_id")["term_id"].transform("size").astype(float)
    for c in RANK_SIGNALS:
        rank = out.groupby("term_id")[c].rank(method="average", ascending=False)
        mean = out.groupby("term_id")[c].transform("mean")
        std = out.groupby("term_id")[c].transform("std").fillna(0).clip(lower=1e-6)
        out[f"safe_{c}_pct"] = ((rank - 0.5) / n).astype("float32")
        out[f"safe_{c}_z"] = ((out[c] - mean) / std).astype("float32")
        out[f"safe_{c}_delta_top"] = (out.groupby("term_id")[c].transform("max") - out[c]).astype("float32")
    return out


def pair_metrics(y, p):
    return {
        "macro_f1": float(f1_score(y, p, average="macro")),
        "positive_f1": float(f1_score(y, p, pos_label=1)),
        "negative_f1": float(f1_score(y, p, pos_label=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "pred_ratio": float(p.mean()),
        "true_ratio": float(y.mean()),
    }


def main() -> None:
    d = pd.read_parquet(TRAIN)
    d["term_id"] = d["term_id"].astype(str)
    d["item_id"] = d["item_id"].astype(str)
    d = add_history_and_safe_ranks(d, True)
    hold = set(HOLDOUT.read_text(encoding="utf8").splitlines())
    non_features = {"id", "term_id", "item_id", "label", "fold", "candidate_source"}
    # These columns leak the held-out query's positives or encode the synthetic
    # candidate construction. They are deliberately excluded.
    banned_prefix = ("item_train_pos_", "brand_train_pos_", "root_train_pos_")
    banned_exact = {"candidate_count"}
    # V82 exposed another subtle leak: tied group ranks inherit candidate row
    # order, while count aggregates encode how the synthetic pool was mined.
    # V83 is deliberately pair-local and therefore invariant to pool ordering.
    banned_suffix = ("_rank", "_pct_rank", "_term_max", "_delta_top",
                     "_count_in_term", "_count_ratio")
    features = [c for c in d.columns if c not in non_features and c not in banned_exact
                and not c.startswith(banned_prefix) and not c.endswith(banned_suffix)
                and not c.startswith("loo_")
                and not (c.startswith("safe_loo_") and not c.endswith("_pct"))]

    tr = d[~d["term_id"].isin(hold)].copy()
    va = d[d["term_id"].isin(hold)].copy()
    xtr = tr[features].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
    xva = va[features].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
    ytr = tr["label"].astype("int8").to_numpy()
    yva = va["label"].astype("int8").to_numpy()

    params = dict(
        loss_function="Logloss", eval_metric="AUC", iterations=1400,
        learning_rate=0.04, depth=8, l2_leaf_reg=10, random_strength=0.7,
        bootstrap_type="Bernoulli", subsample=0.85, random_seed=20260704,
        od_type="Iter", od_wait=120, verbose=100, allow_writing_files=False,
        task_type="GPU", devices="0",
    )
    model = CatBoostClassifier(**params)
    try:
        model.fit(xtr, ytr, eval_set=(xva, yva), use_best_model=True)
    except Exception as exc:
        print("GPU fallback:", repr(exc))
        params.update(task_type="CPU", thread_count=max(1, (os.cpu_count() or 2) - 1))
        params.pop("devices", None)
        model = CatBoostClassifier(**params)
        model.fit(xtr, ytr, eval_set=(xva, yva), use_best_model=True)

    sva = model.predict_proba(xva)[:, 1]
    rows = []
    for th in np.linspace(0.05, 0.95, 181):
        p = (sva >= th).astype("int8")
        rows.append({"threshold": float(th), **pair_metrics(yva, p)})
    best = max(rows, key=lambda x: x["macro_f1"])
    report = {
        "anchor_free": True,
        "holdout_queries": int(va["term_id"].nunique()),
        "holdout_rows": int(len(va)),
        "features": features,
        "auc": float(roc_auc_score(yva, sva)),
        "best": best,
        "threshold_grid_top10": sorted(rows, key=lambda x: x["macro_f1"], reverse=True)[:10],
        "best_iteration": int(model.get_best_iteration()),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({k: report[k] for k in ["holdout_queries", "holdout_rows", "auc", "best", "best_iteration"]}, indent=2))

    # Refit on all queries for the chosen number of trees.
    final_iter = max(300, int(model.get_best_iteration() * 1.05))
    final_params = dict(params)
    final_params.update(iterations=final_iter, verbose=100)
    final_params.pop("od_type", None); final_params.pop("od_wait", None)
    final_model = CatBoostClassifier(**final_params)
    xall = d[features].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
    final_model.fit(xall, d["label"].astype("int8"))
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(MODEL))

    test = pd.read_parquet(TEST)
    test["term_id"] = test["term_id"].astype(str)
    test["item_id"] = test["item_id"].astype(str)
    test = add_history_and_safe_ranks(test, False)
    score = np.empty(len(test), dtype="float32")
    for start in range(0, len(test), 400_000):
        end = min(len(test), start + 400_000)
        xt = test.iloc[start:end][features].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
        score[start:end] = final_model.predict_proba(xt)[:, 1]
        print("test", end, "/", len(test))
    pd.DataFrame({"id": test["id"].astype(str), "term_id": test["term_id"].astype(str),
                  "item_id": test["item_id"].astype(str), "v85_score": score}).to_parquet(OUT_SCORE, index=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Threshold robustness band; all are genuinely independent of v22/v71.
    thresholds = sorted(set([round(best["threshold"] + x, 3) for x in (-0.04, -0.02, 0, 0.02, 0.04)]))
    candidates = []
    for th in thresholds:
        pred = (score >= th).astype("int8")
        path = OUT_DIR / f"FINAL_CANDIDATE_v85_independent_thr{str(th).replace('.', 'p')}.csv"
        pd.DataFrame({"id": test["id"].astype(str), "prediction": pred}).to_csv(path, index=False)
        candidates.append({"file": str(path), "threshold": th, "positives": int(pred.sum()), "ratio": float(pred.mean())})
    report["candidates"] = candidates
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()
