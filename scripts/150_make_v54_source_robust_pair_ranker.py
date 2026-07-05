from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
LABEL_FEATURES = ROOT / "data/processed/v53_swap_preference_features.parquet"

ANCHORS = {
    "raw_v35": ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    "v40_history2200": ROOT / "submissions/final_candidates_v40/A_v40_raw35_history_impact2200.csv",
}

ROW_SCORES = {
    "v38_lex_score": ROOT / "data/processed/v38_lexical_pair_scores.parquet",
    "v33_ft_score": ROOT / "data/processed/v33_ft_pair_scores.parquet",
    "model_support": ROOT / "data/processed/v48_train_memory_bridge_scores.parquet",
    "v47_mutual_score": ROOT / "data/processed/v47_reciprocal_shadow_scores.parquet",
    "v39_hist_score": ROOT / "data/processed/v39_item_history_pair_scores.parquet",
}

FEATURES = [
    "diff_v38_lex_score",
    "diff_v33_ft_score",
    "diff_model_support",
    "diff_v47_mutual_score",
    "diff_v39_hist_score",
]

OUT_FEATURES = ROOT / "data/processed/v54_source_robust_candidate_pairs.parquet"
OUT_REVIEW = ROOT / "reports/manual_review/review_v54_source_robust_swaps.csv"
OUT_SUMMARY = ROOT / "reports/manual_review/v54_source_robust_candidate_summary.csv"
OUT_SAVED = ROOT / "reports/manual_review/v54_source_robust_saved_candidates.csv"
OUT_AUDIT = ROOT / "reports/manual_review/v54_source_robust_model_audit.json"
SUB_DIR = ROOT / "submissions/final_candidates_v54"


def read_prediction(path: Path, ids: pd.Series) -> np.ndarray:
    d = pd.read_csv(path, usecols=["id", "prediction"])
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(ids.reset_index(drop=True)):
        raise RuntimeError(f"ID order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def add_aligned_score(frame: pd.DataFrame, name: str, path: Path) -> None:
    d = pd.read_parquet(path, columns=["id", name])
    d["id"] = d["id"].astype(str)
    if d["id"].reset_index(drop=True).equals(frame["id"].reset_index(drop=True)):
        frame[name] = pd.to_numeric(d[name], errors="coerce").astype(np.float32).to_numpy()
    else:
        m = frame[["id"]].merge(d, on="id", how="left", validate="one_to_one")
        frame[name] = pd.to_numeric(m[name], errors="coerce").astype(np.float32).to_numpy()
    print(name, "loaded", int(frame[name].notna().sum()), "std", float(frame[name].std()))


def fit_pair_model(labels: pd.DataFrame):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=.10, max_iter=2000, class_weight="balanced", random_state=42),
    )
    x = labels[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = labels["label"].astype(np.int8).to_numpy()
    # Antisymmetric augmentation turns the task into pairwise preference learning.
    xa = pd.concat([x, -x], ignore_index=True)
    ya = np.concatenate([y, 1 - y])
    source = np.concatenate([labels["source"].to_numpy(), labels["source"].to_numpy()])
    counts = pd.Series(source).value_counts()
    weights = np.array([1.0 / counts[s] for s in source], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    model.fit(xa, ya, logisticregression__sample_weight=weights)
    return model


def source_holdout_audit(labels: pd.DataFrame) -> list[dict]:
    rows = []
    for train_source, test_source in [("v29", "v33"), ("v33", "v29")]:
        train = labels[labels.source.eq(train_source)].copy()
        test = labels[labels.source.eq(test_source)].copy()
        heldout_ids = set(test.id_add.astype(str)) | set(test.id_drop.astype(str))
        train = train[~train.id_add.astype(str).isin(heldout_ids) & ~train.id_drop.astype(str).isin(heldout_ids)]
        model = fit_pair_model(train)
        prob = model.predict_proba(test[FEATURES])[:, 1]
        row = {
            "train_source": train_source,
            "test_source": test_source,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "auc": float(roc_auc_score(test.label, prob)),
            "ap": float(average_precision_score(test.label, prob)),
        }
        for threshold in [.80, .85, .90, .95]:
            take = prob >= threshold
            row[f"n_at_{threshold:.2f}"] = int(take.sum())
            row[f"precision_at_{threshold:.2f}"] = float(test.loc[take, "label"].mean()) if take.any() else None
        rows.append(row)
    return rows


def build_rows(sample: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    for c in ["id", "term_id", "item_id"]:
        pairs[c] = pairs[c].astype(str)
    if not pairs.id.reset_index(drop=True).equals(sample.id.reset_index(drop=True)):
        raise RuntimeError("sample/pairs ID mismatch")
    for name, path in ROW_SCORES.items():
        add_aligned_score(pairs, name, path)
    for name in ROW_SCORES:
        r = pairs.groupby("term_id", sort=False)[name].rank(method="average", pct=True)
        pairs[name + "_pct"] = r.fillna(.5).astype(np.float32)
    pairs["robust_row_score"] = (
        .40 * pairs["v38_lex_score_pct"]
        + .25 * pairs["v33_ft_score_pct"]
        + .15 * pairs["model_support_pct"]
        + .10 * pairs["v47_mutual_score_pct"]
        + .10 * pairs["v39_hist_score_pct"]
    ).astype(np.float32)
    return pairs


def build_candidate_pairs(rows: pd.DataFrame, anchor: np.ndarray, model, anchor_name: str) -> pd.DataFrame:
    d = rows.copy()
    d["anchor"] = anchor
    adds = d[d.anchor.eq(0)].sort_values(["term_id", "robust_row_score"], ascending=[True, False]).groupby("term_id", sort=False).head(3)
    drops = d[d.anchor.eq(1)].sort_values(["term_id", "robust_row_score"], ascending=[True, True]).groupby("term_id", sort=False).head(3)
    add_cols = ["term_id", "id", "item_id", "robust_row_score"] + list(ROW_SCORES) + [x + "_pct" for x in ROW_SCORES]
    drop_cols = add_cols.copy()
    a = adds[add_cols].rename(columns={c: c + "_add" for c in add_cols if c != "term_id"})
    b = drops[drop_cols].rename(columns={c: c + "_drop" for c in drop_cols if c != "term_id"})
    pairs = a.merge(b, on="term_id", how="inner")
    for name in ROW_SCORES:
        pairs["diff_" + name] = pairs[name + "_add"] - pairs[name + "_drop"]
        pairs["pct_diff_" + name] = pairs[name + "_pct_add"] - pairs[name + "_pct_drop"]
    pairs["robust_gain"] = pairs.robust_row_score_add - pairs.robust_row_score_drop
    pairs["signal_win_count"] = sum((pairs["diff_" + n] > 0).astype(np.int8) for n in ROW_SCORES)
    pairs["rank_win_count"] = sum((pairs["pct_diff_" + n] > 0).astype(np.int8) for n in ROW_SCORES)
    pairs["pair_prob"] = model.predict_proba(pairs[FEATURES])[:, 1].astype(np.float32)
    pairs["anchor_name"] = anchor_name
    pairs = pairs.sort_values(["pair_prob", "robust_gain"], ascending=False)
    pairs = pairs.drop_duplicates("id_add").drop_duplicates("id_drop")
    pairs = pairs.drop_duplicates("term_id")
    return pairs.reset_index(drop=True)


def accepted(pool: pd.DataFrame, mode: str) -> pd.DataFrame:
    common = (
        (pool.robust_gain >= .12)
        & (pool.diff_v38_lex_score > 0)
        & (pool.signal_win_count >= 3)
        & (pool.rank_win_count >= 4)
    )
    if mode == "ultra":
        keep = common & (pool.pair_prob >= .95) & (pool.diff_v33_ft_score > 0) & (pool.diff_model_support >= 0) & (pool.signal_win_count >= 4) & (pool.robust_gain >= .18)
    elif mode == "safe":
        keep = common & (pool.pair_prob >= .90) & (pool.diff_v33_ft_score > 0) & (pool.robust_gain >= .15)
    elif mode == "balanced":
        keep = common & (pool.pair_prob >= .85)
    else:
        raise ValueError(mode)
    return pool[keep].sort_values(["pair_prob", "robust_gain"], ascending=False).reset_index(drop=True)


def enrich_review(review: pd.DataFrame) -> pd.DataFrame:
    terms = pd.read_csv(TERMS)
    if "query" not in terms.columns:
        terms = terms.rename(columns={[c for c in terms.columns if c != "term_id"][0]: "query"})
    terms["term_id"] = terms.term_id.astype(str)
    review = review.merge(terms[["term_id", "query"]], on="term_id", how="left")
    items = pd.read_csv(ITEMS)
    for c in items.columns:
        if c.lower() in ["item_id", "product_id"]:
            item_col = c
            break
    else:
        item_col = items.columns[0]
    items[item_col] = items[item_col].astype(str)
    title = next((c for c in items.columns if c.lower() in ["title", "name", "product_name"]), None)
    brand = next((c for c in items.columns if "brand" in c.lower() or "marka" in c.lower()), None)
    category = next((c for c in items.columns if "category" in c.lower() or "kategori" in c.lower()), None)
    cols = [item_col] + [c for c in [title, brand, category] if c]
    meta = items[cols].drop_duplicates(item_col)
    ren_add = {item_col: "item_id_add", title: "title_add", brand: "brand_add", category: "category_add"}
    ren_drop = {item_col: "item_id_drop", title: "title_drop", brand: "brand_drop", category: "category_drop"}
    ren_add = {k: v for k, v in ren_add.items() if k is not None}
    ren_drop = {k: v for k, v in ren_drop.items() if k is not None}
    review = review.merge(meta.rename(columns=ren_add), on="item_id_add", how="left")
    review = review.merge(meta.rename(columns=ren_drop), on="item_id_drop", how="left")
    front = ["anchor_name", "mode", "term_id", "query", "id_add", "item_id_add", "title_add", "brand_add", "category_add", "id_drop", "item_id_drop", "title_drop", "brand_drop", "category_drop", "pair_prob", "robust_gain", "signal_win_count", "rank_win_count"]
    return review[[c for c in front if c in review.columns] + [c for c in review.columns if c not in front]]


def main():
    for p in [OUT_REVIEW.parent, OUT_FEATURES.parent, SUB_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample.id.astype(str)
    labels = pd.read_parquet(LABEL_FEATURES)
    labels = labels[labels.source.isin(["v29", "v33"]) & labels.label.isin([0, 1])].copy()
    audit = source_holdout_audit(labels)
    print("source holdout audit", json.dumps(audit, indent=2))
    model = fit_pair_model(labels)
    rows = build_rows(sample)

    summaries, reviews, saved = [], [], []
    all_pools = []
    caps = [100, 250, 500, 1000]
    for anchor_name, anchor_path in ANCHORS.items():
        if not anchor_path.exists():
            continue
        anchor = read_prediction(anchor_path, sample.id)
        pool = build_candidate_pairs(rows, anchor, model, anchor_name)
        all_pools.append(pool)
        print(anchor_name, "raw pool", len(pool), "p95", int((pool.pair_prob >= .95).sum()))
        for mode in ["ultra", "safe", "balanced"]:
            swaps = accepted(pool, mode)
            print(anchor_name, mode, "accepted", len(swaps))
            if len(swaps):
                x = swaps.head(100).copy()
                x["mode"] = mode
                reviews.append(x)
            for cap in caps:
                take = swaps.head(cap)
                if take.empty:
                    continue
                pred = anchor.copy()
                index = pd.Series(np.arange(len(sample)), index=sample.id)
                pred[index.loc[take.id_add].to_numpy()] = 1
                pred[index.loc[take.id_drop].to_numpy()] = 0
                variant = f"v54_{anchor_name}_{mode}_cap{cap}"
                out = SUB_DIR / f"{variant}.csv"
                pd.DataFrame({"id": sample.id, "prediction": pred}).to_csv(out, index=False)
                row = {
                    "variant": variant, "anchor": anchor_name, "mode": mode, "cap": cap,
                    "accepted_pool": int(len(swaps)), "used_swaps": int(len(take)),
                    "pair_prob_mean": float(take.pair_prob.mean()), "pair_prob_min": float(take.pair_prob.min()),
                    "robust_gain_mean": float(take.robust_gain.mean()), "signal_win_mean": float(take.signal_win_count.mean()),
                    "ones": int(pred.sum()), "pos_ratio": float(pred.mean()), "diff_vs_anchor": int((pred != anchor).sum()),
                    "file": str(out),
                }
                summaries.append(row)
                saved.append(row)

    if all_pools:
        pd.concat(all_pools, ignore_index=True).to_parquet(OUT_FEATURES, index=False)
    if reviews:
        review = pd.concat(reviews, ignore_index=True).drop_duplicates(["anchor_name", "id_add", "id_drop"])
        enrich_review(review).to_csv(OUT_REVIEW, index=False)
    pd.DataFrame(summaries).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)
    OUT_AUDIT.write_text(json.dumps({"features": FEATURES, "source_holdout": audit, "warning": "Assistant labels are useful preference supervision but may retain shared heuristic bias."}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("outputs", OUT_REVIEW, OUT_SUMMARY, OUT_FEATURES, OUT_AUDIT)


if __name__ == "__main__":
    main()
