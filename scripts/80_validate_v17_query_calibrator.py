from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

FEATURES = ROOT / "data/processed/v17_query_level_features.parquet"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
}

OUT_GRID = ROOT / "reports/manual_review/v17_query_calibrator_loso_grid.csv"
OUT_AGG = ROOT / "reports/manual_review/v17_query_calibrator_loso_aggregate.csv"

FEATURE_COLS = [
    "proba_avg", "cross_sigmoid", "bge_sigmoid", "bge_blend_w020", "bge_blend_w030",
    "bge_rank_score", "bge_blend_score", "ce_rank_score",
    "candidate_count",
    "query_token_count", "query_char_len",
    "query_has_brand", "query_has_color", "query_has_gender", "query_has_digit",
    "query_is_generic", "query_is_specific",
    "v5_high", "v5_mid", "has_bge", "has_ce",
    "pred_v13", "pred_v15", "pred_v16",
    "pred_v13_term_sum", "pred_v13_term_ratio",
    "pred_v15_term_sum", "pred_v15_term_ratio",
    "pred_v16_term_sum", "pred_v16_term_ratio",
    "proba_avg_rank", "proba_avg_pct_rank", "proba_avg_term_max", "proba_avg_delta_top",
    "bge_rank_score_rank", "bge_rank_score_pct_rank", "bge_rank_score_term_max", "bge_rank_score_delta_top",
    "bge_blend_score_rank", "bge_blend_score_pct_rank", "bge_blend_score_term_max", "bge_blend_score_delta_top",
    "ce_rank_score_rank", "ce_rank_score_pct_rank", "ce_rank_score_term_max", "ce_rank_score_delta_top",
]

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1]),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def sample_weight_from_labels(lab):
    w = np.ones(len(lab), dtype=np.float32)

    if "needs_recheck" in lab.columns:
        try:
            nr = lab["needs_recheck"].fillna(0).astype(int).to_numpy()
            w[nr == 1] *= 0.35
        except Exception:
            pass

    if "assistant_confidence" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        w[conf.eq("high").to_numpy()] *= 1.20
        w[conf.eq("medium").to_numpy()] *= 1.00
        w[conf.eq("low").to_numpy()] *= 0.50

    return w

print("loading features...")
feat = pd.read_parquet(FEATURES)
feat["id"] = feat["id"].astype(str)

print("loading labels...")
labs = []
for name, path in LABEL_FILES.items():
    if not path.exists():
        print("missing:", path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)
    lab["label_set"] = name
    lab["sample_weight"] = sample_weight_from_labels(lab)
    labs.append(lab[["id", "assistant_label", "label_set", "sample_weight"]])

labels = pd.concat(labs, ignore_index=True)

# Aynı id birden fazla sette varsa çoğunluk label, weight ortalaması.
labels = (
    labels.groupby(["id", "label_set"], as_index=False)
    .agg(
        assistant_label=("assistant_label", lambda x: int(round(x.mean()))),
        sample_weight=("sample_weight", "mean"),
    )
)

data = labels.merge(feat, on="id", how="inner", validate="many_to_one")
print("labeled rows:", len(data))
print(data["label_set"].value_counts().to_string())

X_all = data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype("float32")
y_all = data["assistant_label"].astype(int).to_numpy()

thresholds = np.round(np.arange(0.10, 0.91, 0.025), 3)

rows = []

for holdout in sorted(data["label_set"].unique()):
    train_idx = data["label_set"] != holdout
    test_idx = data["label_set"] == holdout

    if train_idx.sum() < 100 or test_idx.sum() < 30:
        continue

    X_train = X_all.loc[train_idx]
    y_train = y_all[train_idx.to_numpy()]
    w_train = data.loc[train_idx, "sample_weight"].to_numpy()

    X_test = X_all.loc[test_idx]
    y_test = y_all[test_idx.to_numpy()]

    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        min_samples_leaf=20,
        random_state=2026,
    )

    model.fit(X_train, y_train, sample_weight=w_train)
    proba = model.predict_proba(X_test)[:, 1]

    # calibrator thresholds
    for th in thresholds:
        pred = (proba >= th).astype(np.int8)
        row = {
            "holdout": holdout,
            "method": "v17_hgb_calibrator",
            "threshold": float(th),
            "n": int(test_idx.sum()),
        }
        row.update(metrics(y_test, pred))
        rows.append(row)

    # baselines
    for baseline_col in ["pred_v13", "pred_v15", "pred_v16"]:
        if baseline_col in data.columns:
            pred = data.loc[test_idx, baseline_col].astype(int).to_numpy()
            ok = pred >= 0
            if ok.sum() > 10:
                row = {
                    "holdout": holdout,
                    "method": baseline_col,
                    "threshold": -1.0,
                    "n": int(ok.sum()),
                }
                row.update(metrics(y_test[ok], pred[ok]))
                rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(OUT_GRID, index=False)

agg = (
    res.groupby(["method", "threshold"])
    .agg(
        mean_macro=("macro_f1", "mean"),
        min_macro=("macro_f1", "min"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_pred_pos_ratio=("pred_pos_ratio", "mean"),
        sets=("holdout", "nunique"),
    )
    .reset_index()
    .sort_values(["min_macro", "mean_macro"], ascending=False)
)

agg.to_csv(OUT_AGG, index=False)

print("\nTOP PER HOLDOUT")
print(res.sort_values(["holdout", "macro_f1"], ascending=[True, False]).groupby("holdout").head(12).to_string(index=False))

print("\nAGG TOP")
print(agg.head(40).to_string(index=False))

print("saved:", OUT_GRID)
print("saved:", OUT_AGG)
