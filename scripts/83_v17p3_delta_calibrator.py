from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

FEATURES = ROOT / "data/processed/v17_query_level_features.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
}

OUT_GRID = ROOT / "reports/manual_review/v17p3_delta_calibrator_grid.csv"
OUT_AGG = ROOT / "reports/manual_review/v17p3_delta_calibrator_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v17p3_delta_calibrator_full_summary.csv"

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

BASES = ["pred_v13", "pred_v16"]
MODES = ["towards_v15", "towards_v16", "any_disagreement"]
ADD_THS = [0.60, 0.65, 0.70, 0.75, 0.80]
REMOVE_THS = [0.40, 0.35, 0.30, 0.25, 0.20]
BUDGET_RATIOS = [0.0025, 0.005, 0.0075, 0.010, 0.015]

def clean_name(x):
    return (
        x.replace(".", "p")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
    )

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

def make_delta_pred(base_pred, p1, pred_v13, pred_v15, pred_v16, mode, add_th, remove_th, budget_ratio):
    pred = base_pred.copy()
    n = len(pred)
    budget = max(1, int(round(n * budget_ratio)))

    if mode == "towards_v15":
        add_cand = (base_pred == 0) & (pred_v15 == 1) & (p1 >= add_th)
        rem_cand = (base_pred == 1) & (pred_v15 == 0) & (p1 <= remove_th)
    elif mode == "towards_v16":
        add_cand = (base_pred == 0) & (pred_v16 == 1) & (p1 >= add_th)
        rem_cand = (base_pred == 1) & (pred_v16 == 0) & (p1 <= remove_th)
    else:
        disagree = (pred_v13 != pred_v15) | (pred_v13 != pred_v16) | (pred_v15 != pred_v16)
        add_cand = (base_pred == 0) & disagree & (p1 >= add_th)
        rem_cand = (base_pred == 1) & disagree & (p1 <= remove_th)

    add_idx = np.flatnonzero(add_cand)
    rem_idx = np.flatnonzero(rem_cand)

    changes = []

    if len(add_idx):
        add_margin = p1[add_idx] - add_th
        for i, m in zip(add_idx, add_margin):
            changes.append((float(m), int(i), 1))

    if len(rem_idx):
        rem_margin = remove_th - p1[rem_idx]
        for i, m in zip(rem_idx, rem_margin):
            changes.append((float(m), int(i), 0))

    if not changes:
        return pred

    changes.sort(reverse=True, key=lambda x: x[0])
    for _, i, val in changes[:budget]:
        pred[i] = val

    return pred

print("loading features...")
feat = pd.read_parquet(FEATURES)
feat["id"] = feat["id"].astype(str)

sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
assert feat["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))

ids = feat["id"].to_numpy()

print("loading labels...")
labs = []
for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing:", path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)
    lab["label_set"] = label_set
    lab["sample_weight"] = sample_weight_from_labels(lab)
    labs.append(lab[["id", "assistant_label", "label_set", "sample_weight"]])

labels = pd.concat(labs, ignore_index=True)
data = labels.merge(feat, on="id", how="inner", validate="many_to_one")

print("labeled rows:", len(data))
print(data["label_set"].value_counts().to_string())

X_all = data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype("float32")
y_all = data["assistant_label"].astype(int).to_numpy()

rows = []

for holdout in sorted(data["label_set"].unique()):
    train_mask = data["label_set"] != holdout
    test_mask = data["label_set"] == holdout

    if train_mask.sum() < 100 or test_mask.sum() < 30:
        continue

    X_train = X_all.loc[train_mask]
    y_train = y_all[train_mask.to_numpy()]
    w_train = data.loc[train_mask, "sample_weight"].to_numpy()

    X_test = X_all.loc[test_mask]
    y_test = y_all[test_mask.to_numpy()]

    model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.20,
        min_samples_leaf=25,
        random_state=2027,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    p1 = model.predict_proba(X_test)[:, 1]

    part = data.loc[test_mask].copy()

    pred_v13 = part["pred_v13"].astype(int).to_numpy()
    pred_v15 = part["pred_v15"].astype(int).to_numpy()
    pred_v16 = part["pred_v16"].astype(int).to_numpy()

    # baselines
    for baseline_name, base_pred in [
        ("pred_v13", pred_v13),
        ("pred_v15", pred_v15),
        ("pred_v16", pred_v16),
    ]:
        row = {
            "holdout": holdout,
            "variant": baseline_name,
            "base": baseline_name,
            "mode": "baseline",
            "add_th": -1,
            "remove_th": -1,
            "budget_ratio": 0,
            "n": len(part),
        }
        row.update(metrics(y_test, base_pred))
        rows.append(row)

    for base_col in BASES:
        base_pred = part[base_col].astype(int).to_numpy()

        for mode in MODES:
            for add_th in ADD_THS:
                for remove_th in REMOVE_THS:
                    if remove_th >= add_th:
                        continue

                    for budget_ratio in BUDGET_RATIOS:
                        pred = make_delta_pred(
                            base_pred=base_pred,
                            p1=p1,
                            pred_v13=pred_v13,
                            pred_v15=pred_v15,
                            pred_v16=pred_v16,
                            mode=mode,
                            add_th=add_th,
                            remove_th=remove_th,
                            budget_ratio=budget_ratio,
                        )

                        variant = f"v17p3_{base_col}_{mode}_add{add_th}_rem{remove_th}_bud{budget_ratio}"
                        row = {
                            "holdout": holdout,
                            "variant": variant,
                            "base": base_col,
                            "mode": mode,
                            "add_th": add_th,
                            "remove_th": remove_th,
                            "budget_ratio": budget_ratio,
                            "n": len(part),
                            "changes_vs_base": int((pred != base_pred).sum()),
                        }
                        row.update(metrics(y_test, pred))
                        rows.append(row)

grid = pd.DataFrame(rows)
grid.to_csv(OUT_GRID, index=False)

weights = {
    "random_clean_v2": 0.30,
    "manual_v1": 0.35,
    "review_v15_vs_v13": 0.25,
    "review_v13_vs_v5": 0.10,
}

agg_rows = []
for variant, g in grid.groupby("variant"):
    score = 0
    wsum = 0
    vals = []

    for _, r in g.iterrows():
        w = weights.get(r["holdout"], 0)
        if w > 0:
            score += w * r["macro_f1"]
            wsum += w
            vals.append(r["macro_f1"])

    if wsum == 0:
        continue

    agg_rows.append({
        "variant": variant,
        "weighted_macro": score / wsum,
        "min_macro": float(np.min(vals)),
        "mean_macro": float(np.mean(vals)),
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "mean_pred_pos_ratio": float(g["pred_pos_ratio"].mean()),
        "mean_changes_vs_base": float(g.get("changes_vs_base", pd.Series([0])).fillna(0).mean()),
        "sets": int(g["holdout"].nunique()),
    })

agg = pd.DataFrame(agg_rows).sort_values(
    ["weighted_macro", "min_macro", "mean_macro"],
    ascending=False,
)
agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP")
print(agg.head(50).to_string(index=False))

# Train final model on all labels
print("\ntraining final model...")
X_label = feat.loc[feat["id"].isin(data["id"]), FEATURE_COLS]
# daha sağlam: data sırasındaki X_all/y_all ile fit
final_model = HistGradientBoostingClassifier(
    max_iter=220,
    learning_rate=0.04,
    max_leaf_nodes=15,
    l2_regularization=0.20,
    min_samples_leaf=25,
    random_state=2027,
)
final_model.fit(X_all, y_all, sample_weight=data["sample_weight"].to_numpy())

X_full = feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype("float32")
p_full = final_model.predict_proba(X_full)[:, 1]

pred_v13_full = feat["pred_v13"].astype(int).to_numpy()
pred_v15_full = feat["pred_v15"].astype(int).to_numpy()
pred_v16_full = feat["pred_v16"].astype(int).to_numpy()

full_rows = []

top_variants = agg.head(10)["variant"].tolist()
for baseline in ["pred_v13", "pred_v15", "pred_v16"]:
    if baseline not in top_variants:
        top_variants.append(baseline)

for variant in top_variants:
    if variant == "pred_v13":
        pred = pred_v13_full.copy()
    elif variant == "pred_v15":
        pred = pred_v15_full.copy()
    elif variant == "pred_v16":
        pred = pred_v16_full.copy()
    else:
        # parse
        m = re.match(r"v17p3_(pred_v13|pred_v16)_(.+)_add([0-9.]+)_rem([0-9.]+)_bud([0-9.]+)", variant)
        if not m:
            print("cannot parse:", variant)
            continue

        base_col = m.group(1)
        mode = m.group(2)
        add_th = float(m.group(3))
        remove_th = float(m.group(4))
        budget_ratio = float(m.group(5))

        base_pred = pred_v13_full.copy() if base_col == "pred_v13" else pred_v16_full.copy()

        pred = make_delta_pred(
            base_pred=base_pred,
            p1=p_full,
            pred_v13=pred_v13_full,
            pred_v15=pred_v15_full,
            pred_v16=pred_v16_full,
            mode=mode,
            add_th=add_th,
            remove_th=remove_th,
            budget_ratio=budget_ratio,
        )

    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v17p3_{clean_name(variant)}.csv"
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)

    full_rows.append({
        "variant": variant,
        "file": str(out),
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v13": int((pred != pred_v13_full).sum()),
        "diff_vs_v13_ratio": float((pred != pred_v13_full).mean()),
        "diff_vs_v16": int((pred != pred_v16_full).sum()),
        "diff_vs_v16_ratio": float((pred != pred_v16_full).mean()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

print("\nFULL SUMMARY")
print(full.to_string(index=False))

print("saved:", OUT_GRID)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
