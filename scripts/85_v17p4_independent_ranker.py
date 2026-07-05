from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

BASE_FEATURES = ROOT / "data/processed/v17_query_level_features.parquet"
LEX_FEATURES = ROOT / "data/processed/v17p4_independent_lexical_features.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
}

OUT_GRID = ROOT / "reports/manual_review/v17p4_independent_grid.csv"
OUT_AGG = ROOT / "reports/manual_review/v17p4_independent_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v17p4_independent_full_summary.csv"
OUT_PROBA = ROOT / "data/processed/v17p4_independent_full_proba.parquet"

BASE_FEATURE_COLS = [
    "proba_avg", "cross_sigmoid", "bge_sigmoid", "bge_blend_w020", "bge_blend_w030",
    "bge_rank_score", "bge_blend_score", "ce_rank_score",
    "candidate_count",
    "query_token_count", "query_char_len",
    "query_has_brand", "query_has_color", "query_has_gender", "query_has_digit",
    "query_is_generic", "query_is_specific",
    "v5_high", "v5_mid", "has_bge", "has_ce",
    "proba_avg_rank", "proba_avg_pct_rank", "proba_avg_term_max", "proba_avg_delta_top",
    "bge_rank_score_rank", "bge_rank_score_pct_rank", "bge_rank_score_term_max", "bge_rank_score_delta_top",
    "bge_blend_score_rank", "bge_blend_score_pct_rank", "bge_blend_score_term_max", "bge_blend_score_delta_top",
    "ce_rank_score_rank", "ce_rank_score_pct_rank", "ce_rank_score_term_max", "ce_rank_score_delta_top",
]

LEX_FEATURE_COLS = [
    "title_overlap", "cat_overlap",
    "title_cov", "cat_cov",
    "title_jacc", "cat_jacc",
    "brand_in_query", "brand_match", "brand_mismatch",
    "gender_match", "gender_mismatch",
    "item_title_len", "item_category_len", "item_brand_empty",
]

FEATURE_COLS = BASE_FEATURE_COLS + LEX_FEATURE_COLS

def clean_name(x):
    return (
        x.replace(".", "p")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace("+", "plus")
        .replace("-", "m")
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

def make_configs():
    configs = []

    # Düz threshold
    for th in np.round(np.arange(0.15, 0.86, 0.025), 3):
        configs.append({
            "name": f"raw_t{th}",
            "base_th": float(th),
            "brand_adj": 0.0,
            "specific_adj": 0.0,
            "generic_adj": 0.0,
            "cap_brand": 9999,
            "cap_spec": 9999,
            "cap_generic": 9999,
            "rescue": 0,
        })

    # Query tipine göre threshold
    caps = [
        ("nocap", 9999, 9999, 9999),
        ("mildcap", 100, 150, 9999),
        ("midcap", 60, 100, 9999),
        ("brandcap", 35, 9999, 9999),
    ]

    for base in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for brand_adj in [0.0, 0.04, 0.08, 0.12]:
            for specific_adj in [0.0, 0.03, 0.06]:
                for generic_adj in [-0.06, -0.03, 0.0]:
                    for cap_name, cb, cs, cg in caps:
                        for rescue in [0, 1]:
                            configs.append({
                                "name": f"qt_b{base}_ba{brand_adj}_sa{specific_adj}_ga{generic_adj}_{cap_name}_r{rescue}",
                                "base_th": base,
                                "brand_adj": brand_adj,
                                "specific_adj": specific_adj,
                                "generic_adj": generic_adj,
                                "cap_brand": cb,
                                "cap_spec": cs,
                                "cap_generic": cg,
                                "rescue": rescue,
                            })

    return configs

CONFIGS = make_configs()

def apply_config(p1, part, cfg):
    n = len(part)

    brand = part["query_has_brand"].astype(bool).to_numpy()
    specific = part["query_is_specific"].astype(bool).to_numpy()
    generic = part["query_is_generic"].astype(bool).to_numpy()

    row_th = np.full(n, cfg["base_th"], dtype=np.float32)
    row_th += brand.astype(np.float32) * cfg["brand_adj"]
    row_th += ((~brand) & specific).astype(np.float32) * cfg["specific_adj"]
    row_th += generic.astype(np.float32) * cfg["generic_adj"]

    pred = (p1 >= row_th).astype(np.int8)

    rank = part["bge_blend_score_rank"].astype(float).to_numpy()

    pred[brand & (rank > cfg["cap_brand"])] = 0
    pred[(~brand) & specific & (rank > cfg["cap_spec"])] = 0
    pred[generic & (rank > cfg["cap_generic"])] = 0

    if cfg["rescue"]:
        v5_rank = part["proba_avg_rank"].astype(float).to_numpy()
        bge_rank = part["bge_blend_score_rank"].astype(float).to_numpy()
        bge_score = part["bge_blend_score"].astype(float).to_numpy()
        proba = part["proba_avg"].astype(float).to_numpy()

        rescue = (
            (pred == 0)
            & (bge_rank <= 3)
            & (v5_rank <= 8)
            & (bge_score >= 0.55)
            & (proba >= 0.55)
        )
        pred[rescue] = 1

    return pred

print("loading features...")
base = pd.read_parquet(BASE_FEATURES)
base["id"] = base["id"].astype(str)

lex = pd.read_parquet(LEX_FEATURES)
lex["id"] = lex["id"].astype(str)

feat = base.merge(lex, on="id", how="left", validate="one_to_one")

for c in LEX_FEATURE_COLS:
    feat[c] = feat[c].fillna(0)

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
    part = data.loc[test_mask].copy()

    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.25,
        min_samples_leaf=25,
        random_state=2028,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    p1 = model.predict_proba(X_test)[:, 1]

    # Baselines sadece kıyas için.
    for baseline in ["pred_v13", "pred_v15", "pred_v16"]:
        pred = part[baseline].astype(int).to_numpy()
        row = {
            "holdout": holdout,
            "variant": baseline,
            "threshold_config": "baseline",
            "n": len(part),
        }
        row.update(metrics(y_test, pred))
        rows.append(row)

    for cfg in CONFIGS:
        pred = apply_config(p1, part, cfg)
        row = {
            "holdout": holdout,
            "variant": "v17p4_" + cfg["name"],
            "threshold_config": cfg["name"],
            "n": len(part),
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
    score = 0.0
    wsum = 0.0
    vals = []

    for _, r in g.iterrows():
        w = weights.get(r["holdout"], 0.0)
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
        "sets": int(g["holdout"].nunique()),
    })

agg = pd.DataFrame(agg_rows).sort_values(
    ["weighted_macro", "min_macro", "mean_macro"],
    ascending=False,
)
agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP")
print(agg.head(60).to_string(index=False))

print("\ntraining final independent model on all labels...")
final_model = HistGradientBoostingClassifier(
    max_iter=260,
    learning_rate=0.035,
    max_leaf_nodes=15,
    l2_regularization=0.25,
    min_samples_leaf=25,
    random_state=2028,
)

final_model.fit(
    X_all,
    y_all,
    sample_weight=data["sample_weight"].to_numpy(),
)

X_full = feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype("float32")
p_full = final_model.predict_proba(X_full)[:, 1]

pd.DataFrame({"id": ids, "v17p4_proba": p_full.astype("float32")}).to_parquet(OUT_PROBA, index=False)

full_rows = []

top_variants = agg.head(10)["variant"].tolist()
for b in ["pred_v13", "pred_v15", "pred_v16"]:
    if b not in top_variants:
        top_variants.append(b)

for variant in top_variants:
    if variant == "pred_v13":
        pred = feat["pred_v13"].astype(np.int8).to_numpy()
    elif variant == "pred_v15":
        pred = feat["pred_v15"].astype(np.int8).to_numpy()
    elif variant == "pred_v16":
        pred = feat["pred_v16"].astype(np.int8).to_numpy()
    else:
        cfg_name = variant.replace("v17p4_", "")
        cfg = None
        for c in CONFIGS:
            if c["name"] == cfg_name:
                cfg = c
                break
        if cfg is None:
            print("missing config:", variant)
            continue
        pred = apply_config(p_full, feat, cfg)

    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v17p4_{clean_name(variant)}.csv"
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)

    pred_v13 = feat["pred_v13"].astype(np.int8).to_numpy()
    pred_v16 = feat["pred_v16"].astype(np.int8).to_numpy()

    full_rows.append({
        "variant": variant,
        "file": str(out),
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v13": int((pred != pred_v13).sum()),
        "diff_vs_v13_ratio": float((pred != pred_v13).mean()),
        "diff_vs_v16": int((pred != pred_v16).sum()),
        "diff_vs_v16_ratio": float((pred != pred_v16).mean()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

print("\nFULL SUMMARY")
print(full.to_string(index=False))

print("saved:", OUT_GRID)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
print("saved:", OUT_PROBA)
