from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

BASE_FEATURES = ROOT / "data/processed/v17_query_level_features.parquet"
LEX_FEATURES = ROOT / "data/processed/v17p4_independent_lexical_features.parquet"
V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

SUB_PATHS = {
    "v13": [
        ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
    ],
    "v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
    "v17p4": [
        ROOT / "submissions/FINAL_MAIN_v17p4_independent_b04_ba008_r1.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v17p4_v17p4_qt_b0p4_ba0p08_sa0p0_ga0p0_nocap_r1.csv",
    ],
}

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
}

OUT_GRID = ROOT / "reports/manual_review/v19p1_honest_loso_grid.csv"
OUT_AGG = ROOT / "reports/manual_review/v19p1_honest_loso_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v19p1_honest_loso_full_summary.csv"

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

V17_CONFIGS = [
    {
        "name": "b04_ba008_r1",
        "base_th": 0.40,
        "brand_adj": 0.08,
        "specific_adj": 0.0,
        "generic_adj": 0.0,
        "cap_brand": 9999,
        "cap_spec": 9999,
        "cap_generic": 9999,
        "rescue": 1,
    },
    {
        "name": "b04_ba012_r1",
        "base_th": 0.40,
        "brand_adj": 0.12,
        "specific_adj": 0.0,
        "generic_adj": 0.0,
        "cap_brand": 9999,
        "cap_spec": 9999,
        "cap_generic": 9999,
        "rescue": 1,
    },
    {
        "name": "b04_raw",
        "base_th": 0.40,
        "brand_adj": 0.0,
        "specific_adj": 0.0,
        "generic_adj": 0.0,
        "cap_brand": 9999,
        "cap_spec": 9999,
        "cap_generic": 9999,
        "rescue": 0,
    },
]

V18_THS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def sample_weight_from_labels(lab):
    w = np.ones(len(lab), dtype=np.float32)

    if "needs_recheck" in lab.columns:
        nr = lab["needs_recheck"].fillna(0).astype(int).to_numpy()
        w[nr == 1] *= 0.35

    if "assistant_confidence" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        w[conf.eq("high").to_numpy()] *= 1.20
        w[conf.eq("medium").to_numpy()] *= 1.00
        w[conf.eq("low").to_numpy()] *= 0.50

    return w

def apply_v17_config(p1, part, cfg):
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

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

def load_sub(name):
    p = first_existing(SUB_PATHS[name])
    if p is None:
        raise FileNotFoundError(f"Missing submission for {name}")
    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    assert d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), name
    return d["prediction"].astype(np.int8).to_numpy()

pred_v13_full = load_sub("v13")
pred_v16_full = load_sub("v16")
pred_v17_full = load_sub("v17p4")

print("loading features...")
base = pd.read_parquet(BASE_FEATURES)
base["id"] = base["id"].astype(str)

lex = pd.read_parquet(LEX_FEATURES)
lex["id"] = lex["id"].astype(str)

feat = base.merge(lex, on="id", how="left", validate="one_to_one")
for c in LEX_FEATURE_COLS:
    if c not in feat.columns:
        feat[c] = 0
    feat[c] = feat[c].fillna(0)

def safe_merge_scores(feat, path, wanted_cols):
    path = Path(path)
    if not path.exists():
        print("score file missing:", path)
        return feat

    print("checking score file:", path)
    d = pd.read_parquet(path)
    if "id" not in d.columns:
        print("  no id column, skipped")
        return feat

    d["id"] = d["id"].astype(str)

    keep = []
    for c in wanted_cols:
        if c in d.columns and c not in feat.columns:
            keep.append(c)

    if not keep:
        print("  no new columns")
        return feat

    print("  merging:", keep)
    return feat.merge(d[["id"] + keep], on="id", how="left", validate="one_to_one")

feat = safe_merge_scores(
    feat,
    ROOT / "data/processed/v5_e5base_full900_test_proba.parquet",
    ["proba_avg"]
)

feat = safe_merge_scores(
    feat,
    ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet",
    ["cross_sigmoid", "ce_sigmoid", "cross_score", "ce_score"]
)

feat = safe_merge_scores(
    feat,
    ROOT / "data/processed/v15_bge_reranker_candidate_scores.parquet",
    ["bge_sigmoid", "bge_blend_w020", "bge_blend_w030"]
)

# Alias d?zeltmeleri
if "cross_sigmoid" not in feat.columns and "ce_sigmoid" in feat.columns:
    feat["cross_sigmoid"] = feat["ce_sigmoid"]

# Eksik feature kal?rsa modeli patlatma; -1 ile doldur.
missing_cols = [c for c in FEATURE_COLS if c not in feat.columns]
if missing_cols:
    print("WARNING missing feature cols filled with -1:", missing_cols)
    for c in missing_cols:
        feat[c] = -1.0

for c in FEATURE_COLS:
    feat[c] = pd.to_numeric(feat[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1.0)

assert feat["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))

print("loading v18...")
v18 = pd.read_parquet(V18)
v18["id"] = v18["id"].astype(str)

v18_score_full = np.full(n, -1.0, dtype=np.float32)
idx = v18["id"].map(id_to_idx)
ok = idx.notna()
v18_score_full[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()
has_v18_full = v18_score_full >= 0
print("v18 scored rows:", int(has_v18_full.sum()))

print("loading labels...")
labs = []
for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label:", path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)
    lab["label_set"] = label_set
    lab["sample_weight"] = sample_weight_from_labels(lab)
    labs.append(lab)

labels = pd.concat(labs, ignore_index=True)

data = labels.merge(feat, on="id", how="inner", validate="many_to_one")

# Merge sonras? ayn? isimli feature kolonlar? _x/_y olarak b?l?nebilir.
# ?ncelik feat taraf?ndan gelen _y kolonunda; yoksa _x; o da yoksa -1.
for c in FEATURE_COLS:
    if c not in data.columns:
        if f"{c}_y" in data.columns:
            data[c] = data[f"{c}_y"]
        elif f"{c}_x" in data.columns:
            data[c] = data[f"{c}_x"]
        else:
            print("WARNING data missing feature filled -1:", c)
            data[c] = -1.0

for c in FEATURE_COLS:
    data[c] = pd.to_numeric(data[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1.0)

print("labeled rows:", len(data))
print(data["label_set"].value_counts().to_string())

X_all = data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype("float32")
y_all = data["assistant_label"].astype(int).to_numpy()

rows = []

for holdout in sorted(data["label_set"].unique()):
    print("\nHOLDOUT:", holdout)

    train_mask = data["label_set"] != holdout
    test_mask = data["label_set"] == holdout

    X_train = X_all.loc[train_mask]
    y_train = y_all[train_mask.to_numpy()]
    w_train = data.loc[train_mask, "sample_weight"].to_numpy()

    part = data.loc[test_mask].copy()
    X_test = X_all.loc[test_mask]
    y_test = y_all[test_mask.to_numpy()]

    mapped = part["id"].map(id_to_idx).astype(int).to_numpy()

    pred_v13 = pred_v13_full[mapped]
    pred_v16 = pred_v16_full[mapped]
    pred_v17_leaky = pred_v17_full[mapped]

    v18_score = v18_score_full[mapped]
    has_v18 = v18_score >= 0

    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.25,
        min_samples_leaf=25,
        random_state=2029,
    )

    model.fit(X_train, y_train, sample_weight=w_train)
    p17 = model.predict_proba(X_test)[:, 1].astype(np.float32)

    variant_preds = {}

    variant_preds["v13_public_0p77"] = pred_v13
    variant_preds["v16_current"] = pred_v16
    variant_preds["v17p4_leaky_reference"] = pred_v17_leaky

    for add_th in [0.90, 0.95, 0.97, 0.99]:
        pred = pred_v16.copy()
        pred[(pred_v16 == 0) & has_v18 & (v18_score >= add_th)] = 1
        variant_preds[f"v18_addonly_v16_add{add_th}"] = pred

    for cfg in V17_CONFIGS:
        honest_v17 = apply_v17_config(p17, part, cfg)
        variant_preds[f"honest_v17_{cfg['name']}"] = honest_v17

        for v18_th in V18_THS:
            v18_bin = np.zeros(len(part), dtype=np.int8)
            v18_bin[has_v18] = (v18_score[has_v18] >= v18_th).astype(np.int8)

            pred = pred_v16.copy()
            votes = pred_v16 + honest_v17 + v18_bin
            pred[has_v18] = (votes[has_v18] >= 2).astype(np.int8)
            variant_preds[f"v19p1_honest_vote2of3_{cfg['name']}_v18th{v18_th}"] = pred

            pred = pred_v16.copy()
            add = (pred_v16 == 0) & (honest_v17 == 1) & has_v18 & (v18_score >= v18_th)
            pred[add] = 1
            variant_preds[f"v19p1_honest_addonly_{cfg['name']}_v18th{v18_th}"] = pred

    # evaluate subsets
    hold_lab = labels[labels["label_set"] == holdout].copy()

    subsets = {"all": hold_lab}

    if "needs_recheck" in hold_lab.columns:
        nr = hold_lab["needs_recheck"].fillna(0).astype(int)
        subsets["clean"] = hold_lab[nr == 0].copy()

    if "assistant_confidence" in hold_lab.columns and "needs_recheck" in hold_lab.columns:
        conf = hold_lab["assistant_confidence"].astype(str).str.lower()
        nr = hold_lab["needs_recheck"].fillna(0).astype(int)
        subsets["high_medium_clean"] = hold_lab[(nr == 0) & conf.isin(["high", "medium"])].copy()
        subsets["high_clean"] = hold_lab[(nr == 0) & conf.eq("high")].copy()

    # map subset ids to part positions
    part_pos = pd.Series(np.arange(len(part)), index=part["id"])

    for subset_name, sub in subsets.items():
        if len(sub) < 30 or sub["assistant_label"].nunique() < 2:
            continue

        pos = sub["id"].map(part_pos)
        ok = pos.notna()
        sub = sub.loc[ok].copy()
        pos = pos.loc[ok].astype(int).to_numpy()
        y = sub["assistant_label"].to_numpy()

        for name, pred_all in variant_preds.items():
            p = pred_all[pos]
            row = {
                "holdout": holdout,
                "subset": subset_name,
                "eval_key": f"{holdout}_{subset_name}",
                "variant": name,
                "n": len(sub),
            }
            row.update(metrics(y, p))
            rows.append(row)

grid = pd.DataFrame(rows)
grid.to_csv(OUT_GRID, index=False)

weights = {
    "random_clean_v2_all": 0.30,
    "random_clean_v2_clean": 0.30,
    "manual_v1_clean": 0.35,
    "manual_v1_high_clean": 0.20,
    "manual_v1_all": 0.15,
    "review_v15_vs_v13_clean": 0.20,
    "review_v15_vs_v13_high_medium_clean": 0.20,
    "review_v13_vs_v5_clean": 0.10,
    "review_v13_vs_v5_high_medium_clean": 0.10,
}

agg_rows = []
for name, g in grid.groupby("variant"):
    score = 0.0
    wsum = 0.0
    vals = []

    for _, r in g.iterrows():
        w = weights.get(r["eval_key"], 0.0)
        if w > 0:
            score += w * r["macro_f1"]
            wsum += w
            vals.append(r["macro_f1"])

    if wsum == 0:
        continue

    main = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "weighted_macro": score / wsum,
        "min_macro_used": float(np.min(vals)),
        "mean_macro_used": float(np.mean(vals)),
        "main_min_macro": float(main["macro_f1"].min()) if len(main) else np.nan,
        "main_mean_macro": float(main["macro_f1"].mean()) if len(main) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "mean_pred_pos_ratio": float(g["pred_pos_ratio"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows).sort_values(
    ["weighted_macro", "main_min_macro", "main_mean_macro"],
    ascending=False,
)

agg.to_csv(OUT_AGG, index=False)

print("\nHONEST AGG TOP")
print(agg.head(80).to_string(index=False))

# Final full submissions for top honest variants:
# Final model can use all manual labels; validation above is honest.
print("\ntraining final full v17 model...")
final_model = HistGradientBoostingClassifier(
    max_iter=260,
    learning_rate=0.035,
    max_leaf_nodes=15,
    l2_regularization=0.25,
    min_samples_leaf=25,
    random_state=2029,
)

final_model.fit(X_all, y_all, sample_weight=data["sample_weight"].to_numpy())

X_full = feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype("float32")
p17_full = final_model.predict_proba(X_full)[:, 1].astype(np.float32)

full_variants = {}
full_variants["v13_public_0p77"] = pred_v13_full.copy()
full_variants["v16_current"] = pred_v16_full.copy()
full_variants["v17p4_existing_reference"] = pred_v17_full.copy()

for add_th in [0.90, 0.95, 0.97, 0.99]:
    pred = pred_v16_full.copy()
    pred[(pred_v16_full == 0) & has_v18_full & (v18_score_full >= add_th)] = 1
    full_variants[f"v18_addonly_v16_add{add_th}"] = pred

top_honest = agg.head(12)["variant"].tolist()

for variant in top_honest:
    if variant in full_variants:
        continue

    if variant.startswith("honest_v17_"):
        cfg_name = variant.replace("honest_v17_", "")
        cfg = next((c for c in V17_CONFIGS if c["name"] == cfg_name), None)
        if cfg is None:
            continue
        pred = apply_v17_config(p17_full, feat, cfg)
        full_variants[variant] = pred

    elif variant.startswith("v19p1_honest_vote2of3_"):
        rest = variant.replace("v19p1_honest_vote2of3_", "")
        cfg_name, th_s = rest.rsplit("_v18th", 1)
        v18_th = float(th_s)
        cfg = next((c for c in V17_CONFIGS if c["name"] == cfg_name), None)
        if cfg is None:
            continue

        v17_bin = apply_v17_config(p17_full, feat, cfg)
        v18_bin = np.zeros(n, dtype=np.int8)
        v18_bin[has_v18_full] = (v18_score_full[has_v18_full] >= v18_th).astype(np.int8)

        pred = pred_v16_full.copy()
        votes = pred_v16_full + v17_bin + v18_bin
        pred[has_v18_full] = (votes[has_v18_full] >= 2).astype(np.int8)
        full_variants[variant] = pred

    elif variant.startswith("v19p1_honest_addonly_"):
        rest = variant.replace("v19p1_honest_addonly_", "")
        cfg_name, th_s = rest.rsplit("_v18th", 1)
        v18_th = float(th_s)
        cfg = next((c for c in V17_CONFIGS if c["name"] == cfg_name), None)
        if cfg is None:
            continue

        v17_bin = apply_v17_config(p17_full, feat, cfg)
        pred = pred_v16_full.copy()
        add = (pred_v16_full == 0) & (v17_bin == 1) & has_v18_full & (v18_score_full >= v18_th)
        pred[add] = 1
        full_variants[variant] = pred

full_rows = []
for name, pred in full_variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v13": int((pred != pred_v13_full).sum()),
        "diff_vs_v13_ratio": float((pred != pred_v13_full).mean()),
        "diff_vs_v16": int((pred != pred_v16_full).sum()),
        "diff_vs_v16_ratio": float((pred != pred_v16_full).mean()),
        "diff_vs_v17p4": int((pred != pred_v17_full).sum()),
        "diff_vs_v17p4_ratio": float((pred != pred_v17_full).mean()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

print("\nFULL SUMMARY")
print(full.sort_values("diff_vs_v16").head(80).to_string(index=False))

# save top candidates
top_save = agg.head(8)["variant"].tolist()
for b in ["v16_current", "v13_public_0p77", "v18_addonly_v16_add0.95"]:
    if b not in top_save:
        top_save.append(b)

for name in top_save:
    if name not in full_variants:
        continue
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v19p1_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": full_variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_GRID)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
