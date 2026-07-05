from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V5_SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18_SCORE = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"
V21_SCORE = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V27_SCORE = ROOT / "data/processed/v27_sparse_aug_catboost_test_scores.parquet"
SPARSE_SCORE = ROOT / "data/processed/v26_sparse_text_scores.parquet"

QSWAP_LABELS = ROOT / "reports/manual_review/review_v29_qswap_targets_assistant_labeled.csv"

OUT_OOF = ROOT / "reports/manual_review/v30_micro_qswap_oof_report.csv"
OUT_FULL = ROOT / "reports/manual_review/v30_micro_qswap_full_summary.csv"
OUT_SELECTED = ROOT / "reports/manual_review/v30_micro_qswap_selected_swaps.csv"
OUT_CAND_SUM = ROOT / "reports/manual_review/v30_micro_qswap_candidate_summary.csv"

V24_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
]

TOP_QSWAP_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v29_qswap_v24_legacy_sparse_raw_g0p05_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v29_qswap_v24_legacy_sparse_raw_g0p05_b8000.csv",
]

EXTRA_PREDS = {
    "v13": [
        ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
    ],
    "v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
    "v19": [
        ROOT / "submissions/FINAL_MAIN_v19p1_honest_addonly_ba012_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba012_r1_v18th0p9.csv",
    ],
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
    ],
    "v22": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "strict2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget2000.csv",
    ],
    "supported2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget2000.csv",
    ],
}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def clean_name(x):
    return str(x).replace(".", "p").replace(" ", "_").replace("/", "_")

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def token_set(x):
    return set(clean_text(x).split())

def cov_jac(q, t):
    qset = token_set(q)
    tset = token_set(t)
    if not qset:
        return 0.0, 0.0
    inter = len(qset & tset)
    cov = inter / max(1, len(qset))
    jac = inter / max(1, len(qset | tset))
    return cov, jac

def load_pred_path(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()

def load_optional_pred(paths, sample):
    p = first_existing(paths)
    if p is None:
        return None
    return load_pred_path(p, sample)

def load_score(path, sample, cols, default=0.0):
    arr = np.full(len(sample), default, dtype=np.float32)
    if not path.exists():
        return arr
    d = pd.read_parquet(path)
    d["id"] = d["id"].astype(str)
    col = None
    for c in cols:
        if c in d.columns:
            col = c
            break
    if col is None:
        return arr
    if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        return d[col].astype("float32").fillna(default).to_numpy()
    tmp = sample[["id"]].merge(d[["id", col]], on="id", how="left", validate="one_to_one")
    return tmp[col].astype("float32").fillna(default).to_numpy()

def rank_score_by_term(df, col):
    r = df.groupby("term_id")[col].rank(method="first", ascending=False).astype("float32")
    cnt = df["candidate_count"].astype("float32")
    return (1.0 - ((r - 1.0) / np.maximum(cnt - 1.0, 1.0))).astype("float32").to_numpy()

print("loading sample/pairs")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
    raise RuntimeError("pairs id order mismatch")

v24_path = first_existing(V24_PATHS)
top_path = first_existing(TOP_QSWAP_PATHS)
if v24_path is None:
    raise FileNotFoundError("v24 anchor missing")
if top_path is None:
    raise FileNotFoundError("top qswap candidate missing")

print("v24:", v24_path)
print("qswap:", top_path)

pred_v24 = load_pred_path(v24_path, sample)
pred_top = load_pred_path(top_path, sample)

df = pairs.copy()
df["pred_v24"] = pred_v24
df["pred_top"] = pred_top
df["change"] = df["pred_top"] - df["pred_v24"]
df["candidate_count"] = df.groupby("term_id")["id"].transform("count").astype(np.int32)

adds = df[df["change"] == 1].copy()
drops = df[df["change"] == -1].copy()

adds["pair_rank"] = adds.groupby("term_id").cumcount()
drops["pair_rank"] = drops.groupby("term_id").cumcount()

swaps = adds.merge(
    drops,
    on=["term_id", "pair_rank"],
    suffixes=("_add", "_drop"),
    how="inner",
)

print("all swaps:", len(swaps))

# Preserve original submission row indices for applying swaps later.
swaps["idx_add"] = swaps["id_add"].map(id_to_idx)
swaps["idx_drop"] = swaps["id_drop"].map(id_to_idx)

if swaps["idx_add"].isna().any() or swaps["idx_drop"].isna().any():
    raise RuntimeError("Could not map some swap ids back to sample indices")

swaps["idx_add"] = swaps["idx_add"].astype("int32")
swaps["idx_drop"] = swaps["idx_drop"].astype("int32")

print("loading scores")
df["v5"] = load_score(V5_SCORE, sample, ["proba_avg", "v5_score", "score"])
df["v18"] = load_score(V18_SCORE, sample, ["v18_minilm_sigmoid", "v18_score", "score"])
df["v21"] = load_score(V21_SCORE, sample, ["v21_score", "score"])
df["v27"] = load_score(V27_SCORE, sample, ["v27_score", "score"])

if SPARSE_SCORE.exists():
    sp = pd.read_parquet(SPARSE_SCORE)
    sp["id"] = sp["id"].astype(str)
    tmp = sample[["id"]].merge(sp[["id", "sparse_blend", "sparse_pct_rank"]], on="id", how="left", validate="one_to_one")
    df["sparse_blend"] = tmp["sparse_blend"].astype("float32").fillna(0).to_numpy()
    df["sparse_pct_rank"] = tmp["sparse_pct_rank"].astype("float32").fillna(1).to_numpy()
else:
    df["sparse_blend"] = 0.0
    df["sparse_pct_rank"] = 1.0

for c in ["v5", "v18", "v21", "v27", "sparse_blend"]:
    df[f"{c}_rs"] = rank_score_by_term(df, c)

df["sparse_pct_rs"] = (1.0 - df["sparse_pct_rank"].astype("float32")).clip(0, 1)

preds = []
for name, paths in EXTRA_PREDS.items():
    arr = load_optional_pred(paths, sample)
    if arr is not None:
        preds.append(arr)
if preds:
    vote_mean = np.vstack(preds).mean(axis=0).astype("float32")
else:
    vote_mean = pred_v24.astype("float32")

df["vote_mean"] = vote_mean

score_cols = [
    "v5", "v18", "v21", "v27", "sparse_blend", "sparse_pct_rank",
    "v5_rs", "v18_rs", "v21_rs", "v27_rs", "sparse_blend_rs", "sparse_pct_rs",
    "vote_mean",
]

add_score_df = df[["id"] + score_cols].add_suffix("_add")
drop_score_df = df[["id"] + score_cols].add_suffix("_drop")

swaps = swaps.merge(add_score_df, left_on="id_add", right_on="id_add", how="left", validate="many_to_one")
swaps = swaps.merge(drop_score_df, left_on="id_drop", right_on="id_drop", how="left", validate="many_to_one")

print("loading metadata")
terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)
for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

swaps = swaps.merge(terms[["term_id", "query"]], on="term_id", how="left", validate="many_to_one")

item_add = items[["item_id", "title", "category", "brand", "gender", "age_group"]].add_suffix("_add")
item_drop = items[["item_id", "title", "category", "brand", "gender", "age_group"]].add_suffix("_drop")

swaps = swaps.merge(item_add, on="item_id_add", how="left", validate="many_to_one")
swaps = swaps.merge(item_drop, on="item_id_drop", how="left", validate="many_to_one")

# Text features
cov_add = []
jac_add = []
cov_drop = []
jac_drop = []
brand_add_in_q = []
brand_drop_in_q = []
same_brand = []
same_category = []
same_gender = []

for _, r in swaps.iterrows():
    q = r.get("query", "")
    ta = str(r.get("title_add", "")) + " " + str(r.get("category_add", "")) + " " + str(r.get("brand_add", ""))
    td = str(r.get("title_drop", "")) + " " + str(r.get("category_drop", "")) + " " + str(r.get("brand_drop", ""))

    ca, ja = cov_jac(q, ta)
    cd, jd = cov_jac(q, td)

    cov_add.append(ca)
    jac_add.append(ja)
    cov_drop.append(cd)
    jac_drop.append(jd)

    qclean = " " + clean_text(q) + " "
    ba = clean_text(r.get("brand_add", ""))
    bd = clean_text(r.get("brand_drop", ""))

    brand_add_in_q.append(1 if ba and (" " + ba + " ") in qclean else 0)
    brand_drop_in_q.append(1 if bd and (" " + bd + " ") in qclean else 0)

    same_brand.append(1 if ba and ba == bd else 0)
    same_category.append(1 if clean_text(r.get("category_add", "")) == clean_text(r.get("category_drop", "")) else 0)
    same_gender.append(1 if clean_text(r.get("gender_add", "")) == clean_text(r.get("gender_drop", "")) else 0)

swaps["q_cov_add"] = cov_add
swaps["q_jac_add"] = jac_add
swaps["q_cov_drop"] = cov_drop
swaps["q_jac_drop"] = jac_drop
swaps["q_cov_diff"] = swaps["q_cov_add"] - swaps["q_cov_drop"]
swaps["q_jac_diff"] = swaps["q_jac_add"] - swaps["q_jac_drop"]
swaps["brand_add_in_query"] = brand_add_in_q
swaps["brand_drop_in_query"] = brand_drop_in_q
swaps["brand_in_query_diff"] = swaps["brand_add_in_query"] - swaps["brand_drop_in_query"]
swaps["same_brand"] = same_brand
swaps["same_category"] = same_category
swaps["same_gender"] = same_gender

for c in score_cols:
    swaps[f"{c}_diff"] = swaps[f"{c}_add"] - swaps[f"{c}_drop"]

feature_cols = []
for c in score_cols:
    feature_cols += [f"{c}_add", f"{c}_drop", f"{c}_diff"]

feature_cols += [
    "q_cov_add", "q_cov_drop", "q_cov_diff",
    "q_jac_add", "q_jac_drop", "q_jac_diff",
    "brand_add_in_query", "brand_drop_in_query", "brand_in_query_diff",
    "same_brand", "same_category", "same_gender",
]

for c in feature_cols:
    swaps[c] = pd.to_numeric(swaps[c], errors="coerce").fillna(0).astype("float32")

# Labels
if not QSWAP_LABELS.exists():
    raise FileNotFoundError(QSWAP_LABELS)

lab = pd.read_csv(QSWAP_LABELS)
lab["id_add"] = lab["id_add"].astype(str)
lab["id_drop"] = lab["id_drop"].astype(str)
lab = lab[lab["assistant_swap_label"].isin([0, 1, "0", "1"])].copy()
lab["assistant_swap_label"] = lab["assistant_swap_label"].astype(int)

key_cols = ["id_add", "id_drop"]
train = swaps.merge(
    lab[key_cols + ["assistant_swap_label", "assistant_confidence", "needs_recheck"]],
    on=key_cols,
    how="inner",
    validate="one_to_one",
)

print("labeled swaps matched:", len(train))
print(train["assistant_swap_label"].value_counts().to_string())

X = train[feature_cols].to_numpy(dtype=np.float32)
y = train["assistant_swap_label"].to_numpy(dtype=np.int8)

weights = np.ones(len(train), dtype=np.float32)
conf = train["assistant_confidence"].astype(str).str.lower()
nr = pd.to_numeric(train["needs_recheck"], errors="coerce").fillna(1).astype(int)

weights[(nr == 0) & conf.eq("medium")] = 2.5
weights[(nr == 0) & conf.eq("high")] = 1.5
weights[(nr == 1)] = 0.75

model = HistGradientBoostingClassifier(
    max_iter=350,
    learning_rate=0.035,
    max_leaf_nodes=15,
    l2_regularization=0.08,
    min_samples_leaf=20,
    random_state=2030,
)

# OOF
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2030)
oof = np.zeros(len(train), dtype=np.float32)

for tr, va in skf.split(X, y):
    m = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        min_samples_leaf=20,
        random_state=2030,
    )
    m.fit(X[tr], y[tr], sample_weight=weights[tr])
    oof[va] = m.predict_proba(X[va])[:, 1].astype("float32")

oof_rows = []
auc = roc_auc_score(y, oof) if len(np.unique(y)) == 2 else np.nan

for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    p = (oof >= th).astype(int)
    oof_rows.append({
        "threshold": th,
        "auc": auc,
        "selected": int(p.sum()),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "accuracy": accuracy_score(y, p),
    })

for k in [50, 100, 150, 200, 300, 500]:
    order = np.argsort(-oof)
    p = np.zeros(len(y), dtype=int)
    p[order[:min(k, len(order))]] = 1
    oof_rows.append({
        "threshold": f"top{k}",
        "auc": auc,
        "selected": int(p.sum()),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "accuracy": accuracy_score(y, p),
    })

oof_df = pd.DataFrame(oof_rows)
oof_df.to_csv(OUT_OOF, index=False)
print("\nOOF REPORT")
print(oof_df.to_string(index=False))

# Fit final and score all swaps
model.fit(X, y, sample_weight=weights)
swaps["v30_swap_prob"] = model.predict_proba(swaps[feature_cols].to_numpy(dtype=np.float32))[:, 1].astype("float32")

# Candidate variants
variants = {}
variants["v24_anchor"] = pred_v24.copy()
variants["v29_full_qswap"] = pred_top.copy()

def make_pred(selected_swaps):
    pred = pred_v24.copy()
    pred[selected_swaps["idx_add"].to_numpy(dtype=np.int32)] = 1
    pred[selected_swaps["idx_drop"].to_numpy(dtype=np.int32)] = 0
    return pred

def add_variant(name, selected):
    pred = make_pred(selected)
    variants[name] = pred.astype(np.int8)

# Conservative: top-k only
for k in [25, 50, 100, 150, 200, 300, 500, 750, 1000]:
    selected = swaps.sort_values("v30_swap_prob", ascending=False).head(k).copy()
    add_variant(f"v30_micro_qswap_top{k}", selected)

# Probability thresholds
for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    selected = swaps[swaps["v30_swap_prob"] >= th].sort_values("v30_swap_prob", ascending=False).copy()
    if len(selected) > 0:
        add_variant(f"v30_micro_qswap_prob_ge_{str(th).replace('.', 'p')}", selected)

# Hybrid: probability >= threshold capped top-k
for th in [0.55, 0.60, 0.65]:
    pool = swaps[swaps["v30_swap_prob"] >= th].sort_values("v30_swap_prob", ascending=False)
    for k in [100, 250, 500]:
        selected = pool.head(k).copy()
        if len(selected) >= 25:
            add_variant(f"v30_micro_qswap_ge{str(th).replace('.', 'p')}_top{k}", selected)

# Summaries
summary_rows = []

lab_key = lab.set_index(["id_add", "id_drop"])["assistant_swap_label"]

for name, pred in variants.items():
    changed_add = np.where((pred_v24 == 0) & (pred == 1))[0]
    changed_drop = np.where((pred_v24 == 1) & (pred == 0))[0]

    # labeled selected swap precision
    sel = swaps[(pred[swaps["idx_add"].to_numpy(dtype=np.int32)] == 1) & (pred[swaps["idx_drop"].to_numpy(dtype=np.int32)] == 0)].copy()
    sel_keys = list(zip(sel["id_add"].astype(str), sel["id_drop"].astype(str)))
    labels = []
    for key in sel_keys:
        if key in lab_key.index:
            labels.append(int(lab_key.loc[key]))

    labeled_selected = len(labels)
    labeled_precision = float(np.mean(labels)) if labels else np.nan

    summary_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
        "add_count": int(len(changed_add)),
        "drop_count": int(len(changed_drop)),
        "labeled_selected": labeled_selected,
        "labeled_selected_precision": labeled_precision,
    })

summary = pd.DataFrame(summary_rows).sort_values(
    ["labeled_selected_precision", "diff_vs_v24"],
    ascending=[False, True],
)
summary.to_csv(OUT_FULL, index=False)

print("\nFULL SUMMARY")
print(summary.to_string(index=False))

# Save selected swap table
swaps_out = swaps.sort_values("v30_swap_prob", ascending=False).copy()
swaps_out.to_csv(OUT_SELECTED, index=False, encoding="utf-8-sig")

# Save only reasonable top candidates
save_names = []
for nm in summary[
    (summary["diff_vs_v24"] <= 2000)
    & (summary["labeled_selected_precision"].fillna(0) >= 0.55)
].head(10)["variant"].tolist():
    save_names.append(nm)

for nm in ["v30_micro_qswap_top50", "v30_micro_qswap_top100", "v30_micro_qswap_top200", "v24_anchor"]:
    if nm in variants and nm not in save_names:
        save_names.append(nm)

cand_rows = []

for name in save_names[:12]:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v30_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name].astype(np.int8)}).to_csv(out, index=False)
    row = summary[summary["variant"] == name].iloc[0].to_dict()
    row["file"] = str(out)
    cand_rows.append(row)
    print("saved", out)

pd.DataFrame(cand_rows).to_csv(OUT_CAND_SUM, index=False)

print("saved:", OUT_OOF)
print("saved:", OUT_FULL)
print("saved:", OUT_SELECTED)
print("saved:", OUT_CAND_SUM)
