from pathlib import Path
import re
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"

LABELED_REVIEW_PATHS = [
    ROOT / "reports/manual_review/review_v33_final_qswap_targets_assistant_labeled.csv",
    ROOT / "reports/manual_review/review_v33_final_qswap_targets.csv",
]

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
]

FULL_V33_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v33_v24_qswap_v33_pure_hybrid_safe_g0p06_b24446.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_v24_qswap_v33_pure_hybrid_safe_g0p03_b24446.csv",
]

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
    "v26_sparse": ROOT / "reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv",
}

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_SWAPS = OUT_DIR / "v33_perfected_all_swaps_scored.csv"
OUT_SUMMARY = OUT_DIR / "v33_perfected_candidate_summary.csv"
OUT_MODEL = OUT_DIR / "v33_perfected_model_diagnostics.json"
OUT_SAVED = OUT_DIR / "v33_perfected_saved_candidates.csv"


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def clean_name(x):
    return (
        str(x)
        .replace(".", "p")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace("(", "")
        .replace(")", "")
    )


def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def sigmoid(x):
    x = np.clip(x, -12, 12)
    return 1.0 / (1.0 + np.exp(-x))


def norm_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    tr = str.maketrans("çğıöşüİ", "cgiosui")
    s = s.translate(tr)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


STOP = {
    "ve", "ile", "icin", "bir", "bu", "su", "the", "of", "in",
    "model", "adet", "renk", "beden", "numara", "set"
}


def tokens(s):
    t = [x for x in norm_text(s).split() if len(x) >= 2 and x not in STOP]
    return set(t)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def brand_hit(query, brand):
    q = norm_text(query)
    b = norm_text(brand)
    if not b:
        return 0.0
    parts = [x for x in b.split() if len(x) >= 2]
    if not parts:
        return 0.0
    hit = any(re.search(rf"\b{re.escape(x)}\b", q) for x in parts)
    return 1.0 if hit else 0.0


def expected_gender(query):
    q = norm_text(query)
    words = set(q.split())

    if any(w in words for w in ["erkek", "bay", "men", "man"]):
        return "male"
    if any(w in words for w in ["kadin", "bayan", "woman", "women"]):
        return "female"
    if any(w in words for w in ["kiz", "kız"]):
        return "female_child"
    if any(w in words for w in ["cocuk", "bebek", "baby", "kids", "kid"]):
        return "child"
    return ""


def item_gender_bucket(gender, age_group):
    g = norm_text(gender)
    a = norm_text(age_group)
    joined = f"{g} {a}"

    if any(x in joined for x in ["erkek", "male", "bay"]):
        return "male"
    if any(x in joined for x in ["kadin", "female", "bayan"]):
        return "female"
    if any(x in joined for x in ["cocuk", "bebek", "kids", "baby"]):
        return "child"
    return ""


def gender_match_score(query, gender, age_group):
    eg = expected_gender(query)
    ig = item_gender_bucket(gender, age_group)

    if not eg:
        return 0.0
    if not ig:
        return -0.1
    if eg == ig:
        return 1.0
    if eg == "female_child" and ig in ["female", "child"]:
        return 0.6
    if eg == "child" and ig in ["male", "female"]:
        return 0.2
    return -1.0


def safe_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def label_to_int(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in ["1", "true", "yes", "iyi", "good", "correct", "dogru", "doğru"]:
        return 1
    if s in ["0", "false", "no", "kotu", "kötü", "bad", "wrong", "yanlis", "yanlış"]:
        return 0

    try:
        v = int(float(s))
        if v in [0, 1]:
            return v
    except Exception:
        pass

    return np.nan


def add_text_features(swaps):
    print("adding text features...")

    q_tok = swaps["query"].map(tokens)
    title_add_tok = swaps["title_add"].map(tokens)
    title_drop_tok = swaps["title_drop"].map(tokens)
    cat_add_tok = swaps["category_add"].map(tokens)
    cat_drop_tok = swaps["category_drop"].map(tokens)

    swaps["q_title_jac_add"] = [jaccard(a, b) for a, b in zip(q_tok, title_add_tok)]
    swaps["q_title_jac_drop"] = [jaccard(a, b) for a, b in zip(q_tok, title_drop_tok)]
    swaps["q_title_jac_gain"] = swaps["q_title_jac_add"] - swaps["q_title_jac_drop"]

    swaps["q_cat_jac_add"] = [jaccard(a, b) for a, b in zip(q_tok, cat_add_tok)]
    swaps["q_cat_jac_drop"] = [jaccard(a, b) for a, b in zip(q_tok, cat_drop_tok)]
    swaps["q_cat_jac_gain"] = swaps["q_cat_jac_add"] - swaps["q_cat_jac_drop"]

    swaps["brand_hit_add"] = [brand_hit(q, b) for q, b in zip(swaps["query"], swaps["brand_add"])]
    swaps["brand_hit_drop"] = [brand_hit(q, b) for q, b in zip(swaps["query"], swaps["brand_drop"])]
    swaps["brand_hit_gain"] = swaps["brand_hit_add"] - swaps["brand_hit_drop"]

    swaps["gender_match_add"] = [
        gender_match_score(q, g, a)
        for q, g, a in zip(swaps["query"], swaps["gender_add"], swaps["age_group_add"])
    ]
    swaps["gender_match_drop"] = [
        gender_match_score(q, g, a)
        for q, g, a in zip(swaps["query"], swaps["gender_drop"], swaps["age_group_drop"])
    ]
    swaps["gender_match_gain"] = swaps["gender_match_add"] - swaps["gender_match_drop"]

    swaps["title_len_add"] = swaps["title_add"].fillna("").astype(str).str.len()
    swaps["title_len_drop"] = swaps["title_drop"].fillna("").astype(str).str.len()
    swaps["title_len_gain"] = swaps["title_len_add"] - swaps["title_len_drop"]

    return swaps


def macro_f1(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)


# ------------------------------------------------------------
# Load base data
# ------------------------------------------------------------
print("loading base files...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

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
    raise RuntimeError("submission_pairs id order mismatch")

anchor_path = first_existing(ANCHOR_PATHS)
full_path = first_existing(FULL_V33_PATHS)
labeled_path = first_existing(LABELED_REVIEW_PATHS)

if anchor_path is None:
    raise FileNotFoundError("v24 anchor bulunamadı")
if full_path is None:
    raise FileNotFoundError("full v33 candidate bulunamadı")
if labeled_path is None:
    raise FileNotFoundError("assistant labeled review CSV bulunamadı")

print("anchor:", anchor_path)
print("full_v33:", full_path)
print("labeled_review:", labeled_path)

pred_anchor = load_pred(anchor_path, sample)
pred_full = load_pred(full_path, sample)

# ------------------------------------------------------------
# Rebuild swap pool from full candidate
# ------------------------------------------------------------
print("rebuilding full swap pool...")
df = pairs.copy()
df["pred_anchor"] = pred_anchor
df["pred_full"] = pred_full
df["change"] = df["pred_full"] - df["pred_anchor"]

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

print("adds:", len(adds), "drops:", len(drops), "paired swaps:", len(swaps))

# ------------------------------------------------------------
# Merge V33 score columns
# ------------------------------------------------------------
print("merging v33 scores...")
v33 = pd.read_parquet(V33)
v33["id"] = v33["id"].astype(str)

score_cols = ["id", "v33_ft_score", "v33_ft_rank", "v33_ft_pct_rank", "v33_ft_term_z"]
swaps = swaps.merge(v33[score_cols].add_suffix("_add"), on="id_add", how="left", validate="many_to_one")
swaps = swaps.merge(v33[score_cols].add_suffix("_drop"), on="id_drop", how="left", validate="many_to_one")

for side in ["add", "drop"]:
    swaps[f"v33_rs_{side}"] = 1.0 - safe_num(swaps[f"v33_ft_pct_rank_{side}"], 1.0)
    swaps[f"v33_zsig_{side}"] = sigmoid(safe_num(swaps[f"v33_ft_term_z_{side}"], 0.0).to_numpy() / 1.6)

score_min = float(v33["v33_ft_score"].min())
score_max = float(v33["v33_ft_score"].max())

for side in ["add", "drop"]:
    swaps[f"v33_global_{side}"] = (
        (safe_num(swaps[f"v33_ft_score_{side}"], score_min) - score_min)
        / max(1e-6, score_max - score_min)
    )

    swaps[f"v33_pure_{side}"] = (
        0.70 * swaps[f"v33_rs_{side}"]
        + 0.20 * swaps[f"v33_zsig_{side}"]
        + 0.10 * swaps[f"v33_global_{side}"]
    )

swaps["v33_score_gain"] = safe_num(swaps["v33_ft_score_add"], 0.0) - safe_num(swaps["v33_ft_score_drop"], 0.0)
swaps["v33_rank_gain"] = safe_num(swaps["v33_ft_pct_rank_drop"], 1.0) - safe_num(swaps["v33_ft_pct_rank_add"], 1.0)
swaps["v33_z_gain"] = safe_num(swaps["v33_ft_term_z_add"], 0.0) - safe_num(swaps["v33_ft_term_z_drop"], 0.0)
swaps["v33_pure_gain"] = swaps["v33_pure_add"] - swaps["v33_pure_drop"]

swaps = swaps.sort_values("v33_pure_gain", ascending=False).reset_index(drop=True)
swaps["pure_rank"] = np.arange(1, len(swaps) + 1)
swaps["pure_rank_frac"] = (swaps["pure_rank"] - 1) / max(1, len(swaps) - 1)

raw_rank = swaps["v33_score_gain"].rank(method="first", ascending=False)
swaps["raw_rank_frac"] = (raw_rank - 1) / max(1, len(swaps) - 1)

# ------------------------------------------------------------
# Merge text data
# ------------------------------------------------------------
print("merging text data...")
terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)

for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

item_cols = ["item_id", "title", "category", "brand", "gender", "age_group"]

swaps = swaps.merge(terms[["term_id", "query"]], on="term_id", how="left", validate="many_to_one")
swaps = swaps.merge(items[item_cols].add_suffix("_add"), on="item_id_add", how="left", validate="many_to_one")
swaps = swaps.merge(items[item_cols].add_suffix("_drop"), on="item_id_drop", how="left", validate="many_to_one")

swaps = add_text_features(swaps)

swaps["swap_key"] = swaps["id_add"].astype(str) + "|" + swaps["id_drop"].astype(str)

# ------------------------------------------------------------
# Merge assistant labels
# ------------------------------------------------------------
print("loading assistant labels...")
lab = pd.read_csv(labeled_path)
lab["id_add"] = lab["id_add"].astype(str)
lab["id_drop"] = lab["id_drop"].astype(str)
lab["swap_key"] = lab["id_add"] + "|" + lab["id_drop"]

if "assistant_swap_label" not in lab.columns:
    raise RuntimeError("assistant_swap_label kolonu yok. Labeled CSV doğru dosya değil.")

lab["swap_label"] = lab["assistant_swap_label"].map(label_to_int)
lab = lab[lab["swap_label"].isin([0, 1])].copy()
lab["swap_label"] = lab["swap_label"].astype(int)

if "assistant_confidence" not in lab.columns:
    lab["assistant_confidence"] = "medium"
if "needs_recheck" not in lab.columns:
    lab["needs_recheck"] = 0
if "review_bucket" not in lab.columns:
    lab["review_bucket"] = "unknown"

lab["assistant_confidence"] = lab["assistant_confidence"].astype(str).str.lower()
lab["needs_recheck"] = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)

print("label summary:")
print(lab.groupby("review_bucket")["swap_label"].agg(["count", "mean"]).to_string())
print("overall labeled precision:", lab["swap_label"].mean())

swaps = swaps.merge(
    lab[["swap_key", "swap_label", "assistant_confidence", "needs_recheck", "review_bucket"]],
    on="swap_key",
    how="left",
    validate="one_to_one",
)

# ------------------------------------------------------------
# Train swap quality model
# ------------------------------------------------------------
FEATURES = [
    "v33_ft_score_add",
    "v33_ft_score_drop",
    "v33_score_gain",
    "v33_ft_pct_rank_add",
    "v33_ft_pct_rank_drop",
    "v33_rank_gain",
    "v33_ft_term_z_add",
    "v33_ft_term_z_drop",
    "v33_z_gain",
    "v33_rs_add",
    "v33_rs_drop",
    "v33_zsig_add",
    "v33_zsig_drop",
    "v33_pure_add",
    "v33_pure_drop",
    "v33_pure_gain",
    "pure_rank_frac",
    "raw_rank_frac",
    "q_title_jac_add",
    "q_title_jac_drop",
    "q_title_jac_gain",
    "q_cat_jac_add",
    "q_cat_jac_drop",
    "q_cat_jac_gain",
    "brand_hit_add",
    "brand_hit_drop",
    "brand_hit_gain",
    "gender_match_add",
    "gender_match_drop",
    "gender_match_gain",
    "title_len_add",
    "title_len_drop",
    "title_len_gain",
]

train_mask = swaps["swap_label"].isin([0, 1])
train_df = swaps[train_mask].copy()

if len(train_df) < 200:
    raise RuntimeError(f"Yeterli label yok: {len(train_df)}")

X = train_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
y = train_df["swap_label"].astype(int).to_numpy()

weights = np.ones(len(train_df), dtype=np.float32)
weights[train_df["assistant_confidence"].eq("high").to_numpy()] *= 1.15
weights[train_df["assistant_confidence"].eq("medium").to_numpy()] *= 0.90
weights[train_df["assistant_confidence"].eq("low").to_numpy()] *= 0.55
weights[train_df["needs_recheck"].to_numpy() == 1] *= 0.35

print("training rows:", len(train_df), "positive rate:", float(y.mean()))

oof_hgb = np.zeros(len(train_df), dtype=np.float32)
oof_ext = np.zeros(len(train_df), dtype=np.float32)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

for fold, (tr, va) in enumerate(cv.split(X, y), 1):
    hgb = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.06,
        min_samples_leaf=18,
        random_state=2026 + fold,
    )

    ext = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=6,
        min_samples_leaf=16,
        max_features="sqrt",
        random_state=3026 + fold,
        n_jobs=-1,
        class_weight="balanced",
    )

    hgb.fit(X.iloc[tr], y[tr], sample_weight=weights[tr])
    ext.fit(X.iloc[tr], y[tr], sample_weight=weights[tr])

    oof_hgb[va] = hgb.predict_proba(X.iloc[va])[:, 1]
    oof_ext[va] = ext.predict_proba(X.iloc[va])[:, 1]

oof = 0.55 * oof_hgb + 0.45 * oof_ext

diag = {
    "train_rows": int(len(train_df)),
    "label_positive_rate": float(y.mean()),
    "oof_auc": float(roc_auc_score(y, oof)) if len(np.unique(y)) == 2 else None,
    "oof_ap": float(average_precision_score(y, oof)) if len(np.unique(y)) == 2 else None,
}

for t in [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    m = oof >= t
    if m.sum() > 0:
        diag[f"oof_precision_ge_{t}"] = float(y[m].mean())
        diag[f"oof_count_ge_{t}"] = int(m.sum())

print("model diagnostics:")
print(json.dumps(diag, indent=2))

hgb_full = HistGradientBoostingClassifier(
    max_iter=450,
    learning_rate=0.030,
    max_leaf_nodes=15,
    l2_regularization=0.06,
    min_samples_leaf=18,
    random_state=4040,
)

ext_full = ExtraTreesClassifier(
    n_estimators=900,
    max_depth=6,
    min_samples_leaf=16,
    max_features="sqrt",
    random_state=5050,
    n_jobs=-1,
    class_weight="balanced",
)

hgb_full.fit(X, y, sample_weight=weights)
ext_full.fit(X, y, sample_weight=weights)

X_all = swaps[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
swaps["qprob_hgb"] = hgb_full.predict_proba(X_all)[:, 1].astype(np.float32)
swaps["qprob_ext"] = ext_full.predict_proba(X_all)[:, 1].astype(np.float32)
swaps["qprob"] = (0.55 * swaps["qprob_hgb"] + 0.45 * swaps["qprob_ext"]).astype(np.float32)

pure_gain_rank_score = 1.0 - swaps["pure_rank_frac"].astype(np.float32)
raw_gain_rank_score = 1.0 - swaps["raw_rank_frac"].astype(np.float32)

swaps["utility"] = (
    0.56 * swaps["qprob"].astype(np.float32)
    + 0.26 * pure_gain_rank_score
    + 0.18 * raw_gain_rank_score
).astype(np.float32)

swaps["conservative_utility"] = (
    0.72 * swaps["qprob"].astype(np.float32)
    + 0.18 * pure_gain_rank_score
    + 0.10 * raw_gain_rank_score
).astype(np.float32)

swaps.to_csv(OUT_SWAPS, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------
variants = {}
selected_maps = {}


def make_pred_from_sel(name, sel):
    sel = sel.drop_duplicates(["id_add", "id_drop"]).copy()
    pred = pred_anchor.copy()

    add_idx = sel["id_add"].map(id_to_idx).astype(int).to_numpy()
    drop_idx = sel["id_drop"].map(id_to_idx).astype(int).to_numpy()

    pred[add_idx] = 1
    pred[drop_idx] = 0

    variants[name] = pred.astype(np.int8)
    selected_maps[name] = set(sel["swap_key"].astype(str).tolist())


def take_top(name, sort_col, k, mask=None):
    tmp = swaps.copy()
    if mask is not None:
        tmp = tmp[mask].copy()
    tmp = tmp.sort_values(sort_col, ascending=False).head(k).copy()
    if len(tmp) >= 50:
        make_pred_from_sel(name, tmp)


budgets = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 12500]

for k in budgets:
    take_top(f"puregain_top{k}", "v33_pure_gain", k)
    take_top(f"rawgain_top{k}", "v33_score_gain", k)
    take_top(f"qprob_top{k}", "qprob", k)
    take_top(f"utility_top{k}", "utility", k)
    take_top(f"consutil_top{k}", "conservative_utility", k)

for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = swaps["qprob"] >= t
    for k in [500, 1000, 2000, 3000, 5000, 7500, 10000]:
        take_top(f"qprobge{str(t).replace('.', 'p')}_utility_top{k}", "utility", k, mask=mask)

for t in [0.45, 0.50, 0.55, 0.60]:
    mask = (swaps["qprob"] >= t) & (swaps["pure_rank"] <= 10000)
    for k in [500, 1000, 1500, 2000, 3000, 5000, 7500, 10000]:
        take_top(f"top10k_qprobge{str(t).replace('.', 'p')}_utility_top{k}", "utility", k, mask=mask)

for max_rank in [2000, 3000, 5000, 7500, 10000, 12500]:
    for t in [0.45, 0.50, 0.55, 0.60]:
        mask = (swaps["pure_rank"] <= max_rank) & (swaps["qprob"] >= t)
        tmp = swaps[mask].sort_values("utility", ascending=False).copy()
        if len(tmp) >= 100:
            make_pred_from_sel(f"ranklt{max_rank}_qprobge{str(t).replace('.', 'p')}_all{len(tmp)}", tmp)

variants["v24_anchor"] = pred_anchor.copy()
selected_maps["v24_anchor"] = set()

variants["v33_full_24446_DO_NOT_SUBMIT"] = pred_full.copy()
selected_maps["v33_full_24446_DO_NOT_SUBMIT"] = set(swaps["swap_key"].astype(str).tolist())

print("candidate count:", len(variants))

# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------
def evaluate_swap_labels(name):
    sel_keys = selected_maps[name]
    if not sel_keys:
        return {
            "swap_selected": 0,
            "swap_labeled_n": 0,
            "swap_label_precision": np.nan,
            "swap_top_gain_labeled_n": 0,
            "swap_top_gain_precision": np.nan,
            "swap_mid_gain_labeled_n": 0,
            "swap_mid_gain_precision": np.nan,
            "swap_tail_gain_labeled_n": 0,
            "swap_tail_gain_precision": np.nan,
        }

    part = lab[lab["swap_key"].isin(sel_keys)].copy()

    out = {
        "swap_selected": len(sel_keys),
        "swap_labeled_n": int(len(part)),
        "swap_label_precision": float(part["swap_label"].mean()) if len(part) else np.nan,
    }

    for bucket in ["top_gain", "mid_gain", "tail_gain"]:
        b = part[part["review_bucket"].astype(str).eq(bucket)]
        out[f"swap_{bucket}_labeled_n"] = int(len(b))
        out[f"swap_{bucket}_precision"] = float(b["swap_label"].mean()) if len(b) else np.nan

    return out


def evaluate_pair_labels(pred):
    scores = []
    weights = {
        "random_clean_v2_all": 0.28,
        "manual_v1_clean": 0.24,
        "manual_v1_high_clean": 0.14,
        "review_v15_vs_v13_clean": 0.12,
        "review_v13_vs_v5_clean": 0.06,
        "v20_active_clean": 0.04,
        "v21_active_clean": 0.07,
        "v26_sparse_clean": 0.05,
    }
    detail = {}

    for label_name, path in LABEL_FILES.items():
        if not path.exists():
            continue

        d = pd.read_csv(path)
        if "id" not in d.columns or "assistant_label" not in d.columns:
            continue

        d["id"] = d["id"].astype(str)
        d = d[d["assistant_label"].isin([0, 1, "0", "1"])].copy()
        if len(d) < 30:
            continue

        d["assistant_label"] = d["assistant_label"].astype(int)
        subsets = {"all": d}

        if "needs_recheck" in d.columns:
            nr = pd.to_numeric(d["needs_recheck"], errors="coerce").fillna(0).astype(int)
            subsets["clean"] = d[nr == 0].copy()

        if "assistant_confidence" in d.columns and "needs_recheck" in d.columns:
            conf = d["assistant_confidence"].astype(str).str.lower()
            nr = pd.to_numeric(d["needs_recheck"], errors="coerce").fillna(0).astype(int)
            subsets["high_clean"] = d[(nr == 0) & conf.eq("high")].copy()

        for subset_name, part in subsets.items():
            if len(part) < 30 or part["assistant_label"].nunique() < 2:
                continue

            idx = part["id"].map(id_to_idx)
            ok = idx.notna()
            if ok.sum() < 30:
                continue

            idx = idx[ok].astype(int).to_numpy()
            y_true = part.loc[ok, "assistant_label"].to_numpy()
            p = pred[idx]

            key = f"{label_name}_{subset_name}"
            m = macro_f1(y_true, p)
            detail[f"pair_macro_{key}"] = m

            w = weights.get(key, 0.0)
            if w and not np.isnan(m):
                scores.append((w, m))

    if not scores:
        return np.nan, detail

    wsum = sum(w for w, _ in scores)
    weighted = sum(w * m for w, m in scores) / wsum
    return float(weighted), detail


rows = []

for name, pred in variants.items():
    swap_eval = evaluate_swap_labels(name)
    pair_score, pair_detail = evaluate_pair_labels(pred)

    sel = swaps[swaps["swap_key"].isin(selected_maps[name])].copy() if selected_maps[name] else pd.DataFrame()

    mean_qprob = float(sel["qprob"].mean()) if len(sel) else np.nan
    min_qprob = float(sel["qprob"].min()) if len(sel) else np.nan
    mean_gain = float(sel["v33_pure_gain"].mean()) if len(sel) else np.nan

    labeled_precision = swap_eval["swap_label_precision"]
    labeled_precision_filled = 0.0 if np.isnan(labeled_precision) else labeled_precision

    label_n = swap_eval["swap_labeled_n"]
    label_support = min(1.0, label_n / 250.0)

    impact_bonus = min(0.018, math.log1p(max(0, swap_eval["swap_selected"])) / math.log1p(10000) * 0.018)
    small_penalty = 0.012 if 0 < swap_eval["swap_selected"] < 500 else 0.0
    huge_penalty = max(0, swap_eval["swap_selected"] - 10000) / 400000.0

    pair_score_filled = 0.0 if np.isnan(pair_score) else pair_score

    final_score = (
        0.48 * pair_score_filled
        + 0.33 * labeled_precision_filled
        + 0.10 * label_support
        + 0.07 * (mean_qprob if not np.isnan(mean_qprob) else 0.0)
        + impact_bonus
        - small_penalty
        - huge_penalty
    )

    row = {
        "variant": name,
        "final_score": float(final_score),
        "pair_weighted_macro": pair_score,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v24": int((pred != pred_anchor).sum()),
        "diff_vs_full_v33": int((pred != pred_full).sum()),
        "mean_qprob": mean_qprob,
        "min_qprob": min_qprob,
        "mean_pure_gain": mean_gain,
        **swap_eval,
    }

    for k in [
        "pair_macro_random_clean_v2_all",
        "pair_macro_manual_v1_clean",
        "pair_macro_manual_v1_high_clean",
        "pair_macro_review_v15_vs_v13_clean",
        "pair_macro_v21_active_clean",
        "pair_macro_v26_sparse_clean",
    ]:
        row[k] = pair_detail.get(k, np.nan)

    rows.append(row)

summary = pd.DataFrame(rows).sort_values(
    ["final_score", "swap_label_precision", "pair_weighted_macro"],
    ascending=False,
)

summary.to_csv(OUT_SUMMARY, index=False)

with open(OUT_MODEL, "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------
# Save top candidates
# ------------------------------------------------------------
save_names = []

for nm in summary.head(16)["variant"].tolist():
    if nm not in save_names:
        save_names.append(nm)

forced = [
    "puregain_top400",
    "puregain_top750",
    "puregain_top1000",
    "puregain_top1500",
    "puregain_top2000",
    "puregain_top3000",
    "utility_top1000",
    "utility_top2000",
    "utility_top3000",
    "qprob_top1000",
    "qprob_top2000",
    "qprob_top3000",
    "top10k_qprobge0p55_utility_top3000",
    "top10k_qprobge0p50_utility_top5000",
]

for nm in forced:
    if nm in variants and nm not in save_names:
        save_names.append(nm)

saved_rows = []

for nm in save_names[:24]:
    pred = variants[nm]
    out = SUB_DIR / f"FINAL_CANDIDATE_v33_PERFECTED_{clean_name(nm)}.csv"
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

    r = summary[summary["variant"] == nm].iloc[0].to_dict()
    r["file"] = str(out)
    saved_rows.append(r)
    print("saved:", out)

saved = pd.DataFrame(saved_rows)
saved.to_csv(OUT_SAVED, index=False)

print("\nMODEL DIAGNOSTICS")
print(json.dumps(diag, indent=2))

print("\nTOP SUMMARY")
cols = [
    "variant",
    "final_score",
    "pair_weighted_macro",
    "swap_selected",
    "swap_labeled_n",
    "swap_label_precision",
    "swap_top_gain_labeled_n",
    "swap_top_gain_precision",
    "swap_mid_gain_labeled_n",
    "swap_mid_gain_precision",
    "mean_qprob",
    "min_qprob",
    "diff_vs_v24",
    "ones",
    "pos_ratio",
]
print(summary[cols].head(80).to_string(index=False))

print("\nSAVED")
print(saved[["variant", "file", "final_score", "swap_selected", "swap_label_precision", "pair_weighted_macro", "diff_vs_v24"]].to_string(index=False))

print("\noutputs:")
print(OUT_SWAPS)
print(OUT_SUMMARY)
print(OUT_MODEL)
print(OUT_SAVED)
