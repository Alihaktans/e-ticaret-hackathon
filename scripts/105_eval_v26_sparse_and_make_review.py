from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

SPARSE_SCORE = ROOT / "data/processed/v26_sparse_text_scores.parquet"

OUT_EVAL = ROOT / "reports/manual_review/v26_sparse_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v26_sparse_aggregate.csv"
OUT_REVIEW = ROOT / "reports/manual_review/review_v26_sparse_additions_targets.csv"
OUT_REVIEW_SUMMARY = ROOT / "reports/manual_review/review_v26_sparse_additions_summary.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
}

PATHS = {
    "v22_base": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "v24_anchor": [
        ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
    ],
}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_csv_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()

def load_named(name, sample):
    p = first_existing(PATHS[name])
    if p is None:
        raise FileNotFoundError(name)
    print("loading", name, p)
    return load_csv_pred(p, sample)

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

def clean_variant_from_path(p):
    name = p.stem
    return name.replace("FINAL_CANDIDATE_v26_", "")

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

pred_base = load_named("v22_base", sample)
pred_v24 = load_named("v24_anchor", sample)

# Load all v26 sparse candidates
candidates = {
    "v22_base_public080": pred_base,
    "v24_anchor": pred_v24,
}

for p in sorted((ROOT / "submissions").glob("FINAL_CANDIDATE_v26_sparse_*.csv")):
    name = clean_variant_from_path(p)
    print("loading candidate", name)
    candidates[name] = load_csv_pred(p, sample)

print("candidate count:", len(candidates))

# Full summary
full_rows = []
base_pos = float(pred_base.mean())
v24_pos = float(pred_v24.mean())

for name, pred in candidates.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_base": int((pred != pred_base).sum()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
    })

full = pd.DataFrame(full_rows)

# Validation
eval_rows = []

for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label:", label_set)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)

    if "assistant_label" not in lab.columns:
        continue

    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)

    subsets = {"all": lab}

    if "needs_recheck" in lab.columns:
        nr = lab["needs_recheck"].fillna(0).astype(str).str.replace(".0", "", regex=False)
        nr = pd.to_numeric(nr, errors="coerce").fillna(0).astype(int)
        subsets["clean"] = lab[nr == 0].copy()

    if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        nr = lab["needs_recheck"].fillna(0).astype(str).str.replace(".0", "", regex=False)
        nr = pd.to_numeric(nr, errors="coerce").fillna(0).astype(int)
        subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()
        subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()

    if label_set in ["v20_active", "v21_active"] and "review_bucket" in lab.columns:
        for b, g in lab.groupby("review_bucket"):
            if len(g) >= 20 and g["assistant_label"].nunique() >= 2:
                subsets[f"bucket_{b}"] = g.copy()

    for subset_name, part in subsets.items():
        if len(part) < 30 or part["assistant_label"].nunique() < 2:
            continue

        mapped = part["id"].map(id_to_idx)
        ok = mapped.notna()
        part = part.loc[ok].copy()
        idx = mapped.loc[ok].astype(int).to_numpy()
        y = part["assistant_label"].to_numpy()

        for name, pred in candidates.items():
            p = pred[idx]
            row = {
                "label_set": label_set,
                "subset": subset_name,
                "eval_key": f"{label_set}_{subset_name}",
                "variant": name,
                "n": len(part),
            }
            row.update(metrics(y, p))
            eval_rows.append(row)

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT_EVAL, index=False)

legacy_weights = {
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

active_weights = {
    "v20_active_clean": 0.05,
    "v20_active_high_medium_clean": 0.04,
    "v21_active_clean": 0.18,
    "v21_active_high_medium_clean": 0.14,
    "v21_active_bucket_v21_add5k_added_over_v20": 0.12,
    "v21_active_bucket_v21_vote_added_over_v20": 0.20,
    "v21_active_bucket_v21_vote_removed_from_v20": 0.20,
    "v21_active_bucket_v21_high_but_v18_low_risky_add": 0.10,
}

agg_rows = []

for name, g in eval_df.groupby("variant"):
    legacy_score = 0.0
    legacy_wsum = 0.0
    active_score = 0.0
    active_wsum = 0.0
    legacy_vals = []
    active_vals = []

    for _, r in g.iterrows():
        key = r["eval_key"]

        lw = legacy_weights.get(key, 0.0)
        if lw:
            legacy_score += lw * r["macro_f1"]
            legacy_wsum += lw
            legacy_vals.append(r["macro_f1"])

        aw = active_weights.get(key, 0.0)
        if aw:
            active_score += aw * r["macro_f1"]
            active_wsum += aw
            active_vals.append(r["macro_f1"])

    if legacy_wsum == 0:
        continue

    legacy_weighted = legacy_score / legacy_wsum
    active_weighted = active_score / active_wsum if active_wsum else np.nan
    combined_score = 0.72 * legacy_weighted + 0.28 * active_weighted if active_wsum else legacy_weighted

    f = full[full["variant"] == name].iloc[0]
    pos_ratio = float(f["pos_ratio"])
    diff_v24 = int(f["diff_vs_v24"])
    diff_base = int(f["diff_vs_base"])

    # v24 anchor public-aday; sparse add-only fazla şişerse cezalandır.
    pos_penalty = max(0.0, pos_ratio - v24_pos) * 1.35
    diff_penalty = max(0, diff_v24 - 6000) / 900000.0

    public_safe_score = combined_score - pos_penalty - diff_penalty

    main = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "public_safe_score": public_safe_score,
        "combined_score": combined_score,
        "legacy_weighted": legacy_weighted,
        "active_weighted": active_weighted,
        "legacy_min_macro": float(np.min(legacy_vals)) if legacy_vals else np.nan,
        "active_min_macro": float(np.min(active_vals)) if active_vals else np.nan,
        "main_min_macro": float(main["macro_f1"].min()) if len(main) else np.nan,
        "main_mean_macro": float(main["macro_f1"].mean()) if len(main) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "eval_count": int(len(g)),
        "ones": int(f["ones"]),
        "pos_ratio": pos_ratio,
        "diff_vs_base": diff_base,
        "diff_vs_v24": diff_v24,
    })

agg = pd.DataFrame(agg_rows)
agg = agg.sort_values(["public_safe_score", "combined_score", "legacy_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nTOP PUBLIC SAFE")
print(agg.head(80).to_string(index=False))

print("\nTOP COMBINED")
print(agg.sort_values("combined_score", ascending=False).head(80).to_string(index=False))

# Review target generation
print("creating sparse review targets...")

score = pd.read_parquet(SPARSE_SCORE)
score["id"] = score["id"].astype(str)

pairs = pd.read_csv(PAIRS)
pairs["id"] = pairs["id"].astype(str)
pairs["term_id"] = pairs["term_id"].astype(str)
pairs["item_id"] = pairs["item_id"].astype(str)

terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)
for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

base_df = pd.DataFrame({
    "id": ids,
    "pred_v24": pred_v24,
})

# helper load candidate by variant name
def get_candidate(name):
    p = ROOT / "submissions" / f"FINAL_CANDIDATE_v26_{name}.csv"
    if not p.exists():
        return None
    return load_csv_pred(p, sample)

cand_names = [
    "sparse_addonly_supported_budget500",
    "sparse_addonly_supported_budget1000",
    "sparse_addonly_supported_budget2000",
    "sparse_addonly_supported_budget3000",
    "sparse_addonly_budget3000",
    "sparse_addonly_strictpct3_budget3000",
    "sparse_addonly_supported_budget5000",
    "sparse_addonly_budget5000",
]

for nm in cand_names:
    arr = get_candidate(nm)
    if arr is not None:
        base_df[f"pred_{nm}"] = arr

df = base_df.merge(score, on="id", how="left", validate="one_to_one")

parts = []

def add_bucket(mask, bucket, n_take=180):
    g = df[mask].copy()
    if len(g) == 0:
        return
    g["review_bucket"] = bucket
    g = g.sort_values(["sparse_blend", "sparse_pct_rank"], ascending=[False, True])
    top = g.head(n_take // 2)
    rest = g.drop(top.index)
    if len(rest) > n_take - len(top):
        rest = rest.sample(n=n_take - len(top), random_state=2026)
    parts.append(pd.concat([top, rest], ignore_index=True))

# budget katmanları
if "pred_sparse_addonly_supported_budget500" in df.columns:
    add_bucket((df.pred_v24 == 0) & (df.pred_sparse_addonly_supported_budget500 == 1),
               "supported_0_500_added", 220)

if "pred_sparse_addonly_supported_budget1000" in df.columns and "pred_sparse_addonly_supported_budget500" in df.columns:
    add_bucket((df.pred_sparse_addonly_supported_budget1000 == 1) & (df.pred_sparse_addonly_supported_budget500 == 0),
               "supported_500_1000_extra", 180)

if "pred_sparse_addonly_supported_budget2000" in df.columns and "pred_sparse_addonly_supported_budget1000" in df.columns:
    add_bucket((df.pred_sparse_addonly_supported_budget2000 == 1) & (df.pred_sparse_addonly_supported_budget1000 == 0),
               "supported_1000_2000_extra", 180)

if "pred_sparse_addonly_supported_budget3000" in df.columns and "pred_sparse_addonly_supported_budget2000" in df.columns:
    add_bucket((df.pred_sparse_addonly_supported_budget3000 == 1) & (df.pred_sparse_addonly_supported_budget2000 == 0),
               "supported_2000_3000_extra", 180)

if "pred_sparse_addonly_supported_budget5000" in df.columns and "pred_sparse_addonly_supported_budget3000" in df.columns:
    add_bucket((df.pred_sparse_addonly_supported_budget5000 == 1) & (df.pred_sparse_addonly_supported_budget3000 == 0),
               "supported_3000_5000_extra", 180)

if "pred_sparse_addonly_strictpct3_budget3000" in df.columns and "pred_sparse_addonly_supported_budget3000" in df.columns:
    add_bucket((df.pred_sparse_addonly_strictpct3_budget3000 == 1) & (df.pred_sparse_addonly_supported_budget3000 == 0),
               "strictpct3_extra_not_supported3000", 150)

if "pred_sparse_addonly_budget5000" in df.columns and "pred_sparse_addonly_supported_budget5000" in df.columns:
    add_bucket((df.pred_sparse_addonly_budget5000 == 1) & (df.pred_sparse_addonly_supported_budget5000 == 0),
               "raw_sparse_extra_not_supported5000", 150)

review = pd.concat(parts, ignore_index=True).drop_duplicates("id").reset_index(drop=True)

review = (
    review
    .merge(pairs[["id", "term_id", "item_id"]], on="id", how="left")
    .merge(terms[["term_id", "query"]], on="term_id", how="left")
    .merge(items[["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]], on="item_id", how="left")
)

review["assistant_label"] = ""
review["assistant_confidence"] = ""
review["needs_recheck"] = ""
review["reason"] = ""

keep_cols = [
    "id", "review_bucket", "query", "title", "category", "brand", "gender", "age_group",
    "sparse_word", "sparse_char", "sparse_blend", "sparse_rank", "sparse_pct_rank",
    "sparse_delta_from_max",
    "assistant_label", "assistant_confidence", "needs_recheck", "reason",
    "term_id", "item_id", "attributes",
]

review = review[keep_cols]
review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

summary = (
    review.groupby("review_bucket")
    .agg(
        rows=("id", "count"),
        sparse_mean=("sparse_blend", "mean"),
        sparse_min=("sparse_blend", "min"),
        pct_rank_mean=("sparse_pct_rank", "mean"),
    )
    .reset_index()
)

summary.to_csv(OUT_REVIEW_SUMMARY, index=False)

print("\nREVIEW SUMMARY")
print(summary.to_string(index=False))

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_REVIEW)
print("saved:", OUT_REVIEW_SUMMARY)
