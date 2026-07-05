from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
V21 = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"

PATHS = {
    "v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
    "v19p1": [
        ROOT / "submissions/FINAL_MAIN_v19p1_honest_addonly_ba012_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba012_r1_v18th0p9.csv",
    ],
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
    ],
    "v21_add5k": [
        ROOT / "submissions/FINAL_MAIN_v21_v20_add5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add5000.csv",
    ],
    "v21_add8k": [
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add8000.csv",
    ],
    "v21_add12k": [
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add12000.csv",
    ],
    "v21_vote": [
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_vote_v16_v17_v21_2of3_323.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_vote_v16_v17_v21_2of3_325.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_vote_v16_v17_v21_2of3_032.csv",
    ],
}

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
}

OUT_EVAL = ROOT / "reports/manual_review/v22_addonly_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v22_addonly_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v22_addonly_full_summary.csv"

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def clean_name(x):
    return x.replace(".", "p").replace(" ", "_").replace("/", "_").replace("+", "plus")

def load_sub(name, sample, required=True):
    p = first_existing(PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(f"missing {name}")
        print("missing optional:", name)
        return None

    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    assert d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), name
    return d["prediction"].astype(np.int8).to_numpy()

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

def add_variant(variants, name, pred):
    variants[name] = pred.astype(np.int8)

def add_only(base, source):
    pred = base.copy()
    pred[(base == 0) & (source == 1)] = 1
    return pred

def add_masked(base, add_mask):
    pred = base.copy()
    pred[(base == 0) & add_mask] = 1
    return pred

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

pred_v16 = load_sub("v16", sample)
pred_v19 = load_sub("v19p1", sample)
pred_v20 = load_sub("v20", sample)
pred_add5 = load_sub("v21_add5k", sample)
pred_add8 = load_sub("v21_add8k", sample, required=False)
pred_add12 = load_sub("v21_add12k", sample, required=False)
pred_vote = load_sub("v21_vote", sample)

if pred_add8 is None:
    pred_add8 = pred_add5.copy()
if pred_add12 is None:
    pred_add12 = pred_add5.copy()

print("loading scores...")
v21 = pd.read_parquet(V21)
v21["id"] = v21["id"].astype(str)
assert v21["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
v21_score = v21["v21_score"].astype("float32").to_numpy()

v5_score = np.full(n, -1.0, dtype=np.float32)
if V5.exists():
    v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
    v5["id"] = v5["id"].astype(str)
    assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
    v5_score = v5["proba_avg"].astype("float32").to_numpy()

v18_score = np.full(n, -1.0, dtype=np.float32)
if V18.exists():
    v18 = pd.read_parquet(V18)
    v18["id"] = v18["id"].astype(str)
    idx = v18["id"].map(id_to_idx)
    ok = idx.notna()
    v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()

variants = {}

add_variant(variants, "v16_current", pred_v16)
add_variant(variants, "v19p1_current", pred_v19)
add_variant(variants, "v20_current", pred_v20)
add_variant(variants, "v21_add5k_existing", pred_add5)
add_variant(variants, "v21_add8k_existing", pred_add8)
add_variant(variants, "v21_add12k_existing", pred_add12)
add_variant(variants, "v21_vote_full_risky", pred_vote)

# Add masks
add5_mask = (pred_v20 == 0) & (pred_add5 == 1)
add8_mask = (pred_v20 == 0) & (pred_add8 == 1)
add12_mask = (pred_v20 == 0) & (pred_add12 == 1)
vote_add_mask = (pred_v20 == 0) & (pred_vote == 1)

# Risk block: v21 çok yüksek ama v18 açıkça düşük ve v5 orta/yüksekse review'da kötü çıktı.
risky_v18_low = (v18_score >= 0) & (v18_score <= 0.10) & (v5_score >= 0.40)
risky_v18_low_strict = (v18_score >= 0) & (v18_score <= 0.10) & (v5_score >= 0.55)

# Core v22
add_variant(variants, "v22_v20_plus_vote_addonly", add_masked(pred_v20, vote_add_mask))
add_variant(variants, "v22_v20_plus_add5k_plus_vote_addonly", add_masked(pred_v20, add5_mask | vote_add_mask))
add_variant(variants, "v22_v20_plus_add5k_plus_vote_block_v18low_v5ge040", add_masked(pred_v20, (add5_mask | vote_add_mask) & ~risky_v18_low))
add_variant(variants, "v22_v20_plus_add5k_plus_vote_block_v18low_v5ge055", add_masked(pred_v20, (add5_mask | vote_add_mask) & ~risky_v18_low_strict))

# Add8/Add12 cautious tests
add_variant(variants, "v22_v20_plus_add8k_plus_vote_block_v18low_v5ge040", add_masked(pred_v20, (add8_mask | vote_add_mask) & ~risky_v18_low))
add_variant(variants, "v22_v20_plus_add12k_plus_vote_block_v18low_v5ge040", add_masked(pred_v20, (add12_mask | vote_add_mask) & ~risky_v18_low))

# Vote + subset of add12 extra only if old models are not hostile.
extra12 = add12_mask & ~add5_mask
extra12_safer = extra12 & ((v5_score >= 0.25) | ((v18_score >= 0) & (v18_score >= 0.50)))
add_variant(variants, "v22_v20_plus_add5k_vote_plus_extra12_safe", add_masked(pred_v20, add5_mask | vote_add_mask | extra12_safer))

# Budgeted vote additions ordered by v21 score
vote_idxs = np.where(vote_add_mask)[0]
vote_order = vote_idxs[np.argsort(-v21_score[vote_idxs])] if len(vote_idxs) else np.array([], dtype=int)

for budget in [250, 500, 750, 1000, 1500, 2000, 3000, 5000]:
    take = vote_order[:min(budget, len(vote_order))]
    mask = add5_mask.copy()
    mask[take] = True
    add_variant(variants, f"v22_v20_add5k_plus_vote_budget{budget}", add_masked(pred_v20, mask))

# Budgeted add expansion beyond add5, from add12 extra
extra_idxs = np.where(extra12 & ~risky_v18_low)[0]
extra_order = extra_idxs[np.argsort(-v21_score[extra_idxs])] if len(extra_idxs) else np.array([], dtype=int)

for budget in [500, 1000, 2000, 3000, 5000, 7000]:
    take = extra_order[:min(budget, len(extra_order))]
    mask = add5_mask | vote_add_mask
    mask = mask.copy()
    mask[take] = True
    add_variant(variants, f"v22_v20_add5k_vote_plus_extra12budget{budget}", add_masked(pred_v20, mask))

print("variants:", len(variants))

# Full summary
full_rows = []
for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v16": int((pred != pred_v16).sum()),
        "diff_vs_v16_ratio": float((pred != pred_v16).mean()),
        "diff_vs_v19p1": int((pred != pred_v19).sum()),
        "diff_vs_v19p1_ratio": float((pred != pred_v19).mean()),
        "diff_vs_v20": int((pred != pred_v20).sum()),
        "diff_vs_v20_ratio": float((pred != pred_v20).mean()),
        "diff_vs_add5": int((pred != pred_add5).sum()),
        "diff_vs_vote_full": int((pred != pred_vote).sum()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

# Validation
eval_rows = []

for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label:", label_set, path)
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

        for name, pred in variants.items():
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
    "v20_active_clean": 0.08,
    "v20_active_high_medium_clean": 0.06,
    "v21_active_clean": 0.25,
    "v21_active_high_medium_clean": 0.20,
    "v21_active_bucket_v21_add5k_added_over_v20": 0.20,
    "v21_active_bucket_v21_vote_added_over_v20": 0.25,
    "v21_active_bucket_v21_vote_removed_from_v20": 0.15,
    "v21_active_bucket_v21_add12k_extra_over_add5k": 0.10,
    "v21_active_bucket_v21_high_but_v18_low_risky_add": 0.12,
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
    combined_score = 0.70 * legacy_weighted + 0.30 * active_weighted if active_wsum else legacy_weighted

    main = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "combined_score": combined_score,
        "legacy_weighted": legacy_weighted,
        "active_weighted": active_weighted,
        "legacy_min_macro": float(np.min(legacy_vals)) if legacy_vals else np.nan,
        "active_min_macro": float(np.min(active_vals)) if active_vals else np.nan,
        "main_min_macro": float(main["macro_f1"].min()) if len(main) else np.nan,
        "main_mean_macro": float(main["macro_f1"].mean()) if len(main) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "mean_pred_pos_ratio": float(g["pred_pos_ratio"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows)
agg = agg.merge(full, on="variant", how="left")
agg = agg.sort_values(["combined_score", "legacy_weighted", "active_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP COMBINED")
print(agg.head(80).to_string(index=False))

print("\nAGG TOP LEGACY")
print(agg.sort_values("legacy_weighted", ascending=False).head(80).to_string(index=False))

print("\nFULL")
print(full.sort_values("diff_vs_v20").to_string(index=False))

# save top candidates
top_names = []
for col in ["combined_score", "legacy_weighted", "active_weighted", "main_min_macro"]:
    for nm in agg.sort_values(col, ascending=False).head(8)["variant"].tolist():
        if nm not in top_names:
            top_names.append(nm)

for nm in ["v22_v20_plus_add5k_plus_vote_addonly", "v22_v20_plus_vote_addonly", "v21_add5k_existing", "v20_current"]:
    if nm in variants and nm not in top_names:
        top_names.append(nm)

for name in top_names[:25]:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v22_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
