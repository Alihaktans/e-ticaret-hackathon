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
    "vote_public080": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "vote_addonly": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_addonly.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v22_v20_plus_vote_addonly.csv",
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

OUT_EVAL = ROOT / "reports/manual_review/v23_vote_refinement_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v23_vote_refinement_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v23_vote_refinement_full_summary.csv"

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
        .replace("+", "plus")
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
    )

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

    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {name}")

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

def restore_budget(base, restore_order, budget):
    pred = base.copy()
    take = restore_order[:min(budget, len(restore_order))]
    pred[take] = 1
    return pred

def trim_budget(base, trim_order, budget):
    pred = base.copy()
    take = trim_order[:min(budget, len(trim_order))]
    pred[take] = 0
    return pred

def swap_budget(base, restore_order, trim_order, budget):
    pred = base.copy()

    add_take = restore_order[:min(budget, len(restore_order))]
    rem_take = trim_order[:min(budget, len(trim_order))]

    pred[add_take] = 1
    pred[rem_take] = 0

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
pred_vote = load_sub("vote_public080", sample)
pred_addonly = load_sub("vote_addonly", sample, required=False)

if pred_addonly is None:
    pred_addonly = pred_vote.copy()

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

v18_filled = np.where(v18_score >= 0, v18_score, 0.0).astype("float32")
v5_filled = np.where(v5_score >= 0, v5_score, 0.0).astype("float32")

variants = {}

add_variant(variants, "v16_current", pred_v16)
add_variant(variants, "v19p1_current", pred_v19)
add_variant(variants, "v20_current", pred_v20)
add_variant(variants, "v21_add5k_existing", pred_add5)
add_variant(variants, "v22_vote_public080_base", pred_vote)
add_variant(variants, "v22_vote_addonly", pred_addonly)

# ------------------------------------------------------------------
# Candidate groups
# ------------------------------------------------------------------

# Vote full'un v20'den sildiği yerler: geri ekleme adayları.
restore_mask = (pred_vote == 0) & (pred_v20 == 1)

# Vote full'un v20 üstüne eklediği yerler: temizleme adayları.
trim_mask = (pred_vote == 1) & (pred_v20 == 0)

# Restore score: eski güçlü modeller destekliyorsa geri ekle.
restore_score = (
    0.50 * v5_filled
    + 0.30 * v18_filled
    + 0.10 * pred_v19.astype("float32")
    + 0.10 * pred_add5.astype("float32")
)

restore_idxs = np.where(restore_mask)[0]
restore_order = restore_idxs[np.argsort(-restore_score[restore_idxs])] if len(restore_idxs) else np.array([], dtype=int)

# Trim score: düşükse vote'un eklediği pozitiflerden çıkar.
# Eklemeyi tutmak için destek skoru.
add_keep_score = (
    0.45 * v21_score
    + 0.25 * v5_filled
    + 0.20 * v18_filled
    + 0.10 * pred_add5.astype("float32")
)

trim_idxs = np.where(trim_mask)[0]
trim_order = trim_idxs[np.argsort(add_keep_score[trim_idxs])] if len(trim_idxs) else np.array([], dtype=int)

print("restore candidates:", len(restore_idxs))
print("trim candidates:", len(trim_idxs))

# ------------------------------------------------------------------
# 1) Restore-only around public 0.80
# ------------------------------------------------------------------
for budget in [500, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 16000, 24000]:
    add_variant(
        variants,
        f"v23_vote_restore_blend_budget{budget}",
        restore_budget(pred_vote, restore_order, budget),
    )

# Threshold-based restore
for th in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    mask = restore_mask & (v5_filled >= th)
    pred = pred_vote.copy()
    pred[mask] = 1
    add_variant(variants, f"v23_vote_restore_v5ge{th}", pred)

for th in [0.30, 0.50, 0.70, 0.90]:
    mask = restore_mask & (v18_score >= th)
    pred = pred_vote.copy()
    pred[mask] = 1
    add_variant(variants, f"v23_vote_restore_v18ge{th}", pred)

combo_specs = [
    ("v5ge080_v18ge020", (v5_filled >= 0.80) & (v18_score >= 0.20)),
    ("v5ge085_v18ge010", (v5_filled >= 0.85) & (v18_score >= 0.10)),
    ("v5ge090_or_v18ge070", (v5_filled >= 0.90) | (v18_score >= 0.70)),
    ("v5ge080_or_v18ge090", (v5_filled >= 0.80) | (v18_score >= 0.90)),
]

for name, cond in combo_specs:
    pred = pred_vote.copy()
    pred[restore_mask & cond] = 1
    add_variant(variants, f"v23_vote_restore_{name}", pred)

# ------------------------------------------------------------------
# 2) Trim-only around public 0.80
# ------------------------------------------------------------------
for budget in [500, 1000, 1500, 2000, 3000, 5000, 8000, 12000]:
    add_variant(
        variants,
        f"v23_vote_trim_lowadd_budget{budget}",
        trim_budget(pred_vote, trim_order, budget),
    )

# Remove only clearly risky additions
risky_add_specs = [
    ("v5lt005_v18lt010", (v5_filled < 0.05) & ((v18_score < 0.10) | (v18_score < 0))),
    ("v5lt010_v18lt010", (v5_filled < 0.10) & ((v18_score < 0.10) | (v18_score < 0))),
    ("v5lt020_v18lt010", (v5_filled < 0.20) & ((v18_score < 0.10) | (v18_score < 0))),
    ("v21lt050_v5lt020", (v21_score < 0.50) & (v5_filled < 0.20)),
    ("v21lt020_v5lt030", (v21_score < 0.20) & (v5_filled < 0.30)),
]

for name, cond in risky_add_specs:
    pred = pred_vote.copy()
    pred[trim_mask & cond] = 0
    add_variant(variants, f"v23_vote_trim_{name}", pred)

# ------------------------------------------------------------------
# 3) Swap: same pos_ratio, better composition
# ------------------------------------------------------------------
for budget in [500, 1000, 2000, 3000, 5000, 8000, 12000, 16000]:
    add_variant(
        variants,
        f"v23_vote_swap_restore_trim_budget{budget}",
        swap_budget(pred_vote, restore_order, trim_order, budget),
    )

# ------------------------------------------------------------------
# 4) Union with add5k, but controlled
# ------------------------------------------------------------------
add5_extra = (pred_add5 == 1) & (pred_vote == 0)

pred = pred_vote.copy()
pred[add5_extra] = 1
add_variant(variants, "v23_vote_union_add5k_all", pred)

safe_add5 = add5_extra & ~((v18_score >= 0) & (v18_score <= 0.10) & (v5_filled >= 0.40))
pred = pred_vote.copy()
pred[safe_add5] = 1
add_variant(variants, "v23_vote_union_add5k_block_v18low_v5ge040", pred)

safe_add5_2 = add5_extra & ((v5_filled >= 0.25) | (v18_score >= 0.50) | (v21_score >= 0.99999))
pred = pred_vote.copy()
pred[safe_add5_2] = 1
add_variant(variants, "v23_vote_union_add5k_supported", pred)

# Add5 budgeted union
add5_idxs = np.where(add5_extra)[0]
add5_order = add5_idxs[np.argsort(-v21_score[add5_idxs])] if len(add5_idxs) else np.array([], dtype=int)

for budget in [500, 1000, 2000, 3000, 5000, 8000]:
    pred = pred_vote.copy()
    take = add5_order[:min(budget, len(add5_order))]
    pred[take] = 1
    add_variant(variants, f"v23_vote_union_add5k_budget{budget}", pred)

print("variants:", len(variants))

# ------------------------------------------------------------------
# Full summary
# ------------------------------------------------------------------
full_rows = []
for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v16": int((pred != pred_v16).sum()),
        "diff_vs_v19p1": int((pred != pred_v19).sum()),
        "diff_vs_v20": int((pred != pred_v20).sum()),
        "diff_vs_vote_base": int((pred != pred_vote).sum()),
        "diff_vs_add5": int((pred != pred_add5).sum()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
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
    "v20_active_clean": 0.06,
    "v20_active_high_medium_clean": 0.05,
    "v21_active_clean": 0.20,
    "v21_active_high_medium_clean": 0.16,
    "v21_active_bucket_v21_add5k_added_over_v20": 0.15,
    "v21_active_bucket_v21_vote_added_over_v20": 0.22,
    "v21_active_bucket_v21_vote_removed_from_v20": 0.18,
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

print("\nFULL DIFF VS BASE")
print(full.sort_values("diff_vs_vote_base").to_string(index=False))

# Save top candidates
top_names = []
for col in ["combined_score", "legacy_weighted", "active_weighted", "main_min_macro"]:
    for nm in agg.sort_values(col, ascending=False).head(8)["variant"].tolist():
        if nm not in top_names:
            top_names.append(nm)

for nm in ["v22_vote_public080_base", "v23_vote_restore_blend_budget3000", "v23_vote_restore_blend_budget5000"]:
    if nm in variants and nm not in top_names:
        top_names.append(nm)

for name in top_names[:30]:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v23_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
