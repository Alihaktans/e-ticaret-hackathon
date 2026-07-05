from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
V21 = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"

OUT_EVAL = ROOT / "reports/manual_review/v24_grid_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v24_grid_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v24_grid_full_summary.csv"

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
    "add5": [
        ROOT / "submissions/FINAL_MAIN_v21_v20_add5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add5000.csv",
    ],
    "vote_base": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "swap5": [
        ROOT / "submissions/FINAL_MAIN_v23_vote_swap5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_swap_restore_trim_budget5000.csv",
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
        .replace(":", "")
    )

def load_sub(name, sample):
    p = first_existing(PATHS[name])
    if p is None:
        raise FileNotFoundError(f"missing {name}")
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

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

pred_v16 = load_sub("v16", sample)
pred_v19 = load_sub("v19p1", sample)
pred_v20 = load_sub("v20", sample)
pred_add5 = load_sub("add5", sample)
pred_vote = load_sub("vote_base", sample)
pred_swap5 = load_sub("swap5", sample)

print("loading scores...")
v21 = pd.read_parquet(V21)
v21["id"] = v21["id"].astype(str)
assert v21["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
v21_score = v21["v21_score"].astype("float32").to_numpy()

v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
v5["id"] = v5["id"].astype(str)
assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
v5_score = v5["proba_avg"].astype("float32").to_numpy()

v18_score = np.zeros(n, dtype=np.float32)
if V18.exists():
    v18 = pd.read_parquet(V18)
    v18["id"] = v18["id"].astype(str)
    idx = v18["id"].map(id_to_idx)
    ok = idx.notna()
    v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()

# Candidate pools
restore_pool = np.where((pred_vote == 0) & (pred_v20 == 1))[0]
trim_pool = np.where((pred_vote == 1) & (pred_v20 == 0))[0]

print("restore_pool:", len(restore_pool))
print("trim_pool:", len(trim_pool))

variants = {}
add_variant(variants, "v22_vote_public080_base", pred_vote)
add_variant(variants, "v23_swap5000_current", pred_swap5)
add_variant(variants, "v20_current", pred_v20)
add_variant(variants, "v21_add5k_existing", pred_add5)

# Support scores
v5f = np.clip(v5_score, 0, 1)
v18f = np.clip(v18_score, 0, 1)
v21f = np.clip(v21_score, 0, 1)

restore_formulas = {
    "rA": 0.55*v5f + 0.35*v18f + 0.05*pred_v19 + 0.05*pred_add5,
    "rB": 0.40*v5f + 0.45*v18f + 0.10*pred_v19 + 0.05*pred_add5,
    "rC": 0.65*v5f + 0.25*v18f + 0.05*pred_v19 + 0.05*pred_add5,
    "rD": 0.50*v5f + 0.30*v18f + 0.10*pred_v19 + 0.10*pred_add5,
    "rE": 0.35*v5f + 0.50*v18f + 0.05*v21f + 0.10*pred_add5,
}

# keep_score düşükse trimlenecek
trim_formulas = {
    "tA": 0.45*v21f + 0.30*v5f + 0.20*v18f + 0.05*pred_add5,
    "tB": 0.30*v21f + 0.40*v5f + 0.20*v18f + 0.10*pred_add5,
    "tC": 0.55*v21f + 0.20*v5f + 0.20*v18f + 0.05*pred_add5,
    "tD": 0.35*v21f + 0.25*v5f + 0.35*v18f + 0.05*pred_add5,
    "tE": 0.25*v21f + 0.45*v5f + 0.25*v18f + 0.05*pred_v19,
}

restore_budgets = [2000, 3000, 4000, 5000, 6500, 8000, 10000, 12000, 15000, 18000]
trim_budgets = [2000, 3000, 4000, 5000, 6500, 8000, 10000, 12000, 15000]

for rname, rscore in restore_formulas.items():
    r_order = restore_pool[np.argsort(-rscore[restore_pool])]

    for rb in restore_budgets:
        pred = pred_vote.copy()
        pred[r_order[:min(rb, len(r_order))]] = 1
        add_variant(variants, f"v24_restore_{rname}_b{rb}", pred)

for tname, tscore in trim_formulas.items():
    t_order = trim_pool[np.argsort(tscore[trim_pool])]

    for tb in trim_budgets:
        pred = pred_vote.copy()
        pred[t_order[:min(tb, len(t_order))]] = 0
        add_variant(variants, f"v24_trim_{tname}_b{tb}", pred)

# Swap grid: positive ratio same kalır
for rname, rscore in restore_formulas.items():
    r_order = restore_pool[np.argsort(-rscore[restore_pool])]

    for tname, tscore in trim_formulas.items():
        t_order = trim_pool[np.argsort(tscore[trim_pool])]

        for b in [2000, 3000, 4000, 5000, 6500, 8000, 10000, 12000, 15000]:
            pred = pred_vote.copy()
            pred[r_order[:min(b, len(r_order))]] = 1
            pred[t_order[:min(b, len(t_order))]] = 0
            add_variant(variants, f"v24_swap_{rname}_{tname}_b{b}", pred)

# Asymmetric: public 0.80 base üstüne hafif pozitif artırma/azaltma
for rname, rscore in restore_formulas.items():
    r_order = restore_pool[np.argsort(-rscore[restore_pool])]

    for tname, tscore in trim_formulas.items():
        t_order = trim_pool[np.argsort(tscore[trim_pool])]

        for rb, tb in [
            (6500, 5000),
            (8000, 5000),
            (10000, 6500),
            (12000, 8000),
            (15000, 10000),
            (5000, 6500),
            (6500, 8000),
            (8000, 10000),
        ]:
            pred = pred_vote.copy()
            pred[r_order[:min(rb, len(r_order))]] = 1
            pred[t_order[:min(tb, len(t_order))]] = 0
            add_variant(variants, f"v24_asym_{rname}_{tname}_r{rb}_t{tb}", pred)

print("variants:", len(variants))

# Full summary
full_rows = []
for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_vote_base": int((pred != pred_vote).sum()),
        "diff_vs_swap5": int((pred != pred_swap5).sum()),
        "diff_vs_v20": int((pred != pred_v20).sum()),
        "diff_vs_add5": int((pred != pred_add5).sum()),
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

BASE_POS = pred_vote.mean()

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
    pos_penalty = abs(float(f["pos_ratio"]) - BASE_POS) * 1.20
    diff_penalty = max(0, int(f["diff_vs_vote_base"]) - 20000) / 1000000.0

    # public 0.80 anchor: çok uzaklaşanları hafif cezalandır.
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
    })

agg = pd.DataFrame(agg_rows)
agg = agg.merge(full, on="variant", how="left")
agg = agg.sort_values(["public_safe_score", "combined_score", "legacy_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nTOP PUBLIC SAFE")
print(agg.head(100).to_string(index=False))

print("\nTOP COMBINED")
print(agg.sort_values("combined_score", ascending=False).head(80).to_string(index=False))

print("\nTOP LEGACY")
print(agg.sort_values("legacy_weighted", ascending=False).head(80).to_string(index=False))

print("\nFULL NEAR BASE")
near = full[full["diff_vs_vote_base"] <= 30000].sort_values("diff_vs_vote_base")
print(near.head(120).to_string(index=False))

# Save top candidates
top_names = []
for col in ["public_safe_score", "combined_score", "legacy_weighted", "active_weighted", "main_min_macro"]:
    for nm in agg.sort_values(col, ascending=False).head(12)["variant"].tolist():
        if nm not in top_names:
            top_names.append(nm)

for nm in ["v22_vote_public080_base", "v23_swap5000_current"]:
    if nm in variants and nm not in top_names:
        top_names.append(nm)

for name in top_names[:40]:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v24_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
