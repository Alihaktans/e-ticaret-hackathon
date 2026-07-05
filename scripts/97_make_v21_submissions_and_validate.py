from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
V21 = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"

SUB_PATHS = {
    "v13": [
        ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
    ],
    "v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
    "v19p1": [
        ROOT / "submissions/FINAL_MAIN_v19p1_honest_addonly_ba012_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba012_r1_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba008_r1_v18th0p9.csv",
    ],
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
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
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
}

OUT_EVAL = ROOT / "reports/manual_review/v21_submission_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v21_submission_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v21_submission_full_summary.csv"

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
        .replace("+", "plus")
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

def load_sub(name, sample, required=True):
    p = first_existing(SUB_PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(f"Missing submission for {name}")
        print("missing optional:", name)
        return None
    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    assert d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), name
    return d["prediction"].astype(np.int8).to_numpy()

def add_variant(variants, name, pred):
    variants[name] = pred.astype(np.int8)

def make_global_top(score, target_ratio):
    n = len(score)
    k = int(round(n * target_ratio))
    pred = np.zeros(n, dtype=np.int8)
    if k <= 0:
        return pred
    order = np.argsort(-score)
    pred[order[:k]] = 1
    return pred

def make_rank_pct(df, pct, score_floor):
    pred = (
        (df["v21_pct_rank"].to_numpy() <= pct)
        & (df["v21_score"].to_numpy() >= score_floor)
    ).astype(np.int8)
    return pred

def make_adaptive_top(df, pct, min_k, max_k, score_floor):
    pred = np.zeros(len(df), dtype=np.int8)

    term_ids = df["term_id"].to_numpy()
    counts = df["candidate_count"].to_numpy()
    ranks = df["v21_rank"].to_numpy()
    scores = df["v21_score"].to_numpy()

    k = np.ceil(counts * pct).astype(np.int32)
    k = np.maximum(k, min_k)
    k = np.minimum(k, max_k)

    pred[(ranks <= k) & (scores >= score_floor)] = 1
    return pred

def patch_add(base, score, budget):
    pred = base.copy()
    idxs = np.where(base == 0)[0]
    if len(idxs) == 0:
        return pred
    order = idxs[np.argsort(-score[idxs])]
    take = order[:min(budget, len(order))]
    pred[take] = 1
    return pred

def patch_rem(base, score, budget):
    pred = base.copy()
    idxs = np.where(base == 1)[0]
    if len(idxs) == 0:
        return pred
    order = idxs[np.argsort(score[idxs])]
    take = order[:min(budget, len(order))]
    pred[take] = 0
    return pred

def patch_swap(base, score, budget):
    pred = base.copy()

    add_idxs = np.where(base == 0)[0]
    rem_idxs = np.where(base == 1)[0]

    add_order = add_idxs[np.argsort(-score[add_idxs])]
    rem_order = rem_idxs[np.argsort(score[rem_idxs])]

    add_take = add_order[:min(budget, len(add_order))]
    rem_take = rem_order[:min(budget, len(rem_order))]

    pred[add_take] = 1
    pred[rem_take] = 0

    return pred

def patch_asym(base, score, add_budget, rem_budget):
    pred = base.copy()

    if add_budget > 0:
        add_idxs = np.where(base == 0)[0]
        add_order = add_idxs[np.argsort(-score[add_idxs])]
        pred[add_order[:min(add_budget, len(add_order))]] = 1

    if rem_budget > 0:
        rem_idxs = np.where(base == 1)[0]
        rem_order = rem_idxs[np.argsort(score[rem_idxs])]
        pred[rem_order[:min(rem_budget, len(rem_order))]] = 0

    return pred

def score_report(name, score):
    return {
        "name": name,
        "mean": float(np.mean(score)),
        "std": float(np.std(score)),
        "min": float(np.min(score)),
        "p01": float(np.quantile(score, 0.01)),
        "p05": float(np.quantile(score, 0.05)),
        "p10": float(np.quantile(score, 0.10)),
        "p25": float(np.quantile(score, 0.25)),
        "p50": float(np.quantile(score, 0.50)),
        "p75": float(np.quantile(score, 0.75)),
        "p90": float(np.quantile(score, 0.90)),
        "p95": float(np.quantile(score, 0.95)),
        "p99": float(np.quantile(score, 0.99)),
        "max": float(np.max(score)),
    }

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

pred_v13 = load_sub("v13", sample)
pred_v16 = load_sub("v16", sample)
pred_v19 = load_sub("v19p1", sample, required=False)
pred_v20 = load_sub("v20", sample, required=False)
pred_v17 = load_sub("v17p4", sample, required=False)

if pred_v19 is None:
    pred_v19 = pred_v16.copy()
if pred_v20 is None:
    pred_v20 = pred_v19.copy()
if pred_v17 is None:
    pred_v17 = pred_v16.copy()

print("loading v21...")
v21 = pd.read_parquet(V21)
v21["id"] = v21["id"].astype(str)

assert v21["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), "v21 id order mismatch"

v21_score = v21["v21_score"].astype("float32").to_numpy()

print("loading v5...")
v5_score = np.full(n, -1.0, dtype=np.float32)
if V5.exists():
    v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
    v5["id"] = v5["id"].astype(str)
    assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
    v5_score = v5["proba_avg"].astype("float32").to_numpy()

print("loading v18...")
v18_score = np.full(n, -1.0, dtype=np.float32)
if V18.exists():
    v18 = pd.read_parquet(V18)
    v18["id"] = v18["id"].astype(str)
    idx = v18["id"].map(id_to_idx)
    ok = idx.notna()
    v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()

print("adding v21 ranks...")
df = v21[["id", "term_id", "item_id", "v21_score"]].copy()
df["candidate_count"] = df.groupby("term_id")["id"].transform("size").astype("int32")
df["v21_rank"] = df.groupby("term_id")["v21_score"].rank(method="first", ascending=False).astype("int32")
df["v21_pct_rank"] = (df["v21_rank"] / df["candidate_count"]).astype("float32")

v21_rank = df["v21_rank"].to_numpy()
v21_pct = df["v21_pct_rank"].to_numpy()

print("v21 score report:")
print(pd.DataFrame([score_report("v21", v21_score)]).to_string(index=False))

variants = {}

add_variant(variants, "v13_public_0p77", pred_v13)
add_variant(variants, "v16_current", pred_v16)
add_variant(variants, "v19p1_current", pred_v19)
add_variant(variants, "v20_current", pred_v20)
add_variant(variants, "v17p4_reference_risky", pred_v17)

# ------------------------------------------------------------
# 1) Pure v21 global target-ratio variants
# ------------------------------------------------------------
for ratio in [0.24, 0.26, 0.28, 0.30, 0.31, 0.32, 0.323, 0.325, 0.33, 0.34, 0.35, 0.36]:
    pred = make_global_top(v21_score, ratio)
    add_variant(variants, f"v21_global_top_ratio{ratio}", pred)

# ------------------------------------------------------------
# 2) Pure v21 query-rank variants
# ------------------------------------------------------------
for pct in [0.15, 0.20, 0.25, 0.30, 0.32, 0.35, 0.40]:
    for floor in [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]:
        pred = make_rank_pct(df, pct, floor)
        add_variant(variants, f"v21_rankpct{pct}_floor{floor}", pred)

for pct in [0.18, 0.22, 0.26, 0.30, 0.34]:
    for min_k in [1, 2, 3, 5]:
        for max_k in [20, 40, 80, 150]:
            pred = make_adaptive_top(df, pct=pct, min_k=min_k, max_k=max_k, score_floor=1e-7)
            add_variant(variants, f"v21_adapt_pct{pct}_min{min_k}_max{max_k}", pred)

# ------------------------------------------------------------
# 3) Patch current best bases with v21 score
# ------------------------------------------------------------
bases = {
    "v16": pred_v16,
    "v19p1": pred_v19,
    "v20": pred_v20,
}

for bname, base in bases.items():
    for budget in [1000, 2000, 3000, 5000, 8000, 12000, 20000, 35000, 50000, 80000, 120000]:
        add_variant(variants, f"v21_{bname}_add{budget}", patch_add(base, v21_score, budget))
        add_variant(variants, f"v21_{bname}_rem{budget}", patch_rem(base, v21_score, budget))
        add_variant(variants, f"v21_{bname}_swap{budget}", patch_swap(base, v21_score, budget))

    for add_b in [5000, 10000, 20000, 40000, 80000]:
        for rem_b in [1000, 3000, 5000, 10000, 20000, 40000]:
            add_variant(
                variants,
                f"v21_{bname}_asym_add{add_b}_rem{rem_b}",
                patch_asym(base, v21_score, add_b, rem_b),
            )

# ------------------------------------------------------------
# 4) Consensus variants with v21 + old strong models
# ------------------------------------------------------------
# v21 global-top around current pos ratios
v21_bin_032 = make_global_top(v21_score, 0.32)
v21_bin_323 = make_global_top(v21_score, 0.323)
v21_bin_325 = make_global_top(v21_score, 0.325)

for name, v21_bin in [
    ("032", v21_bin_032),
    ("323", v21_bin_323),
    ("325", v21_bin_325),
]:
    votes = pred_v16 + pred_v19 + pred_v20 + v21_bin
    add_variant(variants, f"v21_vote_2of4_{name}", (votes >= 2).astype(np.int8))
    add_variant(variants, f"v21_vote_3of4_{name}", (votes >= 3).astype(np.int8))

    votes2 = pred_v16 + pred_v17 + v21_bin
    add_variant(variants, f"v21_vote_v16_v17_v21_2of3_{name}", (votes2 >= 2).astype(np.int8))

# ------------------------------------------------------------
# 5) V21 + V18 strict add candidates
# ------------------------------------------------------------
for base_name, base in bases.items():
    for add_budget in [2000, 5000, 10000, 20000, 40000]:
        cand = np.where((base == 0) & (v18_score >= 0.90))[0]
        pred = base.copy()
        if len(cand):
            order = cand[np.argsort(-v21_score[cand])]
            take = order[:min(add_budget, len(order))]
            pred[take] = 1
        add_variant(variants, f"v21v18_{base_name}_add{add_budget}", pred)

    for rem_budget in [2000, 5000, 10000, 20000, 40000]:
        cand = np.where((base == 1) & (v18_score <= 0.10))[0]
        pred = base.copy()
        if len(cand):
            order = cand[np.argsort(v21_score[cand])]
            take = order[:min(rem_budget, len(order))]
            pred[take] = 0
        add_variant(variants, f"v21v18_{base_name}_rem{rem_budget}", pred)

print("variants:", len(variants))

# ------------------------------------------------------------
# Full summary
# ------------------------------------------------------------
full_rows = []

for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v13": int((pred != pred_v13).sum()),
        "diff_vs_v13_ratio": float((pred != pred_v13).mean()),
        "diff_vs_v16": int((pred != pred_v16).sum()),
        "diff_vs_v16_ratio": float((pred != pred_v16).mean()),
        "diff_vs_v19p1": int((pred != pred_v19).sum()),
        "diff_vs_v19p1_ratio": float((pred != pred_v19).mean()),
        "diff_vs_v20": int((pred != pred_v20).sum()),
        "diff_vs_v20_ratio": float((pred != pred_v20).mean()),
        "diff_vs_v17p4": int((pred != pred_v17).sum()),
        "diff_vs_v17p4_ratio": float((pred != pred_v17).mean()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

# ------------------------------------------------------------
# Validation on manual labels
# ------------------------------------------------------------
eval_rows = []

for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label:", label_set, path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)

    if "assistant_label" not in lab.columns:
        print("missing assistant_label:", path)
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

    if label_set == "v20_active" and "review_bucket" in lab.columns:
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

v20_weights = {
    "v20_active_all": 0.10,
    "v20_active_clean": 0.20,
    "v20_active_high_medium_clean": 0.15,
    "v20_active_bucket_v19p1_added_over_v16": 0.08,
    "v20_active_bucket_possible_remove_v16_1_v17p4_0_v18_low": 0.20,
    "v20_active_bucket_possible_restore_v13_1_v16_0_v18_high": 0.03,
}

agg_rows = []

for name, g in eval_df.groupby("variant"):
    legacy_score = 0.0
    legacy_wsum = 0.0
    legacy_vals = []

    v20_score = 0.0
    v20_wsum = 0.0
    v20_vals = []

    for _, r in g.iterrows():
        key = r["eval_key"]

        lw = legacy_weights.get(key, 0.0)
        if lw > 0:
            legacy_score += lw * r["macro_f1"]
            legacy_wsum += lw
            legacy_vals.append(r["macro_f1"])

        tw = v20_weights.get(key, 0.0)
        if tw > 0:
            v20_score += tw * r["macro_f1"]
            v20_wsum += tw
            v20_vals.append(r["macro_f1"])

    if legacy_wsum == 0:
        continue

    legacy_weighted = legacy_score / legacy_wsum
    v20_weighted = v20_score / v20_wsum if v20_wsum > 0 else np.nan

    # legacy daha önemli; v20 active biased olduğu için düşük ağırlık.
    if v20_wsum > 0:
        combined_score = 0.80 * legacy_weighted + 0.20 * v20_weighted
    else:
        combined_score = legacy_weighted

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
        "v20_weighted": v20_weighted,
        "legacy_min_macro": float(np.min(legacy_vals)) if legacy_vals else np.nan,
        "v20_min_macro": float(np.min(v20_vals)) if v20_vals else np.nan,
        "main_min_macro": float(main["macro_f1"].min()) if len(main) else np.nan,
        "main_mean_macro": float(main["macro_f1"].mean()) if len(main) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "mean_pred_pos_ratio": float(g["pred_pos_ratio"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows)
agg = agg.merge(full, on="variant", how="left")
agg = agg.sort_values(["legacy_weighted", "combined_score", "main_min_macro"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP LEGACY")
print(agg.sort_values(["legacy_weighted", "combined_score"], ascending=False).head(80).to_string(index=False))

print("\nAGG TOP COMBINED")
print(agg.sort_values(["combined_score", "legacy_weighted"], ascending=False).head(80).to_string(index=False))

print("\nFULL TOP diff_vs_v20")
print(full.sort_values("diff_vs_v20").head(80).to_string(index=False))

print("\nFULL TOP v21 global")
print(full[full["variant"].str.startswith("v21_global")].sort_values("pos_ratio").to_string(index=False))

# Save top candidates.
top_names = []

for col in ["legacy_weighted", "combined_score", "main_min_macro"]:
    names = agg.sort_values(col, ascending=False).head(10)["variant"].tolist()
    for nm in names:
        if nm not in top_names:
            top_names.append(nm)

for b in ["v20_current", "v19p1_current", "v16_current", "v21_global_top_ratio0.323"]:
    if b in variants and b not in top_names:
        top_names.append(b)

for name in top_names[:35]:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v21_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
