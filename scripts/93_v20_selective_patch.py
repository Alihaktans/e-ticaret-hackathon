from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
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
    "v17p4": [
        ROOT / "submissions/FINAL_MAIN_v17p4_independent_b04_ba008_r1.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v17p4_v17p4_qt_b0p4_ba0p08_sa0p0_ga0p0_nocap_r1.csv",
    ],
    "v19p1": [
        ROOT / "submissions/FINAL_MAIN_v19p1_honest_addonly_ba012_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba012_r1_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba008_r1_v18th0p9.csv",
    ],
    "v18add": [
        ROOT / "submissions/FINAL_MAIN_v18_v16_addonly_0p95.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v18_v18_v16_addonly_0p95.csv",
    ],
}

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
}

OUT_EVAL = ROOT / "reports/manual_review/v20_selective_patch_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v20_selective_patch_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v20_selective_patch_full_summary.csv"

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

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

def load_sub(name, required=True):
    p = first_existing(SUB_PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(f"Missing submission for {name}")
        print("missing optional submission:", name)
        return None
    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    assert d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), name
    return d["prediction"].astype(np.int8).to_numpy()

pred_v13 = load_sub("v13")
pred_v16 = load_sub("v16")
pred_v17 = load_sub("v17p4")
pred_v19 = load_sub("v19p1")
pred_v18add = load_sub("v18add", required=False)

print("loading v5...")
v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
v5["id"] = v5["id"].astype(str)
assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
v5_score = v5["proba_avg"].astype("float32").to_numpy()

print("loading v18...")
v18 = pd.read_parquet(V18)
v18["id"] = v18["id"].astype(str)

v18_score = np.full(n, -1.0, dtype=np.float32)
idx = v18["id"].map(id_to_idx)
ok = idx.notna()
v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()
has_v18 = v18_score >= 0

print("v18 scored rows:", int(has_v18.sum()))

variants = {}

def add_variant(name, pred):
    variants[name] = pred.astype(np.int8)

add_variant("v13_public_0p77", pred_v13.copy())
add_variant("v16_current", pred_v16.copy())
add_variant("v17p4_reference_risky", pred_v17.copy())
add_variant("v19p1_current", pred_v19.copy())

if pred_v18add is not None:
    add_variant("v18_addonly_0p95", pred_v18add.copy())

# ------------------------------------------------------------
# V20 main idea:
# base = v19p1
# selective removal where:
#   v19p1=1, original v16=1, v17p4=0, v18 very low, v5 not very high
# ------------------------------------------------------------

base = pred_v19.copy()

# Threshold removal variants
for rem_th in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]:
    for v5_max in [0.50, 0.60, 0.70, 0.75, 0.80]:
        pred = base.copy()
        rem = (
            (pred == 1)
            & (pred_v16 == 1)
            & (pred_v17 == 0)
            & has_v18
            & (v18_score <= rem_th)
            & (v5_score < v5_max)
        )
        pred[rem] = 0
        add_variant(f"v20_rem_t{rem_th}_v5lt{v5_max}", pred)

# Even safer: only remove if v13 also says 0
for rem_th in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]:
    for v5_max in [0.60, 0.70, 0.75, 0.80]:
        pred = base.copy()
        rem = (
            (pred == 1)
            & (pred_v16 == 1)
            & (pred_v17 == 0)
            & (pred_v13 == 0)
            & has_v18
            & (v18_score <= rem_th)
            & (v5_score < v5_max)
        )
        pred[rem] = 0
        add_variant(f"v20_rem_v13zero_t{rem_th}_v5lt{v5_max}", pred)

# Budgeted removals: remove the lowest-risk rows first.
# score lower = more removable
remove_universe = (
    (base == 1)
    & (pred_v16 == 1)
    & (pred_v17 == 0)
    & has_v18
    & (v18_score <= 0.20)
    & (v5_score < 0.80)
)

remove_universe_v13zero = remove_universe & (pred_v13 == 0)

for universe_name, universe in [
    ("all", remove_universe),
    ("v13zero", remove_universe_v13zero),
]:
    idxs = np.where(universe)[0]
    if len(idxs):
        # lower score = stronger negative evidence
        risk = (v18_score[idxs] * 0.70) + (v5_score[idxs] * 0.30)
        order = idxs[np.argsort(risk)]

        for budget in [500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000]:
            take = order[:min(budget, len(order))]
            pred = base.copy()
            pred[take] = 0
            add_variant(f"v20_rembudget_{universe_name}_{budget}", pred)

# Conservative addition expansion:
# only if v19p1 still 0, v17p4=1, v18 very high, and v5 not terrible.
# Review'da missed_add çok güçlü çıkmadığı için bunları ayrı ve küçük tutuyoruz.
for add_th in [0.95, 0.97, 0.99]:
    for v5_min in [0.50, 0.60, 0.70]:
        pred = base.copy()
        add = (
            (pred == 0)
            & (pred_v17 == 1)
            & has_v18
            & (v18_score >= add_th)
            & (v5_score >= v5_min)
        )
        pred[add] = 1
        add_variant(f"v20_add_t{add_th}_v5ge{v5_min}", pred)

# Combined conservative: limited removal + tiny addition
for rem_budget in [1000, 2000, 3000, 4000]:
    idxs = np.where(remove_universe_v13zero)[0]
    if len(idxs):
        risk = (v18_score[idxs] * 0.70) + (v5_score[idxs] * 0.30)
        order = idxs[np.argsort(risk)]
        take = order[:min(rem_budget, len(order))]
    else:
        take = np.array([], dtype=int)

    for add_th in [0.97, 0.99]:
        pred = base.copy()
        pred[take] = 0

        add = (
            (pred == 0)
            & (pred_v17 == 1)
            & has_v18
            & (v18_score >= add_th)
            & (v5_score >= 0.60)
        )
        pred[add] = 1

        add_variant(f"v20_combo_remv13zero{rem_budget}_add{add_th}_v5ge0.6", pred)

print("variants:", len(variants))

# Full summary
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
        "diff_vs_v17p4": int((pred != pred_v17).sum()),
        "diff_vs_v17p4_ratio": float((pred != pred_v17).mean()),
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

    # v20 için bucket bazlı da bak
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
    "v20_active_all": 0.20,
    "v20_active_clean": 0.35,
    "v20_active_high_medium_clean": 0.25,
    "v20_active_bucket_v19p1_added_over_v16": 0.10,
    "v20_active_bucket_possible_remove_v16_1_v17p4_0_v18_low": 0.40,
    "v20_active_bucket_possible_restore_v13_1_v16_0_v18_high": 0.05,
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

    if v20_wsum > 0:
        combined_score = 0.65 * legacy_weighted + 0.35 * v20_weighted
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
agg = agg.sort_values(["combined_score", "legacy_weighted", "v20_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP")
print(agg.head(80).to_string(index=False))

print("\nFULL TOP diff_vs_v19p1")
print(full.sort_values("diff_vs_v19p1").head(80).to_string(index=False))

# Save top candidates
top_names = agg.head(12)["variant"].tolist()
for b in ["v19p1_current", "v16_current", "v18_addonly_0p95"]:
    if b in variants and b not in top_names:
        top_names.append(b)

for name in top_names:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v20_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
