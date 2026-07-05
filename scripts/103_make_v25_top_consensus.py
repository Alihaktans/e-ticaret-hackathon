from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
AGG24 = ROOT / "reports/manual_review/v24_grid_aggregate.csv"

V21 = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V5 = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18 = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"

OUT_EVAL = ROOT / "reports/manual_review/v25_consensus_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v25_consensus_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v25_consensus_full_summary.csv"

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
    "v23_swap5": [
        ROOT / "submissions/FINAL_MAIN_v23_vote_swap5000.csv",
        ROOT / "submissions/FINAL_MAIN_v23_swap5000_try1.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v23_v23_vote_swap_restore_trim_budget5000.csv",
    ],
    "v24_best": [
        ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
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
        str(x).replace(".", "p")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace(":", "")
    )

def load_sub_path(p, sample):
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {p}")
    return d["prediction"].astype(np.int8).to_numpy()

def load_sub(name, sample):
    p = first_existing(PATHS[name])
    if p is None:
        raise FileNotFoundError(f"missing {name}")
    print("loading", name, p)
    return load_sub_path(p, sample)

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
pred_base = load_sub("vote_base", sample)
pred_swap5 = load_sub("v23_swap5", sample)
pred_v24best = load_sub("v24_best", sample)

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

v5f = np.clip(v5_score, 0, 1)
v18f = np.clip(v18_score, 0, 1)
v21f = np.clip(v21_score, 0, 1)

# ------------------------------------------------------------
# Load top v24 candidate predictions
# ------------------------------------------------------------
agg24 = pd.read_csv(AGG24)

# Çok uç dosyaları azalt: base'ten çok uzaklaşmasın, pos ratio çok oynamasın.
base_pos = float(pred_base.mean())

agg24["diff_vs_vote_base"] = pd.to_numeric(agg24["diff_vs_vote_base"], errors="coerce")
agg24["pos_ratio"] = pd.to_numeric(agg24["pos_ratio"], errors="coerce")
agg24["public_safe_score"] = pd.to_numeric(agg24["public_safe_score"], errors="coerce")
agg24["combined_score"] = pd.to_numeric(agg24["combined_score"], errors="coerce")

eligible = agg24[
    (agg24["diff_vs_vote_base"] >= 4000)
    & (agg24["diff_vs_vote_base"] <= 24000)
    & ((agg24["pos_ratio"] - base_pos).abs() <= 0.004)
].copy()

eligible = eligible.sort_values(["public_safe_score", "combined_score"], ascending=False)

top_variants = []
top_preds = []

for _, r in eligible.iterrows():
    v = r["variant"]
    p = ROOT / "submissions" / f"FINAL_CANDIDATE_v24_{clean_name(v)}.csv"
    if not p.exists():
        continue
    try:
        pred = load_sub_path(p, sample)
    except Exception as e:
        print("skip", v, e)
        continue

    top_variants.append(v)
    top_preds.append(pred)

    if len(top_preds) >= 40:
        break

print("loaded top candidate preds:", len(top_preds))
print("top variants:")
for i, v in enumerate(top_variants[:20], 1):
    print(i, v)

if len(top_preds) < 5:
    raise RuntimeError("not enough top v24 predictions found")

P = np.vstack(top_preds).astype(np.int8)

# Changes relative to public 0.80 base.
restore_votes = ((P == 1) & (pred_base[None, :] == 0)).sum(axis=0).astype(np.int16)
trim_votes = ((P == 0) & (pred_base[None, :] == 1)).sum(axis=0).astype(np.int16)

restore_possible = np.where((pred_base == 0) & (restore_votes > 0))[0]
trim_possible = np.where((pred_base == 1) & (trim_votes > 0))[0]

restore_support_score = (
    restore_votes / len(top_preds)
    + 0.35 * v5f
    + 0.35 * v18f
    + 0.15 * pred_add5.astype("float32")
    + 0.05 * pred_v19.astype("float32")
)

trim_support_score = (
    trim_votes / len(top_preds)
    + 0.35 * (1 - v5f)
    + 0.25 * (1 - v18f)
    + 0.25 * (1 - v21f)
    + 0.05 * (1 - pred_add5.astype("float32"))
)

restore_order = restore_possible[np.argsort(-restore_support_score[restore_possible])]
trim_order = trim_possible[np.argsort(-trim_support_score[trim_possible])]

print("restore_possible:", len(restore_possible))
print("trim_possible:", len(trim_possible))

variants = {}

add_variant(variants, "v22_vote_public080_base", pred_base)
add_variant(variants, "v23_swap5000_current", pred_swap5)
add_variant(variants, "v24_best_rE_tE_b6500", pred_v24best)
add_variant(variants, "v20_current", pred_v20)
add_variant(variants, "v21_add5k_existing", pred_add5)

# ------------------------------------------------------------
# 1) Majority consensus direct from top-N
# ------------------------------------------------------------
for topn in [5, 8, 10, 15, 20, 30, 40]:
    if topn > len(top_preds):
        continue

    Q = P[:topn]
    rv = ((Q == 1) & (pred_base[None, :] == 0)).sum(axis=0)
    tv = ((Q == 0) & (pred_base[None, :] == 1)).sum(axis=0)

    for frac in [0.30, 0.40, 0.50, 0.60, 0.70]:
        rth = max(1, int(np.ceil(topn * frac)))
        tth = max(1, int(np.ceil(topn * frac)))

        pred = pred_base.copy()
        pred[(pred_base == 0) & (rv >= rth)] = 1
        pred[(pred_base == 1) & (tv >= tth)] = 0

        add_variant(variants, f"v25_consensus_top{topn}_frac{frac}", pred)

# ------------------------------------------------------------
# 2) Budgeted consensus: balanced swap
# ------------------------------------------------------------
for budget in [3000, 4000, 5000, 6500, 8000, 10000, 12000, 15000]:
    pred = pred_base.copy()
    pred[restore_order[:min(budget, len(restore_order))]] = 1
    pred[trim_order[:min(budget, len(trim_order))]] = 0
    add_variant(variants, f"v25_consensus_swap_budget{budget}", pred)

# ------------------------------------------------------------
# 3) Asymmetric consensus: pos ratio hafif oynasın
# ------------------------------------------------------------
for rb, tb in [
    (5000, 4000),
    (6500, 5000),
    (8000, 6500),
    (10000, 8000),
    (12000, 10000),
    (15000, 12000),
    (4000, 5000),
    (5000, 6500),
    (6500, 8000),
    (8000, 10000),
]:
    pred = pred_base.copy()
    pred[restore_order[:min(rb, len(restore_order))]] = 1
    pred[trim_order[:min(tb, len(trim_order))]] = 0
    add_variant(variants, f"v25_consensus_asym_r{rb}_t{tb}", pred)

# ------------------------------------------------------------
# 4) Top candidates majority over final prediction directly
# ------------------------------------------------------------
for topn in [5, 8, 10, 15, 20, 30, 40]:
    if topn > len(top_preds):
        continue

    Q = P[:topn]
    vote_sum = Q.sum(axis=0)

    for frac in [0.45, 0.50, 0.55, 0.60]:
        th = int(np.ceil(topn * frac))
        pred = (vote_sum >= th).astype(np.int8)

        # pos ratio çok uçsa skip etmeden kaydet; değerlendirme cezalandıracak.
        add_variant(variants, f"v25_direct_top{topn}_frac{frac}", pred)

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
        "diff_vs_vote_base": int((pred != pred_base).sum()),
        "diff_vs_swap5": int((pred != pred_swap5).sum()),
        "diff_vs_v24best": int((pred != pred_v24best).sum()),
        "diff_vs_v20": int((pred != pred_v20).sum()),
        "diff_vs_add5": int((pred != pred_add5).sum()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
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
    diff_base = int(f["diff_vs_vote_base"])

    # Public 0.80 anchor cezası.
    pos_penalty = abs(pos_ratio - base_pos) * 1.40
    diff_penalty = max(0, diff_base - 22000) / 900000.0

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
print(near.head(150).to_string(index=False))

# Save top candidates
top_names = []
for col in ["public_safe_score", "combined_score", "legacy_weighted", "active_weighted", "main_min_macro"]:
    for nm in agg.sort_values(col, ascending=False).head(12)["variant"].tolist():
        if nm not in top_names:
            top_names.append(nm)

for nm in ["v22_vote_public080_base", "v23_swap5000_current", "v24_best_rE_tE_b6500"]:
    if nm in variants and nm not in top_names:
        top_names.append(nm)

for name in top_names[:40]:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v25_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
