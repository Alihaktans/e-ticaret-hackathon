from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
SWAPS = ROOT / "reports/manual_review/v30_micro_qswap_selected_swaps.csv"

OUT_SUMMARY = ROOT / "reports/manual_review/v31_micro_qswap_sparse_combo_summary.csv"
OUT_EVAL = ROOT / "reports/manual_review/v31_micro_qswap_sparse_combo_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v31_micro_qswap_sparse_combo_aggregate.csv"

V24_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
]

STRICT_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget2000.csv",
]

SUPPORTED_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget2000.csv",
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
    )

def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
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

def apply_qswaps(base_pred, swaps, k, mode):
    pred = base_pred.copy()
    take = swaps.head(k).copy()

    idx_add = take["idx_add"].astype(int).to_numpy()
    idx_drop = take["idx_drop"].astype(int).to_numpy()

    if mode == "force":
        pred[idx_add] = 1
        pred[idx_drop] = 0
        applied = len(take)

    elif mode == "preserve":
        # Sadece mevcut base'te add=0 ve drop=1 ise uygula.
        # Böylece pozitif sayısı korunur.
        mask = (pred[idx_add] == 0) & (pred[idx_drop] == 1)
        idx_add = idx_add[mask]
        idx_drop = idx_drop[mask]
        pred[idx_add] = 1
        pred[idx_drop] = 0
        applied = int(mask.sum())

    else:
        raise ValueError(mode)

    return pred.astype(np.int8), applied

print("loading sample")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])

v24_path = first_existing(V24_PATHS)
strict_path = first_existing(STRICT_PATHS)
supported_path = first_existing(SUPPORTED_PATHS)

if v24_path is None:
    raise FileNotFoundError("v24 anchor yok")
if strict_path is None:
    raise FileNotFoundError("v26 strict2000 yok")
if supported_path is None:
    raise FileNotFoundError("v26 supported2000 yok")

print("v24:", v24_path)
print("strict:", strict_path)
print("supported:", supported_path)

pred_v24 = load_pred(v24_path, sample)
pred_strict = load_pred(strict_path, sample)
pred_supported = load_pred(supported_path, sample)

strict_add_mask = (pred_strict == 1) & (pred_v24 == 0)
supported_add_mask = (pred_supported == 1) & (pred_v24 == 0)

print("strict additions over v24:", int(strict_add_mask.sum()))
print("supported additions over v24:", int(supported_add_mask.sum()))

swaps = pd.read_csv(SWAPS)
swaps["id_add"] = swaps["id_add"].astype(str)
swaps["id_drop"] = swaps["id_drop"].astype(str)

if "idx_add" not in swaps.columns:
    swaps["idx_add"] = swaps["id_add"].map(id_to_idx)
if "idx_drop" not in swaps.columns:
    swaps["idx_drop"] = swaps["id_drop"].map(id_to_idx)

if swaps["idx_add"].isna().any() or swaps["idx_drop"].isna().any():
    raise RuntimeError("swap idx map hatası")

swaps["idx_add"] = swaps["idx_add"].astype(int)
swaps["idx_drop"] = swaps["idx_drop"].astype(int)
swaps = swaps.sort_values("v30_swap_prob", ascending=False).reset_index(drop=True)

variants = {}
meta = []

def add_variant(name, pred, source, applied_qswaps):
    pred = pred.astype(np.int8)
    variants[name] = pred

    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v31_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)

    meta.append({
        "variant": name,
        "file": str(out),
        "source": source,
        "applied_qswaps": int(applied_qswaps),
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
        "diff_vs_strict2000": int((pred != pred_strict).sum()),
        "diff_vs_supported2000": int((pred != pred_supported).sum()),
    })

    print("saved", out, "ones", int(pred.sum()), "diff_vs_v24", int((pred != pred_v24).sum()))

# Baselines
add_variant("v24_anchor", pred_v24, "baseline", 0)
add_variant("v26_strict2000", pred_strict, "baseline", 0)
add_variant("v26_supported2000", pred_supported, "baseline", 0)

for k in [50, 100, 150, 200, 250, 300, 400, 500]:
    # 1) v24 üzerine saf micro qswap
    p, applied = apply_qswaps(pred_v24, swaps, k, "force")
    add_variant(f"v24_qtop{k}", p, "v24_force_qswap", applied)

    # 2) strict2000 üstüne sadece preserving qswap
    p, applied = apply_qswaps(pred_strict, swaps, k, "preserve")
    add_variant(f"strict2000_preserve_qtop{k}", p, "strict_preserve_qswap", applied)

    # 3) supported2000 üstüne sadece preserving qswap
    p, applied = apply_qswaps(pred_supported, swaps, k, "preserve")
    add_variant(f"supported2000_preserve_qtop{k}", p, "supported_preserve_qswap", applied)

    # 4) v24 qswap + strict sparse additions, qswap drop'ları geri almadan
    q, applied = apply_qswaps(pred_v24, swaps, k, "force")
    q[strict_add_mask] = 1
    add_variant(f"qtop{k}_plus_strict_adds", q, "v24_qswap_plus_strict_adds", applied)

    # 5) v24 qswap + supported sparse additions
    q, applied = apply_qswaps(pred_v24, swaps, k, "force")
    q[supported_add_mask] = 1
    add_variant(f"qtop{k}_plus_supported_adds", q, "v24_qswap_plus_supported_adds", applied)

summary = pd.DataFrame(meta)
summary.to_csv(OUT_SUMMARY, index=False)

# --------------------------
# Evaluation, qswap labels yok; leakage olmasın.
# --------------------------
eval_rows = []

for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label file:", label_set, path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)

    if "assistant_label" not in lab.columns:
        continue

    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)

    subsets = {"all": lab}

    if "needs_recheck" in lab.columns:
        nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
        subsets["clean"] = lab[nr == 0].copy()

    if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
        subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()
        subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()

    if label_set in ["v20_active", "v21_active", "v26_sparse"] and "review_bucket" in lab.columns:
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
    "v20_active_clean": 0.04,
    "v20_active_high_medium_clean": 0.03,
    "v21_active_clean": 0.12,
    "v21_active_high_medium_clean": 0.10,
    "v21_active_bucket_v21_add5k_added_over_v20": 0.08,
    "v21_active_bucket_v21_vote_added_over_v20": 0.12,
    "v21_active_bucket_v21_vote_removed_from_v20": 0.12,
}

sparse_weights = {
    "v26_sparse_clean": 0.10,
    "v26_sparse_high_medium_clean": 0.08,
    "v26_sparse_bucket_supported_0_500_added": 0.04,
    "v26_sparse_bucket_supported_500_1000_extra": 0.04,
    "v26_sparse_bucket_supported_1000_2000_extra": 0.06,
    "v26_sparse_bucket_supported_2000_3000_extra": 0.04,
    "v26_sparse_bucket_supported_3000_5000_extra_risky": 0.06,
    "v26_sparse_bucket_strictpct3_2000_extra_not_supported": 0.06,
    "v26_sparse_bucket_supported_2000_extra_not_strictpct3": 0.06,
    "v26_sparse_bucket_raw_sparse_5000_extra_not_supported": 0.04,
}

agg_rows = []
anchor_pos = float(pred_v24.mean())

for name, g in eval_df.groupby("variant"):
    legacy_score = legacy_wsum = 0.0
    active_score = active_wsum = 0.0
    sparse_score = sparse_wsum = 0.0
    legacy_vals = []
    active_vals = []
    sparse_vals = []

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

        sw = sparse_weights.get(key, 0.0)
        if sw:
            sparse_score += sw * r["macro_f1"]
            sparse_wsum += sw
            sparse_vals.append(r["macro_f1"])

    if legacy_wsum == 0:
        continue

    legacy_weighted = legacy_score / legacy_wsum
    active_weighted = active_score / active_wsum if active_wsum else np.nan
    sparse_weighted = sparse_score / sparse_wsum if sparse_wsum else np.nan

    if active_wsum and sparse_wsum:
        combined_score = 0.64 * legacy_weighted + 0.24 * active_weighted + 0.12 * sparse_weighted
    elif active_wsum:
        combined_score = 0.72 * legacy_weighted + 0.28 * active_weighted
    else:
        combined_score = legacy_weighted

    s = summary[summary["variant"] == name].iloc[0]
    pos_ratio = float(s["pos_ratio"])
    diff_v24 = int(s["diff_vs_v24"])

    pos_penalty = abs(pos_ratio - anchor_pos) * 0.45
    diff_penalty = max(0, diff_v24 - 20000) / 900000.0
    public_safe_score = combined_score - pos_penalty - diff_penalty

    main_eval = g[g["eval_key"].isin([
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
        "sparse_weighted": sparse_weighted,
        "legacy_min_macro": float(np.min(legacy_vals)) if legacy_vals else np.nan,
        "active_min_macro": float(np.min(active_vals)) if active_vals else np.nan,
        "sparse_min_macro": float(np.min(sparse_vals)) if sparse_vals else np.nan,
        "main_min_macro": float(main_eval["macro_f1"].min()) if len(main_eval) else np.nan,
        "main_mean_macro": float(main_eval["macro_f1"].mean()) if len(main_eval) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows)
agg = agg.merge(summary, on="variant", how="left")
agg = agg.sort_values(["public_safe_score", "combined_score", "legacy_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nTOP PUBLIC SAFE")
print(agg.head(80).to_string(index=False))

print("\nSUMMARY")
print(summary.sort_values("diff_vs_v24").to_string(index=False))

print("saved:", OUT_SUMMARY)
print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
