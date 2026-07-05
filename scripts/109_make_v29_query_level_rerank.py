from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"

V5_SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18_SCORE = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"
V21_SCORE = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V27_SCORE = ROOT / "data/processed/v27_sparse_aug_catboost_test_scores.parquet"
SPARSE_SCORE = ROOT / "data/processed/v26_sparse_text_scores.parquet"

KNOWN_BAD = ROOT / "reports/manual_review/v28_sparse_safety_flagged_additions.csv"

OUT_EVAL = ROOT / "reports/manual_review/v29_qrerank_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v29_qrerank_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v29_qrerank_full_summary.csv"
OUT_SWAPS = ROOT / "reports/manual_review/v29_qrerank_swap_summary.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
    "v26_sparse": ROOT / "reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv",
}

PATHS = {
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
    ],
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
    ],
    "add5": [
        ROOT / "submissions/FINAL_MAIN_v21_v20_add5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add5000.csv",
    ],
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
    "v26_strict2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget2000.csv",
    ],
    "v26_supported2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget2000.csv",
    ],
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
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace(":", "")
    )

def load_pred(name, sample, required=True):
    p = first_existing(PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(name)
        print("missing optional:", name)
        return None

    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)

    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {name}")

    return d["prediction"].astype(np.int8).to_numpy()

def load_score(path, sample, col_candidates, default=0.0):
    n = len(sample)
    arr = np.full(n, default, dtype=np.float32)

    if not path.exists():
        print("missing score:", path)
        return arr

    df = pd.read_parquet(path)
    df["id"] = df["id"].astype(str)

    col = None
    for c in col_candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        print("missing score column in", path, "columns:", df.columns.tolist())
        return arr

    if df["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        arr = df[col].astype("float32").fillna(default).to_numpy()
    else:
        tmp = sample[["id"]].merge(df[["id", col]], on="id", how="left", validate="one_to_one")
        arr = tmp[col].astype("float32").fillna(default).to_numpy()

    return arr

def rank_score_by_term(df, col):
    r = df.groupby("term_id")[col].rank(method="first", ascending=False).astype("float32")
    cnt = df["candidate_count"].astype("float32")
    denom = np.maximum(cnt - 1.0, 1.0)
    out = 1.0 - ((r - 1.0) / denom)
    return out.astype("float32").to_numpy()

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

def save_variant(variants, name, pred, ids):
    pred = pred.astype(np.int8)
    variants[name] = pred
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v29_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)
    print("saved", out, "ones", int(pred.sum()), "pos_ratio", float(pred.mean()))

print("loading sample/pairs...")
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
    raise RuntimeError("submission_pairs id order mismatch sample")

df = pairs[["id", "term_id", "item_id"]].copy()
df["candidate_count"] = df.groupby("term_id")["id"].transform("count").astype(np.int32)

print("loading predictions...")
pred_v13 = load_pred("v13", sample, required=False)
pred_v16 = load_pred("v16", sample)
pred_v19 = load_pred("v19p1", sample)
pred_v20 = load_pred("v20", sample)
pred_add5 = load_pred("add5", sample)
pred_base = load_pred("v22_base", sample)
pred_v24 = load_pred("v24_anchor", sample)
pred_strict = load_pred("v26_strict2000", sample, required=False)
pred_supported = load_pred("v26_supported2000", sample, required=False)

if pred_v13 is None:
    pred_v13 = pred_base.copy()
if pred_strict is None:
    pred_strict = pred_v24.copy()
if pred_supported is None:
    pred_supported = pred_v24.copy()

vote_stack = np.vstack([
    pred_v13,
    pred_v16,
    pred_v19,
    pred_v20,
    pred_add5,
    pred_base,
    pred_v24,
    pred_strict,
    pred_supported,
])

vote_sum = vote_stack.sum(axis=0).astype(np.float32)
vote_mean = vote_sum / vote_stack.shape[0]

print("loading scores...")
df["v5"] = load_score(V5_SCORE, sample, ["proba_avg", "v5_score", "score", "prediction"])
df["v18"] = load_score(V18_SCORE, sample, ["v18_minilm_sigmoid", "v18_score", "score", "sigmoid"])
df["v21"] = load_score(V21_SCORE, sample, ["v21_score", "score", "prediction"])
df["v27"] = load_score(V27_SCORE, sample, ["v27_score", "score", "prediction"])

if SPARSE_SCORE.exists():
    sp = pd.read_parquet(SPARSE_SCORE)
    sp["id"] = sp["id"].astype(str)
    needed = ["id", "sparse_blend", "sparse_pct_rank"]
    for c in needed:
        if c not in sp.columns:
            raise RuntimeError(f"missing sparse col {c}")
    if sp["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        df["sparse_blend"] = sp["sparse_blend"].astype("float32").fillna(0).to_numpy()
        df["sparse_pct_rank"] = sp["sparse_pct_rank"].astype("float32").fillna(1).to_numpy()
    else:
        tmp = sample[["id"]].merge(sp[needed], on="id", how="left", validate="one_to_one")
        df["sparse_blend"] = tmp["sparse_blend"].astype("float32").fillna(0).to_numpy()
        df["sparse_pct_rank"] = tmp["sparse_pct_rank"].astype("float32").fillna(1).to_numpy()
else:
    df["sparse_blend"] = 0.0
    df["sparse_pct_rank"] = 1.0

print("making per-query rank scores...")
for c in ["v5", "v18", "v21", "v27", "sparse_blend"]:
    df[f"{c}_rs"] = rank_score_by_term(df, c)

# sparse_pct_rank düşükse iyi; onu da ters skor olarak kullan
df["sparse_pct_rs"] = (1.0 - df["sparse_pct_rank"].astype("float32")).clip(0, 1)

v5_rs = df["v5_rs"].to_numpy(dtype=np.float32)
v18_rs = df["v18_rs"].to_numpy(dtype=np.float32)
v21_rs = df["v21_rs"].to_numpy(dtype=np.float32)
v27_rs = df["v27_rs"].to_numpy(dtype=np.float32)
sp_rs = (0.65 * df["sparse_blend_rs"].to_numpy(dtype=np.float32) + 0.35 * df["sparse_pct_rs"].to_numpy(dtype=np.float32)).astype(np.float32)

# Known bad rows from v28 safety.
known_bad = np.zeros(n, dtype=bool)
if KNOWN_BAD.exists():
    bad = pd.read_csv(KNOWN_BAD, usecols=["id"])
    bad["id"] = bad["id"].astype(str)
    mapped = bad["id"].map(id_to_idx)
    mapped = mapped.dropna().astype(int).to_numpy()
    known_bad[mapped] = True
    print("known_bad rows:", int(known_bad.sum()))
else:
    print("known bad file not found; continuing without it")

# Scoring flavors.
flavors = {}

flavors["legacy"] = (
    0.30 * v5_rs +
    0.22 * v18_rs +
    0.20 * v21_rs +
    0.15 * sp_rs +
    0.13 * vote_mean
).astype(np.float32)

flavors["v27blend"] = (
    0.24 * v5_rs +
    0.18 * v18_rs +
    0.18 * v21_rs +
    0.18 * v27_rs +
    0.12 * sp_rs +
    0.10 * vote_mean
).astype(np.float32)

flavors["voteheavy"] = (
    0.24 * v5_rs +
    0.16 * v18_rs +
    0.16 * v21_rs +
    0.10 * v27_rs +
    0.10 * sp_rs +
    0.24 * vote_mean
).astype(np.float32)

flavors["sparselegacy"] = (
    0.26 * v5_rs +
    0.14 * v18_rs +
    0.16 * v21_rs +
    0.24 * sp_rs +
    0.20 * vote_mean
).astype(np.float32)

flavors["nov27_conservative"] = (
    0.34 * v5_rs +
    0.22 * v18_rs +
    0.20 * v21_rs +
    0.10 * sp_rs +
    0.14 * vote_mean
).astype(np.float32)

support_masks = {
    "loose": (
        (np.maximum.reduce([v5_rs, v18_rs, v21_rs, v27_rs, sp_rs]) >= 0.86)
        | (vote_mean >= 0.38)
        | (pred_strict == 1)
        | (pred_supported == 1)
    ),
    "strict": (
        (
            (v5_rs >= 0.72)
            | (v18_rs >= 0.72)
            | (v21_rs >= 0.82)
            | (sp_rs >= 0.92)
            | (pred_strict == 1)
        )
        & (vote_mean >= 0.18)
    ),
    "vote": (
        (vote_mean >= 0.42)
        | ((vote_mean >= 0.30) & ((v5_rs >= 0.70) | (v18_rs >= 0.70) | (sp_rs >= 0.90)))
        | (pred_strict == 1)
    ),
    "sparse": (
        (sp_rs >= 0.94)
        | (pred_strict == 1)
        | ((sp_rs >= 0.88) & ((v5_rs >= 0.65) | (v18_rs >= 0.65) | (vote_mean >= 0.30)))
    ),
}

def build_qswap_pairs(anchor_name, anchor_pred, score_name, score_arr, support_name, support_mask, safe=True):
    base0 = anchor_pred == 0
    base1 = anchor_pred == 1

    add_mask = base0 & support_mask
    if safe:
        add_mask = add_mask & (~known_bad)

    # Drop only weak current positives; this makes it query-level but conservative.
    drop_mask = base1 & (
        (vote_mean <= 0.66)
        | (score_arr <= 0.52)
        | ((v5_rs <= 0.30) & (v18_rs <= 0.30))
        | ((v21_rs <= 0.30) & (sp_rs <= 0.35))
    )

    tmp = pd.DataFrame({
        "term_id": df["term_id"].to_numpy(),
        "idx": np.arange(n, dtype=np.int32),
        "score": score_arr.astype(np.float32),
        "add_mask": add_mask,
        "drop_mask": drop_mask,
    })

    adds = tmp[tmp["add_mask"]].copy()
    drops = tmp[tmp["drop_mask"]].copy()

    if len(adds) == 0 or len(drops) == 0:
        return pd.DataFrame()

    adds = adds.sort_values(["term_id", "score"], ascending=[True, False])
    adds["pair_rank"] = adds.groupby("term_id").cumcount()

    drops = drops.sort_values(["term_id", "score"], ascending=[True, True])
    drops["pair_rank"] = drops.groupby("term_id").cumcount()

    pairs2 = adds.merge(
        drops,
        on=["term_id", "pair_rank"],
        suffixes=("_add", "_drop"),
        how="inner",
    )

    pairs2["gain"] = pairs2["score_add"] - pairs2["score_drop"]
    pairs2 = pairs2[pairs2["gain"] > 0].copy()

    if len(pairs2):
        pairs2["anchor"] = anchor_name
        pairs2["score_name"] = score_name
        pairs2["support_name"] = support_name
        pairs2["safe"] = int(safe)
        pairs2 = pairs2.sort_values("gain", ascending=False).reset_index(drop=True)

    return pairs2

variants = {}

save_variant(variants, "v22_base_public080", pred_base, ids)
save_variant(variants, "v24_anchor", pred_v24, ids)
save_variant(variants, "v26_strict2000", pred_strict, ids)
save_variant(variants, "v26_supported2000", pred_supported, ids)

swap_report_rows = []

anchors = {
    "v24": pred_v24,
    "v22": pred_base,
}

for anchor_name, anchor_pred in anchors.items():
    for score_name, score_arr in flavors.items():
        for support_name, support_mask in support_masks.items():
            for safe in [True, False]:
                pairs2 = build_qswap_pairs(anchor_name, anchor_pred, score_name, score_arr, support_name, support_mask, safe=safe)

                if len(pairs2) == 0:
                    continue

                safe_tag = "safe" if safe else "raw"

                for min_gain in [0.03, 0.05, 0.07, 0.10]:
                    p2 = pairs2[pairs2["gain"] >= min_gain].copy()

                    swap_report_rows.append({
                        "anchor": anchor_name,
                        "score_name": score_name,
                        "support_name": support_name,
                        "safe": safe_tag,
                        "min_gain": min_gain,
                        "available_swaps": int(len(p2)),
                        "gain_mean": float(p2["gain"].mean()) if len(p2) else np.nan,
                        "gain_min": float(p2["gain"].min()) if len(p2) else np.nan,
                        "gain_max": float(p2["gain"].max()) if len(p2) else np.nan,
                    })

                    for budget in [250, 500, 1000, 1500, 2000, 3000, 5000, 8000]:
                        if len(p2) < max(50, min(budget, 250)):
                            continue

                        take = p2.head(budget)
                        pred = anchor_pred.copy()
                        pred[take["idx_add"].to_numpy(dtype=np.int32)] = 1
                        pred[take["idx_drop"].to_numpy(dtype=np.int32)] = 0

                        name = f"qswap_{anchor_name}_{score_name}_{support_name}_{safe_tag}_g{str(min_gain).replace('.', 'p')}_b{budget}"
                        save_variant(variants, name, pred, ids)

swap_report = pd.DataFrame(swap_report_rows)
swap_report.to_csv(OUT_SWAPS, index=False)

# ------------------------------------------------------------
# Full summary
# ------------------------------------------------------------
full_rows = []

for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v22": int((pred != pred_base).sum()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
        "diff_vs_strict2000": int((pred != pred_strict).sum()),
        "diff_vs_supported2000": int((pred != pred_supported).sum()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
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
        nr = lab["needs_recheck"].fillna(0).astype(str).str.replace(".0", "", regex=False)
        nr = pd.to_numeric(nr, errors="coerce").fillna(0).astype(int)
        subsets["clean"] = lab[nr == 0].copy()

    if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        nr = lab["needs_recheck"].fillna(0).astype(str).str.replace(".0", "", regex=False)
        nr = pd.to_numeric(nr, errors="coerce").fillna(0).astype(int)
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

    f = full[full["variant"] == name].iloc[0]
    pos_ratio = float(f["pos_ratio"])
    diff_v24 = int(f["diff_vs_v24"])
    diff_v22 = int(f["diff_vs_v22"])

    # Query-swap pos_ratio sabit kalmalı; çok büyük diff cezalı.
    pos_penalty = abs(pos_ratio - anchor_pos) * 1.20
    diff_penalty = max(0, min(diff_v24, diff_v22) - 25000) / 900000.0
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
agg = agg.merge(full, on="variant", how="left")
agg = agg.sort_values(["public_safe_score", "combined_score", "legacy_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nTOP PUBLIC SAFE")
print(agg.head(100).to_string(index=False))

print("\nTOP COMBINED")
print(agg.sort_values("combined_score", ascending=False).head(100).to_string(index=False))

print("\nFULL SUMMARY NEAR V24")
print(full.sort_values("diff_vs_v24").head(120).to_string(index=False))

print("\nSWAP REPORT TOP")
if len(swap_report):
    print(swap_report.sort_values("available_swaps", ascending=False).head(80).to_string(index=False))

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
print("saved:", OUT_SWAPS)
