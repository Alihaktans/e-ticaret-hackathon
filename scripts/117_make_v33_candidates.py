from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"

OUT_SUMMARY = ROOT / "reports/manual_review/v33_candidate_summary.csv"
OUT_EVAL = ROOT / "reports/manual_review/v33_candidate_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v33_candidate_aggregate.csv"
OUT_SWAPS = ROOT / "reports/manual_review/v33_swap_pool_summary.csv"

V5_SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18_SCORE = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"
V21_SCORE = ROOT / "data/processed/v21_catboost_test_scores.parquet"
V27_SCORE = ROOT / "data/processed/v27_sparse_aug_catboost_test_scores.parquet"
SPARSE_SCORE = ROOT / "data/processed/v26_sparse_text_scores.parquet"

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
    "v22": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "v24": [
        ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
    ],
    "strict2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget2000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v31_v26_strict2000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v32_v26_strict2000.csv",
    ],
    "supported2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget2000.csv",
    ],
    "v32": [
        ROOT / "submissions/FINAL_MAIN_v32_strict_qswap_prob_top400.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v32_strict_qswap_prob_top400.csv",
    ],
}

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
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
    )


def load_pred(name, sample, required=False):
    p = first_existing(PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(name)
        print("missing optional pred:", name)
        return None

    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)

    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {name} {p}")

    arr = d["prediction"].astype(np.int8).to_numpy()
    print("loaded", name, p, "ones", int(arr.sum()))
    return arr


def load_score(path, sample, cols, default=0.0):
    arr = np.full(len(sample), default, dtype=np.float32)

    if not path.exists():
        print("missing score:", path)
        return arr

    d = pd.read_parquet(path)
    d["id"] = d["id"].astype(str)

    col = None
    for c in cols:
        if c in d.columns:
            col = c
            break

    if col is None:
        print("missing score col:", path, d.columns.tolist())
        return arr

    if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        return d[col].astype("float32").fillna(default).to_numpy()

    tmp = sample[["id"]].merge(d[["id", col]], on="id", how="left", validate="one_to_one")
    return tmp[col].astype("float32").fillna(default).to_numpy()


def rank_score_by_term(df, col):
    r = df.groupby("term_id")[col].rank(method="first", ascending=False).astype("float32")
    cnt = df["candidate_count"].astype("float32")
    return (1.0 - ((r - 1.0) / np.maximum(cnt - 1.0, 1.0))).astype("float32").to_numpy()


def sigmoid(x):
    x = np.clip(x, -12, 12)
    return 1.0 / (1.0 + np.exp(-x))


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
    raise RuntimeError("submission_pairs id order mismatch")

df = pairs[["id", "term_id", "item_id"]].copy()
df["candidate_count"] = df.groupby("term_id")["id"].transform("count").astype(np.int32)

print("loading v33 scores...")
v33 = pd.read_parquet(V33)
v33["id"] = v33["id"].astype(str)

if not v33["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
    v33 = sample[["id"]].merge(
        v33[["id", "v33_ft_score", "v33_ft_rank", "v33_ft_pct_rank", "v33_ft_term_z"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

df["v33_score"] = v33["v33_ft_score"].astype("float32").fillna(0).to_numpy()
df["v33_pct"] = v33["v33_ft_pct_rank"].astype("float32").fillna(1).to_numpy()
df["v33_z"] = v33["v33_ft_term_z"].astype("float32").fillna(0).to_numpy()
df["v33_rs"] = (1.0 - df["v33_pct"]).astype("float32")
df["v33_zsig"] = sigmoid(df["v33_z"].to_numpy(dtype=np.float32) / 1.6).astype("float32")

score_min = float(df["v33_score"].min())
score_max = float(df["v33_score"].max())
df["v33_global"] = ((df["v33_score"] - score_min) / max(1e-6, score_max - score_min)).astype("float32")

print("loading old scores...")
df["v5"] = load_score(V5_SCORE, sample, ["proba_avg", "v5_score", "score", "prediction"])
df["v18"] = load_score(V18_SCORE, sample, ["v18_minilm_sigmoid", "v18_score", "score", "sigmoid"])
df["v21"] = load_score(V21_SCORE, sample, ["v21_score", "score", "prediction"])
df["v27"] = load_score(V27_SCORE, sample, ["v27_score", "score", "prediction"])

if SPARSE_SCORE.exists():
    sp = pd.read_parquet(SPARSE_SCORE)
    sp["id"] = sp["id"].astype(str)
    tmp = sample[["id"]].merge(
        sp[["id", "sparse_blend", "sparse_pct_rank"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    df["sparse_blend"] = tmp["sparse_blend"].astype("float32").fillna(0).to_numpy()
    df["sparse_pct_rank"] = tmp["sparse_pct_rank"].astype("float32").fillna(1).to_numpy()
else:
    df["sparse_blend"] = 0.0
    df["sparse_pct_rank"] = 1.0

for c in ["v5", "v18", "v21", "v27", "sparse_blend"]:
    df[f"{c}_rs"] = rank_score_by_term(df, c)

df["sparse_pct_rs"] = (1.0 - df["sparse_pct_rank"].astype("float32")).clip(0, 1)

print("loading predictions...")
pred_v13 = load_pred("v13", sample, required=False)
pred_v16 = load_pred("v16", sample, required=False)
pred_v19 = load_pred("v19p1", sample, required=False)
pred_v20 = load_pred("v20", sample, required=False)
pred_v22 = load_pred("v22", sample, required=True)
pred_v24 = load_pred("v24", sample, required=True)
pred_strict = load_pred("strict2000", sample, required=True)
pred_supported = load_pred("supported2000", sample, required=False)
pred_v32 = load_pred("v32", sample, required=False)

pred_list = []
for arr in [pred_v13, pred_v16, pred_v19, pred_v20, pred_v22, pred_v24, pred_strict, pred_supported, pred_v32]:
    if arr is not None:
        pred_list.append(arr)

vote_mean = np.vstack(pred_list).mean(axis=0).astype("float32")
df["vote_mean"] = vote_mean

# Composite score families.
v33_rs = df["v33_rs"].to_numpy(dtype=np.float32)
v33_zsig = df["v33_zsig"].to_numpy(dtype=np.float32)
v33_global = df["v33_global"].to_numpy(dtype=np.float32)
v5_rs = df["v5_rs"].to_numpy(dtype=np.float32)
v18_rs = df["v18_rs"].to_numpy(dtype=np.float32)
v21_rs = df["v21_rs"].to_numpy(dtype=np.float32)
v27_rs = df["v27_rs"].to_numpy(dtype=np.float32)
sp_rs = (
    0.70 * df["sparse_blend_rs"].to_numpy(dtype=np.float32)
    + 0.30 * df["sparse_pct_rs"].to_numpy(dtype=np.float32)
).astype(np.float32)

scores = {}

scores["v33_pure"] = (
    0.70 * v33_rs +
    0.20 * v33_zsig +
    0.10 * v33_global
).astype(np.float32)

scores["v33_safe_blend"] = (
    0.42 * v33_rs +
    0.16 * v33_zsig +
    0.14 * v5_rs +
    0.10 * v18_rs +
    0.08 * v21_rs +
    0.10 * vote_mean
).astype(np.float32)

scores["v33_oldvote_blend"] = (
    0.34 * v33_rs +
    0.12 * v33_zsig +
    0.16 * v5_rs +
    0.12 * v18_rs +
    0.10 * v21_rs +
    0.16 * vote_mean
).astype(np.float32)

scores["v33_sparse_blend"] = (
    0.42 * v33_rs +
    0.14 * v33_zsig +
    0.14 * sp_rs +
    0.12 * v5_rs +
    0.08 * v18_rs +
    0.10 * vote_mean
).astype(np.float32)

scores["v33_probe_blend"] = (
    0.55 * v33_rs +
    0.15 * v33_zsig +
    0.10 * v5_rs +
    0.08 * v18_rs +
    0.05 * v21_rs +
    0.07 * vote_mean
).astype(np.float32)

variants = {}
meta = []
swap_reports = []


def add_variant(name, pred, source, note):
    pred = pred.astype(np.int8)
    if name in variants:
        return

    variants[name] = pred

    meta.append({
        "variant": name,
        "source": source,
        "note": note,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v22": int((pred != pred_v22).sum()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
        "diff_vs_strict2000": int((pred != pred_strict).sum()),
        "diff_vs_v32": int((pred != pred_v32).sum()) if pred_v32 is not None else -1,
    })

    print("variant", name, "ones", int(pred.sum()), "diff_v24", int((pred != pred_v24).sum()))


def build_swap_pool(anchor_name, anchor_pred, score_name, score_arr, support_name, min_gain):
    base0 = anchor_pred == 0
    base1 = anchor_pred == 1

    if support_name == "all":
        add_mask = base0
        drop_mask = base1

    elif support_name == "v33_top20_dropweak":
        add_mask = base0 & (v33_rs >= 0.80)
        drop_mask = base1 & ((v33_rs <= 0.55) | (score_arr <= 0.50))

    elif support_name == "v33_top10_dropweak":
        add_mask = base0 & (v33_rs >= 0.90)
        drop_mask = base1 & ((v33_rs <= 0.60) | (score_arr <= 0.55))

    elif support_name == "hybrid_safe":
        add_mask = base0 & (
            (v33_rs >= 0.84)
            | ((score_arr >= 0.72) & (vote_mean >= 0.22))
            | ((v33_zsig >= 0.72) & (v5_rs >= 0.55))
        )
        drop_mask = base1 & (
            (v33_rs <= 0.62)
            | ((score_arr <= 0.46) & (vote_mean <= 0.55))
        )

    else:
        raise ValueError(support_name)

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

    pool = adds.merge(
        drops,
        on=["term_id", "pair_rank"],
        suffixes=("_add", "_drop"),
        how="inner",
    )

    pool["gain"] = pool["score_add"] - pool["score_drop"]
    pool = pool[pool["gain"] >= min_gain].copy()
    pool = pool.sort_values("gain", ascending=False).reset_index(drop=True)

    swap_reports.append({
        "anchor": anchor_name,
        "score_name": score_name,
        "support_name": support_name,
        "min_gain": min_gain,
        "available_swaps": int(len(pool)),
        "gain_mean": float(pool["gain"].mean()) if len(pool) else np.nan,
        "gain_min": float(pool["gain"].min()) if len(pool) else np.nan,
        "gain_max": float(pool["gain"].max()) if len(pool) else np.nan,
    })

    return pool


def apply_swaps(anchor_pred, pool, budget):
    take = pool.head(budget).copy()
    pred = anchor_pred.copy()

    add_idx = take["idx_add"].to_numpy(dtype=np.int32)
    drop_idx = take["idx_drop"].to_numpy(dtype=np.int32)

    pred[add_idx] = 1
    pred[drop_idx] = 0

    return pred.astype(np.int8), len(take)


# Baselines
add_variant("v22_base", pred_v22, "baseline", "public 0.80 anchor")
add_variant("v24_anchor", pred_v24, "baseline", "old anchor")
add_variant("v26_strict2000", pred_strict, "baseline", "sparse strict add")
if pred_v32 is not None:
    add_variant("v32_strict_qswap_prob_top400", pred_v32, "baseline", "latest failed public")

# Large query-level swaps.
anchors = {
    "v24": pred_v24,
    "strict": pred_strict,
}

budgets = [2500, 5000, 10000, 20000, 40000, 60000]
min_gains = [0.03, 0.06]
supports = ["v33_top20_dropweak", "v33_top10_dropweak", "hybrid_safe"]

for anchor_name, anchor_pred in anchors.items():
    for score_name, score_arr in scores.items():
        for support_name in supports:
            for min_gain in min_gains:
                pool = build_swap_pool(anchor_name, anchor_pred, score_name, score_arr, support_name, min_gain)

                if len(pool) < 500:
                    continue

                for budget in budgets:
                    if len(pool) < min(budget, 1000):
                        continue

                    pred, applied = apply_swaps(anchor_pred, pool, budget)
                    name = f"{anchor_name}_qswap_{score_name}_{support_name}_g{str(min_gain).replace('.', 'p')}_b{applied}"
                    add_variant(name, pred, "qswap", f"{anchor_name} quota, {score_name}, {support_name}")

# Add-only from strict/v24
for base_name, base_pred in [("v24", pred_v24), ("strict", pred_strict)]:
    base0 = base_pred == 0

    for score_name, score_arr in scores.items():
        add_candidates = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int32),
            "score": score_arr,
            "v33_rs": v33_rs,
            "vote_mean": vote_mean,
            "base0": base0,
        })

        add_candidates = add_candidates[
            add_candidates["base0"]
            & (
                (add_candidates["v33_rs"] >= 0.94)
                | ((add_candidates["score"] >= 0.78) & (add_candidates["vote_mean"] >= 0.20))
            )
        ].sort_values("score", ascending=False)

        for budget in [1000, 2500, 5000, 10000, 20000]:
            if len(add_candidates) < min(budget, 500):
                continue

            take = add_candidates.head(budget)
            pred = base_pred.copy()
            pred[take["idx"].to_numpy(dtype=np.int32)] = 1

            name = f"{base_name}_addonly_{score_name}_b{len(take)}"
            add_variant(name, pred, "addonly", f"{base_name} + high v33 adds")

# Remove-only from strict/v24
for base_name, base_pred in [("v24", pred_v24), ("strict", pred_strict)]:
    base1 = base_pred == 1

    for score_name, score_arr in scores.items():
        rem_candidates = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int32),
            "score": score_arr,
            "v33_rs": v33_rs,
            "vote_mean": vote_mean,
            "base1": base1,
        })

        rem_candidates = rem_candidates[
            rem_candidates["base1"]
            & (
                (rem_candidates["v33_rs"] <= 0.35)
                | ((rem_candidates["score"] <= 0.42) & (rem_candidates["vote_mean"] <= 0.55))
            )
        ].sort_values("score", ascending=True)

        for budget in [1000, 2500, 5000, 10000, 20000]:
            if len(rem_candidates) < min(budget, 500):
                continue

            take = rem_candidates.head(budget)
            pred = base_pred.copy()
            pred[take["idx"].to_numpy(dtype=np.int32)] = 0

            name = f"{base_name}_removeonly_{score_name}_b{len(take)}"
            add_variant(name, pred, "removeonly", f"{base_name} - low v33 positives")

summary = pd.DataFrame(meta)
summary.to_csv(OUT_SUMMARY, index=False)
pd.DataFrame(swap_reports).to_csv(OUT_SWAPS, index=False)

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
        combined_score = 0.60 * legacy_weighted + 0.25 * active_weighted + 0.15 * sparse_weighted
    elif active_wsum:
        combined_score = 0.70 * legacy_weighted + 0.30 * active_weighted
    else:
        combined_score = legacy_weighted

    s = summary[summary["variant"] == name].iloc[0]
    pos_ratio = float(s["pos_ratio"])
    diff_v24 = int(s["diff_vs_v24"])
    diff_strict = int(s["diff_vs_strict2000"])

    # Bu V33 public-probe olduğu için diff'i çok cezalandırmıyoruz,
    # ama pos_ratio sapması hâlâ risk.
    pos_penalty = abs(pos_ratio - anchor_pos) * 0.70
    huge_diff_penalty = max(0, diff_v24 - 90000) / 1200000.0

    public_probe_score = combined_score - pos_penalty - huge_diff_penalty

    main_eval = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "public_probe_score": public_probe_score,
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
agg = agg.sort_values(["public_probe_score", "combined_score", "legacy_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

# Save only top candidates to avoid disk explosion.
top_names = []

for col in ["public_probe_score", "combined_score", "legacy_weighted", "active_weighted", "sparse_weighted"]:
    if col in agg.columns:
        for nm in agg.sort_values(col, ascending=False).head(12)["variant"].tolist():
            if nm not in top_names:
                top_names.append(nm)

# Büyük public-probe adaylarından birkaçını zorla kaydet.
probe_rows = agg[
    (agg["diff_vs_v24"] >= 10000)
    & (agg["diff_vs_v24"] <= 70000)
].sort_values("public_probe_score", ascending=False).head(8)

for nm in probe_rows["variant"].tolist():
    if nm not in top_names:
        top_names.append(nm)

for nm in ["v22_base", "v24_anchor", "v26_strict2000", "v32_strict_qswap_prob_top400"]:
    if nm in variants and nm not in top_names:
        top_names.append(nm)

save_rows = []

for name in top_names[:20]:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v33_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name].astype(np.int8)}).to_csv(out, index=False)

    row = agg[agg["variant"] == name].iloc[0].to_dict()
    row["file"] = str(out)
    save_rows.append(row)

    print("saved top candidate:", out)

pd.DataFrame(save_rows).to_csv(ROOT / "reports/manual_review/v33_saved_candidates.csv", index=False)

print("\nTOP PUBLIC PROBE")
print(agg.head(80).to_string(index=False))

print("\nTOP LARGE PROBES")
print(
    agg[(agg["diff_vs_v24"] >= 10000) & (agg["diff_vs_v24"] <= 70000)]
    .sort_values("public_probe_score", ascending=False)
    .head(80)
    .to_string(index=False)
)

print("\nSAVED CANDIDATES")
print(pd.DataFrame(save_rows)[["variant", "file", "public_probe_score", "diff_vs_v24", "ones", "pos_ratio"]].to_string(index=False))

print("saved:", OUT_SUMMARY)
print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_SWAPS)
