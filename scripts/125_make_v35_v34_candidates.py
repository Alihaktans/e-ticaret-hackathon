from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"

V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"

OUT_SUMMARY = ROOT / "reports/manual_review/v35_v34_candidate_summary.csv"
OUT_EVAL = ROOT / "reports/manual_review/v35_v34_candidate_eval.csv"
OUT_SAVED = ROOT / "reports/manual_review/v35_v34_saved_candidates.csv"
OUT_SWAP_POOLS = ROOT / "reports/manual_review/v35_v34_swap_pool_summary.csv"

SUB_DIR = ROOT / "submissions"

PRED_PATHS = {
    "v13": [
        ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
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
    "v26_strict": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget2000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v31_v26_strict2000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v32_v26_strict2000.csv",
    ],
    "v33_qprob2000": [
        ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
    ],
    "v33_qprob065": [
        ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprobge0p65_utility_all2621.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprobge0p65_utility_all2621.csv",
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
        .replace("(", "")
        .replace(")", "")
    )


def sigmoid(x):
    x = np.clip(x, -12, 12)
    return 1.0 / (1.0 + np.exp(-x))


def load_pred(name, sample, required=False):
    p = first_existing(PRED_PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(name)
        print("missing optional prediction:", name)
        return None

    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)

    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {name} {p}")

    arr = d["prediction"].astype(np.int8).to_numpy()
    print("loaded", name, p, "ones", int(arr.sum()))
    return arr


def align_score(score_path, sample, cols):
    d = pd.read_parquet(score_path)
    d["id"] = d["id"].astype(str)

    if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        return d[cols].copy()

    tmp = sample[["id"]].merge(d[["id"] + cols], on="id", how="left", validate="one_to_one")
    return tmp[cols].copy()


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1], zero_division=0),
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
SUB_DIR.mkdir(parents=True, exist_ok=True)

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

df = pairs.copy()
df["candidate_count"] = df.groupby("term_id")["id"].transform("count").astype(np.int32)

print("loading predictions...")
pred_v24 = load_pred("v24", sample, required=True)
pred_v22 = load_pred("v22", sample, required=False)
pred_v13 = load_pred("v13", sample, required=False)
pred_v26 = load_pred("v26_strict", sample, required=False)
pred_v33_2000 = load_pred("v33_qprob2000", sample, required=False)
pred_v33_065 = load_pred("v33_qprob065", sample, required=False)

preds_for_vote = [x for x in [pred_v13, pred_v22, pred_v24, pred_v26, pred_v33_2000, pred_v33_065] if x is not None]
vote_mean = np.vstack(preds_for_vote).mean(axis=0).astype(np.float32)

print("loading V33/V34 scores...")
v33_cols = ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"]
v34_cols = ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"]

v33 = align_score(V33, sample, v33_cols)
v34 = align_score(V34, sample, v34_cols)

for c in v33_cols:
    df[c] = pd.to_numeric(v33[c], errors="coerce").fillna(0).astype(np.float32)

for c in v34_cols:
    df[c] = pd.to_numeric(v34[c], errors="coerce").fillna(0).astype(np.float32)

df["v33_rs"] = (1.0 - df["v33_ft_pct_rank"]).clip(0, 1).astype(np.float32)
df["v34_rs"] = (1.0 - df["v34_ce_pct_rank"]).clip(0, 1).astype(np.float32)

df["v33_zsig"] = sigmoid(df["v33_ft_term_z"].to_numpy(np.float32) / 1.8).astype(np.float32)
df["v34_zsig"] = sigmoid(df["v34_ce_term_z"].to_numpy(np.float32) / 1.8).astype(np.float32)

df["vote_mean"] = vote_mean

v33_rs = df["v33_rs"].to_numpy(np.float32)
v34_rs = df["v34_rs"].to_numpy(np.float32)
v33_zsig = df["v33_zsig"].to_numpy(np.float32)
v34_zsig = df["v34_zsig"].to_numpy(np.float32)
v34_raw = df["v34_ce_score"].to_numpy(np.float32)

score_map = {}

score_map["v34_ce_rank"] = (
    0.62 * v34_rs +
    0.23 * v34_zsig +
    0.15 * v34_raw
).astype(np.float32)

score_map["v34_v33_balanced"] = (
    0.42 * v34_rs +
    0.33 * v33_rs +
    0.11 * v34_zsig +
    0.08 * v33_zsig +
    0.06 * vote_mean
).astype(np.float32)

score_map["v34_precision"] = (
    0.52 * v34_rs +
    0.24 * v34_raw +
    0.16 * v34_zsig +
    0.08 * vote_mean
).astype(np.float32)

score_map["v34_v33_precision"] = (
    0.44 * v34_rs +
    0.24 * v33_rs +
    0.14 * v34_raw +
    0.10 * v34_zsig +
    0.08 * vote_mean
).astype(np.float32)

score_map["v34_v33_aggressive"] = (
    0.50 * v34_rs +
    0.27 * v33_rs +
    0.13 * v34_zsig +
    0.06 * v33_zsig +
    0.04 * vote_mean
).astype(np.float32)

for k, v in score_map.items():
    df[k] = v

variants = {}
meta_rows = []
swap_rows = []


def add_variant(name, pred, source, note):
    if name in variants:
        return

    pred = pred.astype(np.int8)
    variants[name] = pred

    meta_rows.append({
        "variant": name,
        "source": source,
        "note": note,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
        "diff_vs_v22": int((pred != pred_v22).sum()) if pred_v22 is not None else -1,
        "diff_vs_v33_2000": int((pred != pred_v33_2000).sum()) if pred_v33_2000 is not None else -1,
    })

    print("variant", name, "ones", int(pred.sum()), "diff_vs_v24", int((pred != pred_v24).sum()))


def quota_rerank(anchor_name, anchor_pred, score_name, score):
    quota = pd.DataFrame({
        "term_id": df["term_id"].to_numpy(),
        "anchor_pred": anchor_pred,
    }).groupby("term_id")["anchor_pred"].sum()

    tmp = pd.DataFrame({
        "term_id": df["term_id"].to_numpy(),
        "score": score,
    })

    tmp["quota"] = tmp["term_id"].map(quota).fillna(0).astype(np.int32)
    tmp["rank"] = tmp.groupby("term_id")["score"].rank(method="first", ascending=False).astype(np.int32)

    pred = (tmp["rank"].to_numpy() <= tmp["quota"].to_numpy()).astype(np.int8)
    add_variant(f"quota_{anchor_name}_{score_name}", pred, "quota_rerank", f"{anchor_name} quota rerank by {score_name}")


def build_swap_pool(anchor_name, anchor_pred, score_name, score, mode, min_gain):
    base0 = anchor_pred == 0
    base1 = anchor_pred == 1

    if mode == "wide":
        add_mask = base0 & (v34_rs >= 0.62)
        drop_mask = base1 & (v34_rs <= 0.78)

    elif mode == "balanced":
        add_mask = base0 & ((v34_rs >= 0.72) | ((v34_rs >= 0.62) & (v33_rs >= 0.70)))
        drop_mask = base1 & ((v34_rs <= 0.65) | ((v34_rs <= 0.75) & (v33_rs <= 0.55)))

    elif mode == "precision":
        add_mask = base0 & (v34_rs >= 0.80) & ((v33_rs >= 0.55) | (v34_raw >= 0.50))
        drop_mask = base1 & ((v34_rs <= 0.58) | ((v34_rs <= 0.68) & (v33_rs <= 0.50)))

    elif mode == "ce_only_strict":
        add_mask = base0 & (v34_rs >= 0.88)
        drop_mask = base1 & (v34_rs <= 0.62)

    else:
        raise ValueError(mode)

    tmp = pd.DataFrame({
        "term_id": df["term_id"].to_numpy(),
        "idx": np.arange(n, dtype=np.int32),
        "score": score.astype(np.float32),
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

    swap_rows.append({
        "anchor": anchor_name,
        "score_name": score_name,
        "mode": mode,
        "min_gain": min_gain,
        "available_swaps": int(len(pool)),
        "gain_mean": float(pool["gain"].mean()) if len(pool) else np.nan,
        "gain_min": float(pool["gain"].min()) if len(pool) else np.nan,
        "gain_max": float(pool["gain"].max()) if len(pool) else np.nan,
    })

    return pool


def apply_swaps(anchor_name, anchor_pred, score_name, score, mode, min_gain, budgets):
    pool = build_swap_pool(anchor_name, anchor_pred, score_name, score, mode, min_gain)

    if len(pool) < 100:
        return

    for b in budgets:
        if len(pool) < min(b, 500):
            continue

        take = pool.head(b).copy()
        pred = anchor_pred.copy()

        pred[take["idx_add"].to_numpy(np.int32)] = 1
        pred[take["idx_drop"].to_numpy(np.int32)] = 0

        name = f"{anchor_name}_qswap_{score_name}_{mode}_g{str(min_gain).replace('.', 'p')}_b{len(take)}"
        add_variant(name, pred, "qswap", f"{anchor_name}, {score_name}, {mode}")


add_variant("v24_anchor", pred_v24, "baseline", "v24 anchor")

if pred_v22 is not None:
    add_variant("v22_base", pred_v22, "baseline", "v22")

if pred_v33_2000 is not None:
    add_variant("v33_qprob2000", pred_v33_2000, "baseline", "v33 qprob top2000")

if pred_v33_065 is not None:
    add_variant("v33_qprob065", pred_v33_065, "baseline", "v33 qprob >= .65 utility")

anchors = {"v24": pred_v24}

if pred_v33_2000 is not None:
    anchors["v33q2000"] = pred_v33_2000

for anchor_name, anchor_pred in anchors.items():
    for score_name, score in score_map.items():
        quota_rerank(anchor_name, anchor_pred, score_name, score)

budgets = [1000, 2500, 5000, 10000, 20000, 40000, 60000]
modes = ["ce_only_strict", "precision", "balanced", "wide"]
min_gains = [0.03, 0.06, 0.10]

for anchor_name, anchor_pred in anchors.items():
    for score_name, score in score_map.items():
        for mode in modes:
            for min_gain in min_gains:
                apply_swaps(anchor_name, anchor_pred, score_name, score, mode, min_gain, budgets)

summary = pd.DataFrame(meta_rows)
pd.DataFrame(swap_rows).to_csv(OUT_SWAP_POOLS, index=False)

eval_rows = []

for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label:", label_set, path)
        continue

    lab = pd.read_csv(path)

    if "id" not in lab.columns or "assistant_label" not in lab.columns:
        continue

    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()

    if len(lab) < 30:
        continue

    lab["assistant_label"] = lab["assistant_label"].astype(int)

    subsets = {"all": lab}

    if "needs_recheck" in lab.columns:
        nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
        subsets["clean"] = lab[nr == 0].copy()

    if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
        subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()
        subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()

    if label_set in ["v20_active", "v21_active", "v26_sparse"] and "review_bucket" in lab.columns:
        for b, g in lab.groupby("review_bucket"):
            if len(g) >= 20 and g["assistant_label"].nunique() >= 2:
                subsets[f"bucket_{b}"] = g.copy()

    for subset_name, part in subsets.items():
        if len(part) < 30 or part["assistant_label"].nunique() < 2:
            continue

        idx = part["id"].map(id_to_idx)
        ok = idx.notna()

        if ok.sum() < 30:
            continue

        idx = idx[ok].astype(int).to_numpy()
        y = part.loc[ok, "assistant_label"].astype(int).to_numpy()

        for name, pred in variants.items():
            p = pred[idx]
            row = {
                "label_set": label_set,
                "subset": subset_name,
                "eval_key": f"{label_set}_{subset_name}",
                "variant": name,
                "n": int(len(y)),
            }
            row.update(metrics(y, p))
            eval_rows.append(row)

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT_EVAL, index=False)

weights = {
    "random_clean_v2_all": 0.26,
    "random_clean_v2_clean": 0.26,

    "manual_v1_clean": 0.25,
    "manual_v1_high_clean": 0.12,
    "manual_v1_high_medium_clean": 0.10,

    "review_v15_vs_v13_clean": 0.13,
    "review_v15_vs_v13_high_medium_clean": 0.11,
    "review_v13_vs_v5_clean": 0.05,

    "v20_active_clean": 0.04,
    "v21_active_clean": 0.09,
    "v21_active_high_medium_clean": 0.07,

    "v26_sparse_clean": 0.07,
    "v26_sparse_high_medium_clean": 0.05,
}

agg_rows = []
anchor_ratio = float(pred_v24.mean())

for name, g in eval_df.groupby("variant"):
    wsum = 0.0
    score = 0.0
    used_vals = []

    for _, r in g.iterrows():
        w = weights.get(r["eval_key"], 0.0)
        if w:
            wsum += w
            score += w * r["macro_f1"]
            used_vals.append(float(r["macro_f1"]))

    if wsum == 0:
        continue

    weighted_macro = score / wsum
    s = summary[summary["variant"] == name].iloc[0]

    diff = int(s["diff_vs_v24"])
    pos_ratio = float(s["pos_ratio"])

    pos_penalty = abs(pos_ratio - anchor_ratio) * 0.75
    huge_diff_penalty = max(0, diff - 90000) / 900000.0
    impact_bonus = min(0.012, np.log1p(diff) / np.log1p(60000) * 0.012)

    public_break_score = weighted_macro - pos_penalty - huge_diff_penalty + impact_bonus

    main_eval = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "public_break_score": float(public_break_score),
        "weighted_macro": float(weighted_macro),
        "used_min_macro": float(np.min(used_vals)) if used_vals else np.nan,
        "main_min_macro": float(main_eval["macro_f1"].min()) if len(main_eval) else np.nan,
        "main_mean_macro": float(main_eval["macro_f1"].mean()) if len(main_eval) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows)
agg = agg.merge(summary, on="variant", how="left")
agg = agg.sort_values(["public_break_score", "weighted_macro"], ascending=False)
agg.to_csv(OUT_SUMMARY, index=False)

save_names = []

for nm in agg.head(20)["variant"].tolist():
    if nm not in save_names:
        save_names.append(nm)

large = agg[
    (agg["source"] == "quota_rerank")
    & (agg["diff_vs_v24"] >= 10000)
    & (agg["diff_vs_v24"] <= 150000)
].sort_values("public_break_score", ascending=False).head(10)

for nm in large["variant"].tolist():
    if nm not in save_names:
        save_names.append(nm)

mid = agg[
    (agg["source"] == "qswap")
    & (agg["diff_vs_v24"] >= 5000)
    & (agg["diff_vs_v24"] <= 50000)
].sort_values("public_break_score", ascending=False).head(10)

for nm in mid["variant"].tolist():
    if nm not in save_names:
        save_names.append(nm)

for nm in ["v24_anchor", "v33_qprob2000", "v33_qprob065"]:
    if nm in variants and nm not in save_names:
        save_names.append(nm)

saved_rows = []

for nm in save_names[:35]:
    pred = variants[nm]
    out = SUB_DIR / f"FINAL_CANDIDATE_v35_{clean_name(nm)}.csv"

    pd.DataFrame({
        "id": ids,
        "prediction": pred.astype(np.int8),
    }).to_csv(out, index=False)

    row = agg[agg["variant"] == nm].iloc[0].to_dict()
    row["file"] = str(out)
    saved_rows.append(row)

    print("saved:", out)

saved = pd.DataFrame(saved_rows)
saved.to_csv(OUT_SAVED, index=False)

print("\nTOP V35")
cols = [
    "variant",
    "public_break_score",
    "weighted_macro",
    "source",
    "ones",
    "pos_ratio",
    "diff_vs_v24",
    "diff_vs_v33_2000",
    "main_min_macro",
    "main_mean_macro",
    "mean_precision",
    "mean_recall",
]
print(agg[cols].head(80).to_string(index=False))

print("\nTOP LARGE QUOTA")
print(
    agg[(agg["source"] == "quota_rerank")]
    .sort_values("public_break_score", ascending=False)
    [cols]
    .head(40)
    .to_string(index=False)
)

print("\nTOP MID QSWAP")
print(
    agg[
        (agg["source"] == "qswap")
        & (agg["diff_vs_v24"] >= 5000)
        & (agg["diff_vs_v24"] <= 50000)
    ].sort_values("public_break_score", ascending=False)
    [cols]
    .head(60)
    .to_string(index=False)
)

print("\nSAVED")
print(saved[["variant", "file", "public_break_score", "weighted_macro", "source", "diff_vs_v24", "ones", "pos_ratio"]].to_string(index=False))

print("\noutputs:")
print(OUT_SUMMARY)
print(OUT_EVAL)
print(OUT_SAVED)
print(OUT_SWAP_POOLS)
