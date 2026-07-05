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
}

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
}

OUT_EVAL = ROOT / "reports/manual_review/v18_minilm_validation_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v18_minilm_validation_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v18_minilm_full_summary.csv"

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def clean_name(x):
    return x.replace(".", "p").replace(" ", "_").replace("/", "_").replace("=", "").replace("<", "lt").replace(">", "gt")

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
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

print("loading sample...")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)

def load_sub(name):
    p = first_existing(SUB_PATHS[name])
    if p is None:
        print("missing submission:", name)
        return None
    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)
    assert d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)), name
    return d["prediction"].astype(np.int8).to_numpy()

pred_v13 = load_sub("v13")
pred_v16 = load_sub("v16")
pred_v17p4 = load_sub("v17p4")

if pred_v13 is None:
    raise FileNotFoundError("v13 submission missing")
if pred_v16 is None:
    raise FileNotFoundError("v16 submission missing")
if pred_v17p4 is None:
    pred_v17p4 = pred_v16.copy()

print("loading scores...")
v5 = pd.read_parquet(V5, columns=["id", "proba_avg"])
v5["id"] = v5["id"].astype(str)
assert v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))
v5_score = v5["proba_avg"].astype("float32").to_numpy()

v18 = pd.read_parquet(V18)
v18["id"] = v18["id"].astype(str)

id_to_idx = pd.Series(np.arange(n), index=sample["id"])

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
add_variant("v17p4_independent", pred_v17p4.copy())

# Pure v18 only inside candidate band
for th in [0.10, 0.20, 0.30, 0.375, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]:
    pred = np.zeros(n, dtype=np.int8)
    pred[v5_score > 0.80] = 1
    pred[has_v18] = (v18_score[has_v18] >= th).astype(np.int8)
    add_variant(f"v18_pureband_t{th}", pred)

# V5 + v18 blend
for w in [0.15, 0.25, 0.35, 0.50]:
    blended = w * v5_score + (1.0 - w) * np.where(has_v18, v18_score, v5_score)
    for th in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        pred = np.zeros(n, dtype=np.int8)
        pred[v5_score > 0.80] = 1
        pred[has_v18] = (blended[has_v18] >= th).astype(np.int8)
        add_variant(f"v18_blend_v5w{w}_t{th}", pred)

# v16 base + v18 gated patch
for add_th in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    for rem_th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        pred = pred_v16.copy()
        pred[(pred_v16 == 0) & has_v18 & (v18_score >= add_th)] = 1
        pred[(pred_v16 == 1) & has_v18 & (v18_score <= rem_th)] = 0
        add_variant(f"v18_patch_v16_add{add_th}_rem{rem_th}", pred)

# add-only / remove-only
for add_th in [0.80, 0.85, 0.90, 0.95]:
    pred = pred_v16.copy()
    pred[(pred_v16 == 0) & has_v18 & (v18_score >= add_th)] = 1
    add_variant(f"v18_v16_addonly_{add_th}", pred)

for rem_th in [0.05, 0.10, 0.15, 0.20, 0.25]:
    pred = pred_v16.copy()
    pred[(pred_v16 == 1) & has_v18 & (v18_score <= rem_th)] = 0
    add_variant(f"v18_v16_remonly_{rem_th}", pred)

print("variants:", len(variants))

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
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

eval_rows = []

for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label:", path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)

    subsets = {"all": lab}

    if "needs_recheck" in lab.columns:
        nr = lab["needs_recheck"].fillna(0).astype(int)
        subsets["clean"] = lab[nr == 0].copy()

    if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        nr = lab["needs_recheck"].fillna(0).astype(int)
        subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()
        subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()

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

weights = {
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

agg_rows = []
for name, g in eval_df.groupby("variant"):
    score = 0.0
    wsum = 0.0
    vals = []

    for _, r in g.iterrows():
        w = weights.get(r["eval_key"], 0.0)
        if w > 0:
            score += w * r["macro_f1"]
            wsum += w
            vals.append(r["macro_f1"])

    if wsum == 0:
        continue

    main = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "weighted_macro": score / wsum,
        "min_macro_used": float(np.min(vals)),
        "mean_macro_used": float(np.mean(vals)),
        "main_min_macro": float(main["macro_f1"].min()) if len(main) else np.nan,
        "main_mean_macro": float(main["macro_f1"].mean()) if len(main) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "mean_pred_pos_ratio": float(g["pred_pos_ratio"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows)
agg = agg.merge(full, on="variant", how="left")
agg = agg.sort_values(["weighted_macro", "main_min_macro", "main_mean_macro"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP")
print(agg.head(50).to_string(index=False))

print("\nFULL TOP diff_vs_v16")
print(full.sort_values("diff_vs_v16").head(50).to_string(index=False))

top_names = agg.head(8)["variant"].tolist()
for b in ["v13_public_0p77", "v16_current", "v17p4_independent"]:
    if b not in top_names:
        top_names.append(b)

for name in top_names:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v18_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
