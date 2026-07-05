from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

V5_SUB = ROOT / "submissions/FINAL_MAIN_v5_full900_0p6_GENDER_SAFE_BRAND.csv"
V13_SUB = ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

V5_SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
CE_SCORE = ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet"

CHANGE_LABELS = ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv"

OUT_SUMMARY = ROOT / "reports/manual_review/v14_hybrid_candidate_summary.csv"

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1]),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
    }

print("loading submissions...")
v5 = pd.read_csv(V5_SUB)
v13 = pd.read_csv(V13_SUB)
sample = pd.read_csv(SAMPLE)

assert v5["id"].astype(str).equals(sample["id"].astype(str))
assert v13["id"].astype(str).equals(sample["id"].astype(str))

v5["id"] = v5["id"].astype(str)
v13["id"] = v13["id"].astype(str)

base_pred = v5["prediction"].astype(np.int8).to_numpy()
v13_pred = v13["prediction"].astype(np.int8).to_numpy()

scores = pd.read_parquet(V5_SCORE, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

ce = pd.read_parquet(CE_SCORE, columns=["id", "cross_sigmoid"])
ce["id"] = ce["id"].astype(str)

score_df = scores.merge(ce, on="id", how="left", validate="one_to_one")
score_df["cross_sigmoid"] = score_df["cross_sigmoid"].fillna(-1.0)

v5_score = score_df["proba_avg"].to_numpy()
ce_score = score_df["cross_sigmoid"].to_numpy()
blend025 = 0.25 * v5_score + 0.75 * ce_score

# v13 - v5 değişiklik maskeleri
added_by_v13 = (base_pred == 0) & (v13_pred == 1)
removed_by_v13 = (base_pred == 1) & (v13_pred == 0)

variants = []

# V5 baseline
variants.append(("v5_base", base_pred.copy()))

# v13 full
variants.append(("v13_full_aggressive", v13_pred.copy()))

# Hybrid: v13 removal'larını hep uygula, addition'ları seçici uygula
for ce_th in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    pred = base_pred.copy()
    pred[removed_by_v13] = 0
    add_mask = added_by_v13 & (ce_score >= ce_th)
    pred[add_mask] = 1
    variants.append((f"hybrid_remove_all_add_ce_ge_{str(ce_th).replace('.', 'p')}", pred))

for blend_th in [0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45, 0.50]:
    pred = base_pred.copy()
    pred[removed_by_v13] = 0
    add_mask = added_by_v13 & (blend025 >= blend_th)
    pred[add_mask] = 1
    variants.append((f"hybrid_remove_all_add_blend025_ge_{str(blend_th).replace('.', 'p')}", pred))

# Hybrid daha temkinli: sadece CE çok düşükse V13 removal uygula
for remove_ce_max in [0.10, 0.20, 0.30, 0.40]:
    for add_ce_min in [0.30, 0.50, 0.70]:
        pred = base_pred.copy()
        rem_mask = removed_by_v13 & (ce_score <= remove_ce_max)
        add_mask = added_by_v13 & (ce_score >= add_ce_min)
        pred[rem_mask] = 0
        pred[add_mask] = 1
        variants.append((f"hybrid_rem_ce_le_{str(remove_ce_max).replace('.', 'p')}_add_ce_ge_{str(add_ce_min).replace('.', 'p')}", pred))

rows = []

# Full submission summaries
for name, pred in variants:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v14_{name}.csv"
    pd.DataFrame({"id": v5["id"], "prediction": pred}).to_csv(out, index=False)

    rows.append({
        "eval_set": "full_test_summary",
        "variant": name,
        "file": str(out),
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v5": int((pred != base_pred).sum()),
        "diff_vs_v13": int((pred != v13_pred).sum()),
    })

# Change review labels ile ölç
if CHANGE_LABELS.exists():
    lab = pd.read_csv(CHANGE_LABELS)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)

    id_to_idx = pd.Series(np.arange(len(v5)), index=v5["id"].astype(str))
    idx = lab["id"].map(id_to_idx).dropna().astype(int).to_numpy()
    y = lab.loc[lab["id"].isin(id_to_idx.index), "assistant_label"].to_numpy()

    for name, pred in variants:
        p = pred[idx]
        row = {
            "eval_set": "change_review_500",
            "variant": name,
            "file": "",
            "ones": "",
            "pos_ratio": "",
            "diff_vs_v5": "",
            "diff_vs_v13": "",
        }
        row.update(metrics(y, p))
        rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(OUT_SUMMARY, index=False)

print("TOP change_review_500")
if CHANGE_LABELS.exists():
    x = summary[summary["eval_set"].eq("change_review_500")].sort_values("macro_f1", ascending=False)
    print(x.head(30).to_string(index=False))
else:
    print("change labels missing:", CHANGE_LABELS)

print("\nFULL TEST SUMMARY")
print(summary[summary["eval_set"].eq("full_test_summary")].head(40).to_string(index=False))
print("saved:", OUT_SUMMARY)
