from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")
OUT = ROOT / "reports/manual_review/v14_final_candidates_validation_comparison.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
}

SUB_FILES = {
    "v5_0p6_gsb": ROOT / "submissions/FINAL_MAIN_v5_full900_0p6_GENDER_SAFE_BRAND.csv",
    "v13_aggressive": ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
    "v14_blend025_ge_0p45": ROOT / "submissions/FINAL_MAIN_v14_hybrid_blend025_ge_0p45.csv",
    "v14_blend025_ge_0p4": ROOT / "submissions/FINAL_CANDIDATE_v14_hybrid_remove_all_add_blend025_ge_0p4.csv",
    "v14_ce_ge_0p4": ROOT / "submissions/FINAL_CANDIDATE_v14_hybrid_remove_all_add_ce_ge_0p4.csv",
    "v14_ce_ge_0p5": ROOT / "submissions/FINAL_CANDIDATE_v14_hybrid_remove_all_add_ce_ge_0p5.csv",
}

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

subs = {}
for name, path in SUB_FILES.items():
    if not path.exists():
        print("missing:", path)
        continue
    df = pd.read_csv(path, usecols=["id", "prediction"])
    df["id"] = df["id"].astype(str)
    subs[name] = df

rows = []

for label_name, path in LABEL_FILES.items():
    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)

    subsets = {"all": lab}

    if "needs_recheck" in lab.columns and "assistant_confidence" in lab.columns:
        clean = lab[
            (lab["needs_recheck"].fillna(0).astype(int) == 0)
            & (lab["assistant_confidence"].astype(str).str.lower().isin(["high", "medium"]))
        ].copy()
        high_clean = lab[
            (lab["needs_recheck"].fillna(0).astype(int) == 0)
            & (lab["assistant_confidence"].astype(str).str.lower().eq("high"))
        ].copy()
        subsets["clean"] = clean
        subsets["high_clean"] = high_clean

    for subset_name, part in subsets.items():
        if len(part) < 30 or part["assistant_label"].nunique() < 2:
            continue

        for sub_name, sub in subs.items():
            m = part[["id", "assistant_label"]].merge(sub, on="id", how="left", validate="many_to_one")
            m = m.dropna(subset=["prediction"]).copy()
            m["prediction"] = m["prediction"].astype(int)

            row = {
                "label_set": label_name,
                "subset": subset_name,
                "submission": sub_name,
                "n": len(m),
            }
            row.update(metrics(m["assistant_label"].to_numpy(), m["prediction"].to_numpy()))
            rows.append(row)

res = pd.DataFrame(rows)
res = res.sort_values(["label_set", "subset", "macro_f1"], ascending=[True, True, False])
res.to_csv(OUT, index=False)

print(res.to_string(index=False))
print("saved:", OUT)
