from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")
OUT = ROOT / "reports/manual_review/final_candidate_validation_comparison.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1_assistant_full": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
}

SUB_FILES = {
    "base_0p7": ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_RANDOM_VALIDATED.csv",
    "gender_filtered": ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_GENDER_FILTERED.csv",
    "gender_safe_brand": ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_GENDER_SAFE_BRAND_FILTERED.csv",
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

rows = []

subs = {}
for name, path in SUB_FILES.items():
    s = pd.read_csv(path, usecols=["id", "prediction"])
    s["id"] = s["id"].astype(str)
    subs[name] = s

for label_name, label_path in LABEL_FILES.items():
    if not label_path.exists():
        print("missing label file:", label_path)
        continue

    df = pd.read_csv(label_path)
    df["id"] = df["id"].astype(str)

    if "assistant_label" not in df.columns:
        continue

    df = df[df["assistant_label"].isin([0, 1, "0", "1"])].copy()
    df["assistant_label"] = df["assistant_label"].astype(int)

    subsets = {"all": df}

    if "needs_recheck" in df.columns and "assistant_confidence" in df.columns:
        clean = df[
            (df["needs_recheck"].fillna(0).astype(int) == 0)
            & (df["assistant_confidence"].astype(str).str.lower().isin(["high", "medium"]))
        ].copy()
        high_clean = df[
            (df["needs_recheck"].fillna(0).astype(int) == 0)
            & (df["assistant_confidence"].astype(str).str.lower().eq("high"))
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

            y = m["assistant_label"].to_numpy()
            p = m["prediction"].to_numpy()

            row = {
                "label_set": label_name,
                "subset": subset_name,
                "submission": sub_name,
                "n": len(m),
            }
            row.update(metrics(y, p))
            rows.append(row)

res = pd.DataFrame(rows)
res = res.sort_values(["label_set", "subset", "macro_f1"], ascending=[True, True, False])
res.to_csv(OUT, index=False)

print(res.to_string(index=False))
print("saved:", OUT)
