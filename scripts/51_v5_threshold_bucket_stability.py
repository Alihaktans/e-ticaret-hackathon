from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")
LABEL_PATH = ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv"
SCORE_PATH = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
OUT = ROOT / "reports/manual_review/v5_threshold_bucket_stability.csv"

thresholds = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

def metric(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1]),
        "positive_f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, pred, pos_label=0, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "pred_pos_ratio": float(pred.mean()),
        "n": int(len(y)),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

labels = pd.read_csv(LABEL_PATH)
labels["id"] = labels["id"].astype(str)

scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

df = labels.merge(scores, on="id", how="left", validate="many_to_one")
df = df[df["assistant_label"].isin([0, 1])].copy()
df["assistant_label"] = df["assistant_label"].astype(int)

subsets = []

subsets.append(("all", df))

clean = df[
    (df["needs_recheck"].fillna(0).astype(int) == 0)
    & (df["assistant_confidence"].astype(str).str.lower().isin(["high", "medium"]))
].copy()
subsets.append(("clean", clean))

high_clean = df[
    (df["needs_recheck"].fillna(0).astype(int) == 0)
    & (df["assistant_confidence"].astype(str).str.lower().eq("high"))
].copy()
subsets.append(("high_clean", high_clean))

if "human_label" in df.columns:
    human = df[df["human_label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])].copy()
    human["human_label_clean"] = human["human_label"].astype(float).astype(int)
    human_agree = human[human["human_label_clean"].eq(human["assistant_label"])].copy()
    subsets.append(("human_assistant_agree", human_agree))

if "bucket" in df.columns:
    for bucket, part in df.groupby("bucket"):
        if len(part) >= 40 and part["assistant_label"].nunique() == 2:
            subsets.append((f"bucket__{bucket}", part.copy()))

rows = []

for subset_name, part in subsets:
    if len(part) < 30 or part["assistant_label"].nunique() < 2:
        continue

    y = part["assistant_label"].to_numpy()
    s = part["proba_avg"].to_numpy()

    for th in thresholds:
        pred = (s >= th).astype(int)
        row = {
            "subset": subset_name,
            "threshold": th,
        }
        row.update(metric(y, pred))
        rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

summary = (
    res.groupby("threshold")
    .agg(
        mean_macro_f1=("macro_f1", "mean"),
        min_macro_f1=("macro_f1", "min"),
        mean_negative_f1=("negative_f1", "mean"),
        min_negative_f1=("negative_f1", "min"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        subset_count=("subset", "nunique"),
    )
    .reset_index()
    .sort_values(["min_macro_f1", "mean_macro_f1"], ascending=False)
)

print("STABILITY SUMMARY")
print(summary.to_string(index=False))
print("\nSaved:", OUT)
