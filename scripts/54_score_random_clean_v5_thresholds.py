from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")
LABEL_PATH = ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv"
SCORE_PATH = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
OUT = ROOT / "reports/manual_review/random_clean_v5_threshold_metrics.csv"

thresholds = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.955]

def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1]),
        "positive_f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, pred, pos_label=0, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_pos_ratio": float(pred.mean()),
    }

df = pd.read_csv(LABEL_PATH)
df["id"] = df["id"].astype(str)

if "assistant_label" not in df.columns:
    raise SystemExit("assistant_label kolonu yok.")

df = df[df["assistant_label"].isin([0, 1, "0", "1"])].copy()
df["assistant_label"] = df["assistant_label"].astype(int)

scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

df = df.merge(scores, on="id", how="left", validate="many_to_one")
df = df.dropna(subset=["proba_avg"]).copy()

y = df["assistant_label"].to_numpy()
s = df["proba_avg"].to_numpy()

rows = []

for th in thresholds:
    pred = (s >= th).astype(int)
    row = {
        "threshold": th,
        "n": len(df),
        "true_pos_ratio": float(y.mean()),
    }
    row.update(metrics(y, pred))
    rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

print("LABEL SUMMARY")
print("n:", len(df))
print("true_pos_ratio:", y.mean())
print(pd.Series(y).value_counts().sort_index().to_string())

print("\nTHRESHOLD METRICS")
print(res.sort_values("macro_f1", ascending=False).to_string(index=False))

print("\nSaved:", OUT)
