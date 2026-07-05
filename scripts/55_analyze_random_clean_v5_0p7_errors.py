from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
LABEL_PATH = ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv"
SCORE_PATH = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
OUT_DIR = ROOT / "reports/manual_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TH = 0.70

df = pd.read_csv(LABEL_PATH)
df["id"] = df["id"].astype(str)
df = df[df["assistant_label"].isin([0, 1, "0", "1"])].copy()
df["assistant_label"] = df["assistant_label"].astype(int)

scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

df = df.merge(scores, on="id", how="left", validate="many_to_one")
df["pred_0p7"] = (df["proba_avg"] >= TH).astype(int)

def err_type(row):
    y = row["assistant_label"]
    p = row["pred_0p7"]
    if y == 0 and p == 1:
        return "false_positive"
    if y == 1 and p == 0:
        return "false_negative"
    if y == 1 and p == 1:
        return "true_positive"
    return "true_negative"

df["error_type_0p7"] = df.apply(err_type, axis=1)

cols_first = [
    "error_type_0p7",
    "assistant_label",
    "pred_0p7",
    "proba_avg",
    "id",
    "query",
    "title",
    "category",
    "brand",
    "gender",
    "age_group",
    "attributes",
]

cols = [c for c in cols_first if c in df.columns] + [c for c in df.columns if c not in cols_first]

errors = df[df["error_type_0p7"].isin(["false_positive", "false_negative"])].copy()
fp = df[df["error_type_0p7"].eq("false_positive")].copy()
fn = df[df["error_type_0p7"].eq("false_negative")].copy()

errors[cols].sort_values(["error_type_0p7", "proba_avg"], ascending=[True, False]).to_csv(
    OUT_DIR / "random_clean_v5_0p7_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

fp[cols].sort_values("proba_avg", ascending=False).to_csv(
    OUT_DIR / "random_clean_v5_0p7_false_positives.csv",
    index=False,
    encoding="utf-8-sig",
)

fn[cols].sort_values("proba_avg", ascending=False).to_csv(
    OUT_DIR / "random_clean_v5_0p7_false_negatives.csv",
    index=False,
    encoding="utf-8-sig",
)

print("counts")
print(df["error_type_0p7"].value_counts().to_string())

print("\nfalse positives top")
print(fp[["proba_avg", "query", "title", "category", "brand", "gender"]].head(20).to_string(index=False))

print("\nfalse negatives top")
print(fn[["proba_avg", "query", "title", "category", "brand", "gender"]].head(20).to_string(index=False))

print("\nsaved:")
print(OUT_DIR / "random_clean_v5_0p7_errors.csv")
print(OUT_DIR / "random_clean_v5_0p7_false_positives.csv")
print(OUT_DIR / "random_clean_v5_0p7_false_negatives.csv")
