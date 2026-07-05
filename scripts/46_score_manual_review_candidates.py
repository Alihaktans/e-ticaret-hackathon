from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "reports" / "manual_review" / "manual_review_set_v1.csv"
OUT_DIR = ROOT / "reports" / "manual_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_label(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"0", "0.0"}:
        return 0
    if s in {"1", "1.0"}:
        return 1
    return np.nan

def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1])),
        "positive_f1": float(f1_score(y, pred, pos_label=1)),
        "negative_f1": float(f1_score(y, pred, pos_label=0)),
        "precision_pos": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "pred_positive_ratio": float(np.mean(pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def main():
    df = pd.read_csv(INP)
    df["human_label_clean"] = df["human_label"].map(clean_label)
    labeled = df[df["human_label_clean"].isin([0, 1])].copy()
    labeled["human_label_clean"] = labeled["human_label_clean"].astype(int)

    print("Total rows:", len(df))
    print("Labeled rows:", len(labeled))
    print("Human label distribution:")
    print(labeled["human_label_clean"].value_counts(dropna=False))
    print("Human positive ratio:", labeled["human_label_clean"].mean())

    if len(labeled) < 300:
        print("WARNING: labeled sample is small; use results carefully.")

    y = labeled["human_label_clean"].to_numpy(dtype=np.int8)
    rows = []

    # Existing binary prediction columns
    pred_cols = [c for c in labeled.columns if c.startswith("pred_")]
    for col in pred_cols:
        pred = labeled[col].fillna(0).astype(int).to_numpy(dtype=np.int8)
        row = {"candidate": col}
        row.update(metrics(y, pred))
        rows.append(row)

    # Threshold sweep for score columns
    score_cols = [c for c in ["score_v10", "score_v5_full900"] if c in labeled.columns]
    for col in score_cols:
        score = pd.to_numeric(labeled[col], errors="coerce").fillna(-999).to_numpy()
        for th in np.round(np.arange(0.01, 0.991, 0.01), 3):
            pred = (score >= th).astype(np.int8)
            row = {"candidate": f"{col}_threshold_{th:.2f}"}
            row.update(metrics(y, pred))
            row["threshold"] = float(th)
            row["score_col"] = col
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["macro_f1", "positive_f1", "negative_f1"], ascending=False
    )

    out_csv = OUT_DIR / "manual_validation_scores_v1.csv"
    result.to_csv(out_csv, index=False)

    best = result.head(30)
    print("=" * 100)
    print("TOP 30")
    print(best.to_string(index=False))

    # Error analysis for best candidate
    best_name = result.iloc[0]["candidate"]
    if best_name.startswith("pred_"):
        best_pred = labeled[best_name].fillna(0).astype(int).to_numpy(dtype=np.int8)
    else:
        score_col = result.iloc[0]["score_col"]
        threshold = float(result.iloc[0]["threshold"])
        score = pd.to_numeric(labeled[score_col], errors="coerce").fillna(-999).to_numpy()
        best_pred = (score >= threshold).astype(np.int8)

    err = labeled.copy()
    err["best_pred"] = best_pred
    err["error_type"] = np.where(
        (err["human_label_clean"] == 1) & (err["best_pred"] == 0), "false_negative",
        np.where((err["human_label_clean"] == 0) & (err["best_pred"] == 1), "false_positive", "correct")
    )

    err_cols = [
        "error_type", "human_label_clean", "best_pred", "bucket", "id",
        "query", "title", "category", "brand", "gender", "age_group",
        "score_v10", "score_v5_full900"
    ]
    err_cols = [c for c in err_cols if c in err.columns]

    err_out = OUT_DIR / "manual_validation_errors_best_v1.csv"
    err[err["error_type"] != "correct"][err_cols].to_csv(err_out, index=False, encoding="utf-8-sig")

    report = {
        "labeled_rows": int(len(labeled)),
        "human_positive_ratio": float(labeled["human_label_clean"].mean()),
        "best_candidate": best_name,
        "best_metrics": result.iloc[0].to_dict(),
        "score_file": str(out_csv),
        "error_file": str(err_out),
    }

    out_json = OUT_DIR / "manual_validation_report_v1.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", out_csv)
    print("Saved:", err_out)
    print("Saved:", out_json)

if __name__ == "__main__":
    main()
