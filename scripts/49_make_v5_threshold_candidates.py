from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SUB = ROOT / "submissions"
REP = ROOT / "reports" / "manual_review"
REP.mkdir(parents=True, exist_ok=True)

LABEL_PATH = REP / "manual_review_set_v1_assistant_labeled.csv"
SCORE_PATH = PROC / "v5_e5base_full900_test_proba.parquet"
SAMPLE_PATH = RAW / "sample_submission.csv"

def m(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro"),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "manual_pred_pos_ratio": float(p.mean()),
    }

def main():
    labels = pd.read_csv(LABEL_PATH)
    labels["id"] = labels["id"].astype(str)

    labels = labels[labels["assistant_label"].isin([0,1])].copy()
    labels["assistant_label"] = labels["assistant_label"].astype(int)

    clean = labels[
        (labels["needs_recheck"].fillna(0).astype(int) == 0)
        & (labels["assistant_confidence"].astype(str).str.lower().isin(["high", "medium"]))
    ].copy()

    scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
    scores["id"] = scores["id"].astype(str)

    full = labels.merge(scores, on="id", how="left", validate="many_to_one")
    clean = clean.merge(scores, on="id", how="left", validate="many_to_one")

    rows = []
    thresholds = np.round(np.arange(0.50, 0.971, 0.005), 3)

    for th in thresholds:
        for name, df in [("all", full), ("clean", clean)]:
            y = df["assistant_label"].to_numpy()
            p = (df["proba_avg"].to_numpy() >= th).astype(int)
            row = {"subset": name, "threshold": float(th)}
            row.update(m(y, p))
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(REP / "v5_threshold_manual_grid.csv", index=False)

    print("TOP CLEAN")
    print(result[result["subset"].eq("clean")].sort_values("macro_f1", ascending=False).head(20).to_string(index=False))

    print("\nTOP ALL")
    print(result[result["subset"].eq("all")].sort_values("macro_f1", ascending=False).head(20).to_string(index=False))

    # Submission candidates around useful thresholds
    sample = pd.read_csv(SAMPLE_PATH)
    test_scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
    assert test_scores["id"].astype(str).reset_index(drop=True).equals(sample["id"].astype(str).reset_index(drop=True))

    candidate_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    summary = []

    for th in candidate_thresholds:
        pred = (test_scores["proba_avg"].to_numpy() >= th).astype(np.int8)
        out = SUB / f"CANDIDATE_v5_full900_threshold_{str(th).replace('.', 'p')}.csv"
        pd.DataFrame({"id": test_scores["id"], "prediction": pred}).to_csv(out, index=False)
        summary.append({
            "file": str(out),
            "threshold": th,
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
        })

    s = pd.DataFrame(summary)
    s.to_csv(REP / "v5_threshold_submission_candidates_summary.csv", index=False)
    print("\nSUBMISSION CANDIDATES")
    print(s.to_string(index=False))

if __name__ == "__main__":
    main()
