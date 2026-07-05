from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SUB = ROOT / "submissions"
MANUAL = ROOT / "reports" / "manual_review" / "manual_review_set_v1.csv"
OUT_DIR = ROOT / "reports" / "manual_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BINARY_CANDIDATES = {
    "old_v5_failed_public_0p58": SUB / "FINAL_v5_e5base_full900_avg_threshold_0p955.csv",
    "v7_global_prior": SUB / "CANDIDATE_v7_v5_global_prior_0p425.csv",
    "v7_perterm_prior": SUB / "CANDIDATE_v7_v5_perterm_prior_0p425.csv",
    "v10_prior": SUB / "CANDIDATE_v10_embedding_pu_prior.csv",
    "v10_majority": SUB / "FINAL_CANDIDATE_v10_embedding_pu_crossfit_majority.csv",
    "v10_shared_057": SUB / "FINAL_CANDIDATE_v10_embedding_pu_shared_cv_threshold_0p570.csv",
    "v11_safe_overlap": SUB / "FINAL_CANDIDATE_v11_safe_overlap_cv_ratio.csv",
    "v12_fuzzy": SUB / "FINAL_CANDIDATE_v12_fuzzy_cv_ratio.csv",
}

SCORE_FILES = {
    "v5_full900": (PROC / "v5_e5base_full900_test_proba.parquet", "proba_avg"),
    "v10_embedding_pu": (PROC / "embedding_pu_crossfit_v10_test_proba.parquet", "proba"),
    "v11_safe_overlap": (PROC / "v11_safe_overlap_crossfit_test_proba.parquet", "proba"),
    "v12_fuzzy": (PROC / "v12_fuzzy_crossfit_test_proba.parquet", "proba"),
}


def clean_label(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"0", "0.0"}:
        return 0
    if s in {"1", "1.0"}:
        return 1
    return np.nan


def safe_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "agreement_accuracy": float(accuracy_score(y, pred)),
        "macro_f1_vs_human": float(f1_score(y, pred, average="macro", labels=[0, 1])),
        "positive_f1_vs_human": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "negative_f1_vs_human": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "pred_positive_ratio": float(np.mean(pred)),
        "tn_human0_pred0": int(tn),
        "fp_human0_pred1": int(fp),
        "fn_human1_pred0": int(fn),
        "tp_human1_pred1": int(tp),
    }


def main():
    df = pd.read_csv(MANUAL)
    df["id"] = df["id"].astype(str)
    df["term_id"] = df["term_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["human_label_clean"] = df["human_label"].map(clean_label)

    df = df[df["human_label_clean"].isin([0, 1])].copy()
    df["human_label_clean"] = df["human_label_clean"].astype(int)

    print("Labeled rows:", len(df))
    print("Human label distribution:")
    print(df["human_label_clean"].value_counts())
    print("Human positive ratio:", df["human_label_clean"].mean())

    # Merge binary submissions
    pred_cols = []
    for name, path in BINARY_CANDIDATES.items():
        if not path.exists():
            print("Missing binary candidate:", name, path)
            continue
        pred = pd.read_csv(path, usecols=["id", "prediction"])
        pred["id"] = pred["id"].astype(str)
        col = f"pred__{name}"
        pred = pred.rename(columns={"prediction": col})
        df = df.merge(pred, on="id", how="left", validate="many_to_one")
        df[col] = df[col].fillna(0).astype(int)
        pred_cols.append(col)

    # Merge scores directly from parquet
    score_cols = []
    for name, (path, colname) in SCORE_FILES.items():
        if not path.exists():
            print("Missing score file:", name, path)
            continue
        s = pd.read_parquet(path, columns=["id", colname])
        s["id"] = s["id"].astype(str)
        col = f"score__{name}"
        s = s.rename(columns={colname: col})
        df = df.merge(s, on="id", how="left", validate="many_to_one")
        score_cols.append(col)

    # Known positive check from training, usually zero because train/test pair overlap is expected to be none
    train_pos_count = 0
    if (RAW / "training_pairs.csv").exists():
        train = pd.read_csv(RAW / "training_pairs.csv", usecols=["term_id", "item_id", "label"])
        train = train[train["label"].eq(1)].copy()
        train["term_id"] = train["term_id"].astype(str)
        train["item_id"] = train["item_id"].astype(str)
        train["pair_key"] = train["term_id"] + "||" + train["item_id"]
        pos_set = set(train["pair_key"])
        df["pair_key"] = df["term_id"] + "||" + df["item_id"]
        df["known_train_positive"] = df["pair_key"].isin(pos_set).astype(int)
        train_pos_count = int(df["known_train_positive"].sum())
    else:
        df["known_train_positive"] = 0

    # Candidate agreement metrics
    y = df["human_label_clean"].to_numpy(dtype=np.int8)
    metric_rows = []

    for col in pred_cols:
        pred = df[col].to_numpy(dtype=np.int8)
        row = {"candidate": col.replace("pred__", ""), "type": "binary_submission"}
        row.update(safe_metrics(y, pred))
        metric_rows.append(row)

    # Majority / strong consensus
    if pred_cols:
        votes = df[pred_cols].sum(axis=1)
        n_votes = len(pred_cols)
        df["model_vote_count"] = votes
        df["model_vote_ratio"] = votes / n_votes
        df["consensus_majority"] = (votes >= (n_votes / 2)).astype(int)
        df["consensus_strong"] = np.where(
            df["model_vote_ratio"] >= 0.75, 1,
            np.where(df["model_vote_ratio"] <= 0.25, 0, -1)
        )

        for col in ["consensus_majority"]:
            pred = df[col].to_numpy(dtype=np.int8)
            row = {"candidate": col, "type": "consensus"}
            row.update(safe_metrics(y, pred))
            metric_rows.append(row)

        strong = df[df["consensus_strong"].isin([0, 1])].copy()
        if len(strong) > 0:
            pred = strong["consensus_strong"].to_numpy(dtype=np.int8)
            yy = strong["human_label_clean"].to_numpy(dtype=np.int8)
            row = {"candidate": "consensus_strong_only", "type": "consensus"}
            row.update(safe_metrics(yy, pred))
            row["covered_rows"] = int(len(strong))
            row["coverage_ratio"] = float(len(strong) / len(df))
            metric_rows.append(row)

    # Score percentile flags
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_pct_in_manual"] = df[col].rank(pct=True)

    # Suspicion rules
    reasons = []

    for _, r in df.iterrows():
        row_reasons = []

        h = int(r["human_label_clean"])

        if int(r.get("known_train_positive", 0)) == 1 and h == 0:
            row_reasons.append("KNOWN_TRAIN_POSITIVE_BUT_HUMAN_0")

        if pred_cols:
            vote_ratio = float(r["model_vote_ratio"])
            if h == 0 and vote_ratio >= 0.75:
                row_reasons.append("HUMAN_0_BUT_STRONG_MODEL_CONSENSUS_1")
            if h == 1 and vote_ratio <= 0.25:
                row_reasons.append("HUMAN_1_BUT_STRONG_MODEL_CONSENSUS_0")

            # old V5 had the cleanest precision in your manual report, so disagreement deserves recheck
            old_col = "pred__old_v5_failed_public_0p58"
            if old_col in df.columns:
                old = int(r[old_col])
                if h == 0 and old == 1:
                    row_reasons.append("HUMAN_0_BUT_OLD_V5_PRED_1_RECHECK")
                if h == 1 and old == 0:
                    row_reasons.append("HUMAN_1_BUT_OLD_V5_PRED_0_POSSIBLE_MISSED_POSITIVE_OR_MODEL_FN")

        for col in score_cols:
            pct_col = f"{col}_pct_in_manual"
            if pct_col in df.columns and not pd.isna(r[pct_col]):
                pct = float(r[pct_col])
                if h == 0 and pct >= 0.95:
                    row_reasons.append(f"HUMAN_0_BUT_{col}_TOP5PCT_SCORE")
                if h == 1 and pct <= 0.05:
                    row_reasons.append(f"HUMAN_1_BUT_{col}_BOTTOM5PCT_SCORE")

        reasons.append("; ".join(row_reasons))

    df["suspicion_reasons"] = reasons
    df["suspicion_count"] = df["suspicion_reasons"].apply(lambda x: 0 if not x else len(x.split("; ")))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["agreement_accuracy", "macro_f1_vs_human"], ascending=False
    )

    suspicious = df[df["suspicion_count"] > 0].copy()
    suspicious = suspicious.sort_values(
        ["suspicion_count", "model_vote_ratio"], ascending=[False, False]
    )

    keep_cols = [
        "suspicion_count",
        "suspicion_reasons",
        "human_label_clean",
        "model_vote_count",
        "model_vote_ratio",
        "bucket",
        "id",
        "query",
        "title",
        "category",
        "brand",
        "gender",
        "age_group",
    ]
    keep_cols += pred_cols
    keep_cols += score_cols
    keep_cols = [c for c in keep_cols if c in suspicious.columns]

    metrics_path = OUT_DIR / "human_label_consistency_metrics.csv"
    suspicious_path = OUT_DIR / "human_label_suspicious_recheck.csv"
    full_path = OUT_DIR / "human_label_audit_full.csv"
    report_path = OUT_DIR / "human_label_consistency_report.json"

    metrics.to_csv(metrics_path, index=False)
    suspicious[keep_cols].to_csv(suspicious_path, index=False, encoding="utf-8-sig")
    df.to_csv(full_path, index=False, encoding="utf-8-sig")

    report = {
        "labeled_rows": int(len(df)),
        "human_positive_ratio": float(df["human_label_clean"].mean()),
        "known_train_positive_overlap_rows": train_pos_count,
        "available_binary_candidates": [c.replace("pred__", "") for c in pred_cols],
        "available_score_columns": [c.replace("score__", "") for c in score_cols],
        "top_agreement_candidates": metrics.head(10).to_dict(orient="records"),
        "suspicious_rows": int(len(suspicious)),
        "suspicious_ratio": float(len(suspicious) / len(df)) if len(df) else 0.0,
        "files": {
            "metrics": str(metrics_path),
            "suspicious_recheck": str(suspicious_path),
            "full_audit": str(full_path),
        },
        "important_note": (
            "This does not measure true human accuracy because hidden Kaggle labels are unavailable. "
            "It only measures consistency against model predictions, score extremes, and any known train-positive overlaps."
        ),
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("CONSISTENCY METRICS")
    print(metrics.head(20).to_string(index=False))
    print("=" * 100)
    print("SUSPICIOUS ROWS:", len(suspicious))
    print("Saved:", metrics_path)
    print("Saved:", suspicious_path)
    print("Saved:", full_path)
    print("Saved:", report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
