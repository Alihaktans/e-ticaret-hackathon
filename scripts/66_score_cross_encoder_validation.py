from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")
OUT_SCORE = ROOT / "data/processed/v13_cross_encoder_validation_scores.parquet"
OUT_GRID = ROOT / "reports/manual_review/v13_cross_encoder_validation_grid.csv"

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BATCH_SIZE = 64
MAX_LENGTH = 256

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
}

V5_SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def product_text(df):
    return (
        "title: " + df.get("title", "").fillna("").astype(str)
        + " | category: " + df.get("category", "").fillna("").astype(str)
        + " | brand: " + df.get("brand", "").fillna("").astype(str)
        + " | gender: " + df.get("gender", "").fillna("").astype(str)
        + " | age_group: " + df.get("age_group", "").fillna("").astype(str)
        + " | attributes: " + df.get("attributes", "").fillna("").astype(str).str.slice(0, 400)
    )

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

def main():
    from sentence_transformers import CrossEncoder
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("model:", MODEL_NAME)

    v5 = pd.read_parquet(V5_SCORE, columns=["id", "proba_avg"])
    v5["id"] = v5["id"].astype(str)

    frames = []
    for name, path in LABEL_FILES.items():
        df = pd.read_csv(path)
        df["id"] = df["id"].astype(str)
        df = df[df["assistant_label"].isin([0, 1, "0", "1"])].copy()
        df["assistant_label"] = df["assistant_label"].astype(int)
        df["label_set"] = name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.merge(v5, on="id", how="left", validate="many_to_one")
    df = df.dropna(subset=["proba_avg"]).copy()

    pairs = list(zip(
        df["query"].fillna("").astype(str).tolist(),
        product_text(df).tolist()
    ))

    model = CrossEncoder(MODEL_NAME, device=device, max_length=MAX_LENGTH)

    raw_scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    df["cross_raw"] = raw_scores.astype(float)
    df["cross_sigmoid"] = sigmoid(raw_scores)
    df.to_parquet(OUT_SCORE, index=False)

    rows = []

    # Pure V5 thresholds
    for label_set, part in df.groupby("label_set"):
        y = part["assistant_label"].to_numpy()
        v5s = part["proba_avg"].to_numpy()
        ces = part["cross_sigmoid"].to_numpy()

        for th in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
            p = (v5s >= th).astype(int)
            row = {"label_set": label_set, "method": "v5_only", "v5_threshold": th, "blend_weight_v5": 1.0, "final_threshold": th}
            row.update(metrics(y, p))
            rows.append(row)

        # Pure CE thresholds
        for th in np.round(np.arange(0.05, 0.96, 0.025), 3):
            p = (ces >= th).astype(int)
            row = {"label_set": label_set, "method": "ce_only", "v5_threshold": "", "blend_weight_v5": 0.0, "final_threshold": float(th)}
            row.update(metrics(y, p))
            rows.append(row)

        # Blend V5 + CE
        for w in [0.25, 0.40, 0.50, 0.60, 0.75]:
            blend = w * v5s + (1 - w) * ces
            for th in np.round(np.arange(0.05, 0.96, 0.025), 3):
                p = (blend >= th).astype(int)
                row = {"label_set": label_set, "method": "blend", "v5_threshold": "", "blend_weight_v5": w, "final_threshold": float(th)}
                row.update(metrics(y, p))
                rows.append(row)

    res = pd.DataFrame(rows)
    res = res.sort_values(["label_set", "macro_f1"], ascending=[True, False])
    res.to_csv(OUT_GRID, index=False)

    print("\nTOP RESULTS")
    print(res.groupby("label_set").head(20).to_string(index=False))
    print("\nsaved scores:", OUT_SCORE)
    print("saved grid:", OUT_GRID)

if __name__ == "__main__":
    main()
