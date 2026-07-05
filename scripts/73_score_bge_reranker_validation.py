from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(".")
OUT_SCORE = ROOT / "data/processed/v15_bge_reranker_validation_scores.parquet"
OUT_GRID = ROOT / "reports/manual_review/v15_bge_reranker_validation_grid.csv"

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 8
MAX_LENGTH = 512

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
        + " | attributes: " + df.get("attributes", "").fillna("").astype(str).str.slice(0, 500)
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

@torch.no_grad()
def score_pairs(model, tokenizer, queries, products, device):
    scores = []
    model.eval()

    for start in range(0, len(queries), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(queries))
        batch_q = queries[start:end]
        batch_p = products[start:end]

        inputs = tokenizer(
            batch_q,
            batch_p,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        logits = model(**inputs).logits.view(-1).float()
        scores.append(logits.detach().cpu().numpy())

        if end % 256 == 0 or end == len(queries):
            print(f"scored {end}/{len(queries)}")

    return np.concatenate(scores)

def main():
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

    queries = df["query"].fillna("").astype(str).tolist()
    products = product_text(df).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    raw = score_pairs(model, tokenizer, queries, products, device)

    df["bge_raw"] = raw.astype(float)
    df["bge_sigmoid"] = sigmoid(raw)
    df.to_parquet(OUT_SCORE, index=False)

    rows = []

    for label_set, part in df.groupby("label_set"):
        y = part["assistant_label"].to_numpy()
        v5s = part["proba_avg"].to_numpy()
        bge = part["bge_sigmoid"].to_numpy()

        for th in np.round(np.arange(0.02, 0.98, 0.02), 3):
            pred = (bge >= th).astype(int)
            row = {
                "label_set": label_set,
                "method": "bge_only",
                "w_v5": 0.0,
                "threshold": float(th),
            }
            row.update(metrics(y, pred))
            rows.append(row)

        for w in [0.20, 0.30, 0.40, 0.50, 0.60]:
            blend = w * v5s + (1.0 - w) * bge
            for th in np.round(np.arange(0.02, 0.98, 0.02), 3):
                pred = (blend >= th).astype(int)
                row = {
                    "label_set": label_set,
                    "method": "blend",
                    "w_v5": w,
                    "threshold": float(th),
                }
                row.update(metrics(y, pred))
                rows.append(row)

    res = pd.DataFrame(rows)
    res = res.sort_values(["label_set", "macro_f1"], ascending=[True, False])
    res.to_csv(OUT_GRID, index=False)

    print(res.groupby("label_set").head(25).to_string(index=False))
    print("saved scores:", OUT_SCORE)
    print("saved grid:", OUT_GRID)

if __name__ == "__main__":
    main()
