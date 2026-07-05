from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(".")
CAND = ROOT / "data/processed/v13_cross_encoder_candidate_pairs.parquet"
OUT = ROOT / "data/processed/v15_bge_reranker_candidate_scores.parquet"
PARTIAL = ROOT / "data/processed/v15_bge_reranker_candidate_scores_partial.parquet"
REPORT = ROOT / "reports/manual_review/v15_bge_reranker_candidate_score_report.csv"

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 8
CHUNK_SIZE = 5000
MAX_LENGTH = 512

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

@torch.no_grad()
def score_batch(model, tokenizer, queries, products, device):
    inputs = tokenizer(
        queries,
        products,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    logits = model(**inputs).logits.view(-1).float()
    return logits.detach().cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("model:", MODEL_NAME)

    cand = pd.read_parquet(
        CAND,
        columns=["id", "proba_avg", "cross_text_query", "cross_text_item"]
    )
    cand["id"] = cand["id"].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    parts = []
    n = len(cand)
    t0 = time.time()

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        part = cand.iloc[start:end].copy()

        queries = part["cross_text_query"].fillna("").astype(str).tolist()
        products = part["cross_text_item"].fillna("").astype(str).tolist()

        raw_parts = []
        for b in range(0, len(part), BATCH_SIZE):
            be = min(b + BATCH_SIZE, len(part))
            raw_parts.append(
                score_batch(
                    model,
                    tokenizer,
                    queries[b:be],
                    products[b:be],
                    device,
                )
            )

        raw = np.concatenate(raw_parts)

        out_part = pd.DataFrame({
            "id": part["id"].to_numpy(),
            "proba_avg": part["proba_avg"].to_numpy(),
            "bge_raw": raw.astype(float),
        })
        out_part["bge_sigmoid"] = sigmoid(out_part["bge_raw"].to_numpy())

        # Validation'da iyi çıkan iki blend'i şimdiden kaydet
        out_part["bge_blend_w020"] = 0.20 * out_part["proba_avg"] + 0.80 * out_part["bge_sigmoid"]
        out_part["bge_blend_w030"] = 0.30 * out_part["proba_avg"] + 0.70 * out_part["bge_sigmoid"]

        parts.append(out_part)

        elapsed = time.time() - t0
        print(f"done {end}/{n} elapsed={elapsed/60:.1f} min")

        if len(parts) % 10 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(PARTIAL, index=False)
            print("partial saved:", PARTIAL)

    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(OUT, index=False)

    report = pd.DataFrame([{
        "rows": len(scores),
        "v5_min": float(scores["proba_avg"].min()),
        "v5_max": float(scores["proba_avg"].max()),
        "bge_sigmoid_mean": float(scores["bge_sigmoid"].mean()),
        "bge_blend_w020_mean": float(scores["bge_blend_w020"].mean()),
        "bge_blend_w030_mean": float(scores["bge_blend_w030"].mean()),
        "w020_ge_0p16": int((scores["bge_blend_w020"] >= 0.16).sum()),
        "w030_ge_0p34": int((scores["bge_blend_w030"] >= 0.34).sum()),
        "elapsed_min": (time.time() - t0) / 60,
    }])
    report.to_csv(REPORT, index=False)

    print(report.to_string(index=False))
    print("saved:", OUT)
    print("report:", REPORT)

if __name__ == "__main__":
    main()
