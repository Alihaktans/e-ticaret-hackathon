from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(".")
CAND = ROOT / "data/processed/v13_cross_encoder_candidate_pairs.parquet"
OUT = ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet"
PARTIAL = ROOT / "data/processed/v13_cross_encoder_candidate_scores_partial.parquet"
REPORT = ROOT / "reports/manual_review/v13_cross_encoder_candidate_score_report.csv"

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BATCH_SIZE = 64
CHUNK_SIZE = 20000
MAX_LENGTH = 256

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def main():
    from sentence_transformers import CrossEncoder
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("model:", MODEL_NAME)

    cand = pd.read_parquet(
        CAND,
        columns=["id", "proba_avg", "cross_text_query", "cross_text_item"]
    )
    cand["id"] = cand["id"].astype(str)

    model = CrossEncoder(MODEL_NAME, device=device, max_length=MAX_LENGTH)

    parts = []
    n = len(cand)
    t0 = time.time()

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        part = cand.iloc[start:end].copy()

        pairs = list(zip(
            part["cross_text_query"].fillna("").astype(str).tolist(),
            part["cross_text_item"].fillna("").astype(str).tolist(),
        ))

        raw = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        out_part = pd.DataFrame({
            "id": part["id"].to_numpy(),
            "proba_avg": part["proba_avg"].to_numpy(),
            "cross_raw": raw.astype(float),
        })
        out_part["cross_sigmoid"] = sigmoid(out_part["cross_raw"].to_numpy())
        out_part["blend_w060"] = 0.60 * out_part["proba_avg"] + 0.40 * out_part["cross_sigmoid"]

        parts.append(out_part)

        done = end
        elapsed = time.time() - t0
        print(f"done {done}/{n} elapsed={elapsed/60:.1f} min")

        if len(parts) % 5 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(PARTIAL, index=False)
            print("partial saved:", PARTIAL)

    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(OUT, index=False)

    report = pd.DataFrame([{
        "rows": len(scores),
        "v5_min": float(scores["proba_avg"].min()),
        "v5_max": float(scores["proba_avg"].max()),
        "cross_sigmoid_mean": float(scores["cross_sigmoid"].mean()),
        "blend_w060_mean": float(scores["blend_w060"].mean()),
        "blend_w060_ge_060": int((scores["blend_w060"] >= 0.60).sum()),
        "blend_w060_ge_060_ratio_in_candidates": float((scores["blend_w060"] >= 0.60).mean()),
        "elapsed_min": (time.time() - t0) / 60,
    }])
    report.to_csv(REPORT, index=False)

    print(report.to_string(index=False))
    print("saved:", OUT)
    print("report:", REPORT)

if __name__ == "__main__":
    main()
