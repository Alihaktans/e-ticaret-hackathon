"""Score the v22/v71 boundary with the targeted v88 intent reranker."""
from pathlib import Path
import argparse
import gc

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(".")
BOUNDARY = ROOT / "data/processed/v75_boundary_pairbert_scores.parquet"
MODEL = ROOT / "models/v88_intent_berturk_from_v62v29/best"
COL = "v88_intent_score"


def repair(x):
    x = "" if pd.isna(x) else str(x)
    try:
        return x.encode("latin1").decode("utf8")
    except (UnicodeError, UnicodeEncodeError):
        return x


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--column", default=COL)
    args = ap.parse_args()
    model_path = Path(args.model); column = args.column
    d = pd.read_parquet(BOUNDARY)
    if column in d and d[column].notna().all():
        print("already complete", column); return
    q = d["query_text"].map(repair).tolist(); x = d["item_text"].map(repair).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    if device == "cuda": model.half()
    score = np.empty(len(d), dtype=np.float32)
    for start in range(0, len(d), 64):
        end = min(len(d), start + 64)
        enc = tok(q[start:end], x[start:end], padding=True, truncation=True,
                  max_length=176, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            score[start:end] = model(**enc).logits.view(-1).float().cpu().numpy()
        if end % 5000 < 64: print(end, "/", len(d), flush=True)
    d[column] = score; d.to_parquet(BOUNDARY, index=False)
    print(pd.Series(score).describe().to_string())
    del model, tok; gc.collect()


if __name__ == "__main__":
    main()
