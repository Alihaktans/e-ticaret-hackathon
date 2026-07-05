"""Score only the v22<->v71 decision boundary with two source-holdout BERTurks.

This keeps inference tractable on an 8GB GPU and, importantly, asks the
pairwise models only about rows where two strong systems already disagree.
Outputs are resumable at model granularity.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(".")
BASE = ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv"
V71 = ROOT / "submissions/FINAL_CANDIDATE_v71_trendlex_query_v22_budget_full.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
OUT = ROOT / "data/processed/v75_boundary_pairbert_scores.parquet"
MODELS = {
    "v29hold": ROOT / "models/v62_berturk_pairwise_holdout_v29/best",
    "v33hold": ROOT / "models/v62_berturk_pairwise_holdout_v33/best",
}


def clean(x, limit: int) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split())[:limit]


def item_text(r) -> str:
    parts = []
    for col, label, limit in [
        ("title", "başlık", 260), ("category", "kategori", 180),
        ("brand", "marka", 80), ("gender", "cinsiyet", 50),
        ("age_group", "yaş", 50), ("attributes", "özellik", 350),
    ]:
        value = clean(r.get(col, ""), limit)
        if value:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)


@torch.inference_mode()
def score_model(frame: pd.DataFrame, model_path: Path, batch_size: int, max_len: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    if device == "cuda":
        model.half()
    out = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        enc = tok(
            frame["query_text"].iloc[start:end].tolist(),
            frame["item_text"].iloc[start:end].tolist(),
            padding=True, truncation=True, max_length=max_len, return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**enc).logits.view(-1).float()
        out[start:end] = logits.cpu().numpy()
        if end % 5000 < batch_size:
            print(model_path, f"{end:,}/{len(frame):,}", flush=True)
    del model, tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def build_boundary() -> pd.DataFrame:
    base = pd.read_csv(BASE, dtype={"id": "string"})
    v71 = pd.read_csv(V71, dtype={"id": "string"})
    if not base["id"].equals(v71["id"]):
        raise RuntimeError("v22/v71 order mismatch")
    changed = base["prediction"].to_numpy(np.int8) != v71["prediction"].to_numpy(np.int8)
    ids = set(base.loc[changed, "id"].astype(str))
    parts = []
    for chunk in pd.read_csv(PAIRS, dtype={"id": "string", "term_id": "string", "item_id": "string"}, chunksize=400_000):
        keep = chunk["id"].astype(str).isin(ids)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    d = pd.concat(parts, ignore_index=True)
    if len(d) != int(changed.sum()) or d["id"].nunique() != len(d):
        raise RuntimeError("boundary extraction failed")
    terms = pd.read_csv(TERMS, usecols=["term_id", "query"], dtype={"term_id": "string"})
    needed = set(d["item_id"].astype(str))
    item_parts = []
    for chunk in pd.read_csv(ITEMS, low_memory=False, chunksize=100_000):
        chunk["item_id"] = chunk["item_id"].astype(str)
        keep = chunk["item_id"].isin(needed)
        if keep.any():
            item_parts.append(chunk.loc[keep].copy())
    items = pd.concat(item_parts, ignore_index=True).drop_duplicates("item_id")
    items["item_text"] = items.apply(item_text, axis=1)
    d = d.merge(terms, on="term_id", how="left", validate="many_to_one")
    d = d.merge(items[["item_id", "item_text"]], on="item_id", how="left", validate="many_to_one")
    d["query_text"] = "sorgu: " + d["query"].fillna("").astype(str).str.slice(0, 180)
    if d[["query_text", "item_text"]].isna().any().any():
        raise RuntimeError("missing boundary text")
    return d[["id", "term_id", "item_id", "query_text", "item_text"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=160)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if OUT.exists():
        d = pd.read_parquet(OUT)
        print("resuming", OUT, d.shape, flush=True)
    else:
        d = build_boundary()
        d.to_parquet(OUT, index=False)
        print("saved boundary", OUT, d.shape, flush=True)

    for name, model_path in MODELS.items():
        col = f"v75_{name}_score"
        if col in d and d[col].notna().all():
            print("already complete", col, flush=True)
            continue
        d[col] = score_model(d, model_path, args.batch_size, args.max_len)
        d.to_parquet(OUT, index=False)
        print("saved", col, OUT, flush=True)

    print(d.filter(like="score").describe().to_string())


if __name__ == "__main__":
    main()
