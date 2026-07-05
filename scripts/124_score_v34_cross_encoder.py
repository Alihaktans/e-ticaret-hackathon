from pathlib import Path
import argparse
import json
import time
import gc

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(".")

TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"

OUT_DIR = ROOT / "data/processed/v34"
TEMP_SCORE = OUT_DIR / "v34_cross_encoder_pair_scores_raw.parquet"
OUT_SCORE = OUT_DIR / "v34_cross_encoder_pair_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v34_cross_encoder_score_summary.json"


def clean_text(x, max_chars=700):
    if pd.isna(x):
        return ""
    x = str(x).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = " ".join(x.split())
    return x[:max_chars]


def build_query_text(q):
    return "query: " + clean_text(q, 180)


def build_item_text(row):
    parts = []

    for c, lim in [
        ("title", 260),
        ("category", 180),
        ("brand", 80),
        ("gender", 60),
        ("age_group", 60),
        ("attributes", 450),
    ]:
        v = clean_text(row.get(c, ""), lim)
        if v:
            parts.append(f"{c}: {v}")

    return "passage: " + " | ".join(parts)


@torch.no_grad()
def score_pairs(model, tokenizer, queries, items, batch_size, max_len, device):
    scores = np.empty(len(queries), dtype=np.float32)

    for s in range(0, len(queries), batch_size):
        e = min(s + batch_size, len(queries))

        enc = tokenizer(
            queries[s:e],
            items[s:e],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(**enc).logits.squeeze(-1)

        prob = torch.sigmoid(logits).detach().cpu().float().numpy()
        scores[s:e] = prob.astype(np.float32)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/v34_cross_encoder/best")
    parser.add_argument("--chunk-size", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=192)
    args = parser.parse_args()

    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if TEMP_SCORE.exists():
        TEMP_SCORE.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("loading model:", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    print("loading term/item text maps...")
    terms = pd.read_csv(TERMS)
    items = pd.read_csv(ITEMS, low_memory=False)

    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if c not in items.columns:
            items[c] = ""

    terms["query_text"] = terms["query"].map(build_query_text)
    items["item_text"] = items.apply(build_item_text, axis=1)

    query_map = terms.set_index("term_id")["query_text"]
    item_map = items.set_index("item_id")["item_text"]

    del terms, items
    gc.collect()

    writer = None
    total_rows = 0
    score_min = 999.0
    score_max = -999.0
    score_sum = 0.0

    print("scoring submission pairs...")

    reader = pd.read_csv(
        PAIRS,
        usecols=["id", "term_id", "item_id"],
        chunksize=args.chunk_size,
    )

    for chunk_idx, chunk in enumerate(reader, 1):
        chunk["id"] = chunk["id"].astype(str)
        chunk["term_id"] = chunk["term_id"].astype(str)
        chunk["item_id"] = chunk["item_id"].astype(str)

        q_texts = chunk["term_id"].map(query_map).fillna("query: ").astype(str).tolist()
        i_texts = chunk["item_id"].map(item_map).fillna("passage: ").astype(str).tolist()

        scores = score_pairs(
            model=model,
            tokenizer=tokenizer,
            queries=q_texts,
            items=i_texts,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=device,
        )

        out = chunk[["id", "term_id", "item_id"]].copy()
        out["v34_ce_score"] = scores

        table = pa.Table.from_pandas(out, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(TEMP_SCORE, table.schema, compression="zstd")

        writer.write_table(table)

        total_rows += len(out)
        score_min = min(score_min, float(scores.min()))
        score_max = max(score_max, float(scores.max()))
        score_sum += float(scores.sum())

        print(
            f"chunk={chunk_idx} rows_done={total_rows} "
            f"chunk_score_mean={scores.mean():.6f} "
            f"elapsed_min={(time.time() - start) / 60:.1f}"
        )

    if writer is not None:
        writer.close()

    print("raw score saved:", TEMP_SCORE)

    del model
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    print("post-processing ranks...")
    df = pd.read_parquet(TEMP_SCORE)

    df["v34_ce_rank"] = (
        df.groupby("term_id")["v34_ce_score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )

    df["candidate_count"] = (
        df.groupby("term_id")["id"]
        .transform("count")
        .astype(np.int32)
    )

    df["v34_ce_pct_rank"] = (
        (df["v34_ce_rank"] - 1)
        / (df["candidate_count"] - 1).clip(lower=1)
    ).astype(np.float32)

    term_mean = df.groupby("term_id")["v34_ce_score"].transform("mean")
    term_std = df.groupby("term_id")["v34_ce_score"].transform("std").replace(0, np.nan)

    df["v34_ce_term_z"] = (
        (df["v34_ce_score"] - term_mean) / term_std
    ).fillna(0).astype(np.float32)

    keep = [
        "id",
        "term_id",
        "item_id",
        "v34_ce_score",
        "v34_ce_rank",
        "v34_ce_pct_rank",
        "v34_ce_term_z",
        "candidate_count",
    ]

    df[keep].to_parquet(OUT_SCORE, index=False)

    summary = {
        "model_dir": args.model_dir,
        "rows": int(len(df)),
        "score_min": float(df["v34_ce_score"].min()),
        "score_mean": float(df["v34_ce_score"].mean()),
        "score_max": float(df["v34_ce_score"].max()),
        "elapsed_min": float((time.time() - start) / 60),
        "output": str(OUT_SCORE),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", OUT_SCORE)


if __name__ == "__main__":
    main()
