from pathlib import Path
import argparse
import json
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


ROOT = Path(".")

TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"

OUT_DIR = ROOT / "data/processed/v33_ft"
OUT_SCORE = ROOT / "data/processed/v33_ft_pair_scores.parquet"


def clean_text(x, max_chars=700):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = " ".join(x.split())
    if len(x) > max_chars:
        x = x[:max_chars]
    return x


def build_item_text(row):
    parts = []

    title = clean_text(row.get("title", ""), 260)
    category = clean_text(row.get("category", ""), 180)
    brand = clean_text(row.get("brand", ""), 80)
    gender = clean_text(row.get("gender", ""), 60)
    age_group = clean_text(row.get("age_group", ""), 60)
    attrs = clean_text(row.get("attributes", ""), 450)

    if title:
        parts.append(f"title: {title}")
    if category:
        parts.append(f"category: {category}")
    if brand:
        parts.append(f"brand: {brand}")
    if gender:
        parts.append(f"gender: {gender}")
    if age_group:
        parts.append(f"age_group: {age_group}")
    if attrs:
        parts.append(f"attributes: {attrs}")

    return "passage: " + " | ".join(parts)


def build_query_text(q):
    return "query: " + clean_text(q, 180)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def encode_texts(model, tokenizer, texts, max_len, batch_size, device, desc):
    embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        embs.append(emb.detach().cpu().to(torch.float16).numpy())

    return np.vstack(embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/v33_e5_ft/best")
    parser.add_argument("--query-batch", type=int, default=256)
    parser.add_argument("--item-batch", type=int, default=128)
    parser.add_argument("--score-batch", type=int, default=50000)
    parser.add_argument("--max-query-len", type=int, default=48)
    parser.add_argument("--max-item-len", type=int, default=160)
    args = parser.parse_args()

    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    print("loading model:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    print("loading submission pairs...")
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    unique_terms = pd.Series(pairs["term_id"].unique(), name="term_id").astype(str)
    unique_items = pd.Series(pairs["item_id"].unique(), name="item_id").astype(str)

    print("pairs:", len(pairs))
    print("unique terms:", len(unique_terms))
    print("unique items:", len(unique_items))

    print("loading terms/items...")
    terms = pd.read_csv(TERMS)
    items = pd.read_csv(ITEMS, low_memory=False)

    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if c not in items.columns:
            items[c] = ""

    terms["query_text"] = terms["query"].map(build_query_text)
    items["item_text"] = items.apply(build_item_text, axis=1)

    term_text = (
        pd.DataFrame({"term_id": unique_terms})
        .merge(terms[["term_id", "query_text"]], on="term_id", how="left", validate="one_to_one")
    )
    item_text = (
        pd.DataFrame({"item_id": unique_items})
        .merge(items[["item_id", "item_text"]], on="item_id", how="left", validate="one_to_one")
    )

    term_text["query_text"] = term_text["query_text"].fillna("query: ")
    item_text["item_text"] = item_text["item_text"].fillna("passage: ")

    term_ids = term_text["term_id"].to_numpy()
    item_ids = item_text["item_id"].to_numpy()

    print("encoding terms...")
    term_emb = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=term_text["query_text"].tolist(),
        max_len=args.max_query_len,
        batch_size=args.query_batch,
        device=device,
        desc="terms",
    )

    print("encoding items...")
    item_emb = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=item_text["item_text"].tolist(),
        max_len=args.max_item_len,
        batch_size=args.item_batch,
        device=device,
        desc="items",
    )

    print("term_emb:", term_emb.shape, term_emb.dtype)
    print("item_emb:", item_emb.shape, item_emb.dtype)

    np.save(OUT_DIR / "term_ids.npy", term_ids)
    np.save(OUT_DIR / "item_ids.npy", item_ids)
    np.save(OUT_DIR / "term_emb_fp16.npy", term_emb)
    np.save(OUT_DIR / "item_emb_fp16.npy", item_emb)

    del terms, items, term_text, item_text
    gc.collect()

    term_map = pd.Series(np.arange(len(term_ids), dtype=np.int32), index=term_ids)
    item_map = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)

    print("mapping pair indices...")
    t_idx = pairs["term_id"].map(term_map).astype(np.int32).to_numpy()
    i_idx = pairs["item_id"].map(item_map).astype(np.int32).to_numpy()

    print("scoring pairs...")
    scores = np.empty(len(pairs), dtype=np.float32)

    for s in tqdm(range(0, len(pairs), args.score_batch), desc="scoring"):
        e = min(s + args.score_batch, len(pairs))

        qe = term_emb[t_idx[s:e]].astype(np.float32)
        ie = item_emb[i_idx[s:e]].astype(np.float32)

        scores[s:e] = np.einsum("ij,ij->i", qe, ie)

    print("building score frame...")
    out = pairs.copy()
    out["v33_ft_score"] = scores

    print("ranking within term...")
    out["v33_ft_rank"] = (
        out.groupby("term_id")["v33_ft_score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )

    out["candidate_count"] = out.groupby("term_id")["id"].transform("count").astype(np.int32)
    out["v33_ft_pct_rank"] = (
        (out["v33_ft_rank"] - 1) / (out["candidate_count"] - 1).clip(lower=1)
    ).astype(np.float32)

    out["v33_ft_term_z"] = (
        out["v33_ft_score"] - out.groupby("term_id")["v33_ft_score"].transform("mean")
    ) / (out.groupby("term_id")["v33_ft_score"].transform("std").replace(0, np.nan))

    out["v33_ft_term_z"] = out["v33_ft_term_z"].fillna(0).astype(np.float32)

    keep = [
        "id",
        "term_id",
        "item_id",
        "v33_ft_score",
        "v33_ft_rank",
        "v33_ft_pct_rank",
        "v33_ft_term_z",
        "candidate_count",
    ]

    out[keep].to_parquet(OUT_SCORE, index=False)

    summary = {
        "model_dir": str(model_dir),
        "pairs": int(len(pairs)),
        "unique_terms": int(len(term_ids)),
        "unique_items": int(len(item_ids)),
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "elapsed_min": float((time.time() - start) / 60),
        "output_score": str(OUT_SCORE),
    }

    with open(OUT_DIR / "score_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", OUT_SCORE)


if __name__ == "__main__":
    main()
