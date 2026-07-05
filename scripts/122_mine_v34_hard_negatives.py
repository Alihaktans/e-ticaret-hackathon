from pathlib import Path
import json
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss


ROOT = Path(".")

TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
TRAIN = ROOT / "data/raw/training_pairs.csv"

V33_MODEL = ROOT / "models/v33_e5_ft/best"
ITEM_IDS_NPY = ROOT / "data/processed/v33_ft/item_ids.npy"
ITEM_EMB_NPY = ROOT / "data/processed/v33_ft/item_emb_fp16.npy"

OUT_DIR = ROOT / "data/processed/v34"
OUT_PAIRS = OUT_DIR / "v34_hard_negative_pairs.parquet"
OUT_SUMMARY = OUT_DIR / "v34_hard_negative_summary.json"


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


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


@torch.no_grad()
def encode_queries(model, tokenizer, texts, batch_size=256, max_len=48, device="cuda"):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode train queries"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            y = model(**enc)
            emb = mean_pool(y.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        out.append(emb.detach().cpu().float().numpy())

    return np.vstack(out).astype("float32")


def main():
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading train/terms/items...")
    train = pd.read_csv(TRAIN)
    terms = pd.read_csv(TERMS)
    items = pd.read_csv(ITEMS, low_memory=False)

    train["term_id"] = train["term_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)
    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if c not in items.columns:
            items[c] = ""

    print("positive pairs:", len(train))
    print("unique train terms:", train["term_id"].nunique())
    print("unique positive items:", train["item_id"].nunique())

    # Pozitif setleri.
    pos_by_term = train.groupby("term_id")["item_id"].apply(set).to_dict()

    train_terms = (
        pd.DataFrame({"term_id": sorted(train["term_id"].unique())})
        .merge(terms[["term_id", "query"]], on="term_id", how="left", validate="one_to_one")
    )
    train_terms["query_text"] = train_terms["query"].fillna("").map(build_query_text)

    print("loading V33 item embeddings...")
    item_ids = np.load(ITEM_IDS_NPY, allow_pickle=True).astype(str)
    item_emb = np.load(ITEM_EMB_NPY, mmap_mode="r")

    print("item_ids:", item_ids.shape)
    print("item_emb:", item_emb.shape, item_emb.dtype)

    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)

    print("building FAISS index...")
    chunk = 50000
    for s in tqdm(range(0, len(item_ids), chunk), desc="faiss add"):
        e = min(s + chunk, len(item_ids))
        index.add(np.asarray(item_emb[s:e], dtype="float32"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(V33_MODEL)
    model = AutoModel.from_pretrained(V33_MODEL).to(device)
    model.eval()

    q_emb = encode_queries(
        model=model,
        tokenizer=tokenizer,
        texts=train_terms["query_text"].tolist(),
        batch_size=256,
        max_len=48,
        device=device,
    )

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print("searching hard negatives...")
    topk = 220
    D, I = index.search(q_emb, topk)

    rows = []

    # Pozitifleri ekle.
    print("adding positives...")
    pos = train[["term_id", "item_id"]].drop_duplicates().copy()
    pos["label"] = 1
    pos["source"] = "train_positive"
    pos["v33_retrieval_score"] = np.nan
    rows.append(pos)

    # Hard negative ekle.
    print("adding hard negatives...")
    neg_rows = []
    max_neg_per_term = 18

    for qi, term_id in tqdm(enumerate(train_terms["term_id"].tolist()), total=len(train_terms), desc="negatives"):
        positives = pos_by_term.get(term_id, set())
        taken = 0

        for rank, idx in enumerate(I[qi]):
            item_id = str(item_ids[idx])
            score = float(D[qi, rank])

            if item_id in positives:
                continue

            neg_rows.append({
                "term_id": term_id,
                "item_id": item_id,
                "label": 0,
                "source": f"v33_hard_negative_rank_{rank+1}",
                "v33_retrieval_score": score,
            })

            taken += 1
            if taken >= max_neg_per_term:
                break

    neg = pd.DataFrame(neg_rows)
    rows.append(neg)

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(["term_id", "item_id"], keep="first").reset_index(drop=True)

    print("merging text...")
    terms_small = terms[["term_id", "query"]].copy()
    terms_small["query_text"] = terms_small["query"].map(build_query_text)

    item_small = items[["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]].copy()
    item_small["item_text"] = item_small.apply(build_item_text, axis=1)

    out = out.merge(
        terms_small[["term_id", "query", "query_text"]],
        on="term_id",
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        item_small[["item_id", "title", "category", "brand", "gender", "age_group", "item_text"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    out["query_text"] = out["query_text"].fillna("query: ")
    out["item_text"] = out["item_text"].fillna("passage: ")

    out.to_parquet(OUT_PAIRS, index=False)

    summary = {
        "rows": int(len(out)),
        "positives": int((out["label"] == 1).sum()),
        "negatives": int((out["label"] == 0).sum()),
        "positive_rate": float(out["label"].mean()),
        "unique_terms": int(out["term_id"].nunique()),
        "unique_items": int(out["item_id"].nunique()),
        "elapsed_min": float((time.time() - start) / 60),
        "output": str(OUT_PAIRS),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", OUT_PAIRS)


if __name__ == "__main__":
    main()
