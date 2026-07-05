"""Encode and score every competition candidate with Trendyol's e-commerce model.

The artifacts are resumable.  ``v70_trendyol_pair_scores.npy`` is aligned exactly
with ``data/raw/submission_pairs.csv`` and can be joined without string keys.
Run with the dedicated compatibility interpreter created for this model:
    .venv_trendyol/Scripts/python scripts/175_score_full_trendyol_embedding.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(".")
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/v70_trendyol_embedding"
MODEL_ID = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
DIM = 768


def clean(series: pd.Series, limit: int) -> pd.Series:
    return (
        series.fillna("").astype(str).str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, limit)
    )


def product_text(frame: pd.DataFrame) -> list[str]:
    title = clean(frame["title"], 300)
    category = clean(frame["category"], 220)
    brand = clean(frame["brand"], 100)
    return ("Başlık: " + title + " | Kategori: " + category + " | Marka: " + brand).tolist()


def build_indices() -> tuple[np.ndarray, np.ndarray, int]:
    item_set: set[str] = set()
    term_set: set[str] = set()
    n_pairs = 0
    for chunk in pd.read_csv(RAW / "submission_pairs.csv", usecols=["term_id", "item_id"], chunksize=500_000):
        item_set.update(chunk["item_id"].astype(str).unique())
        term_set.update(chunk["term_id"].astype(str).unique())
        n_pairs += len(chunk)
        print(f"index: {n_pairs:,} pairs", flush=True)
    item_ids = np.asarray(sorted(item_set), dtype="U17")
    term_ids = np.asarray(sorted(term_set), dtype="U13")
    np.save(OUT / "item_ids.npy", item_ids)
    np.save(OUT / "term_ids.npy", term_ids)
    (OUT / "index.json").write_text(
        json.dumps({"pairs": n_pairs, "items": len(item_ids), "terms": len(term_ids)}, indent=2), encoding="utf8"
    )
    return item_ids, term_ids, n_pairs


def load_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True, truncate_dim=DIM, device="cuda")
    model.half()
    return model


def encode_queries(model: SentenceTransformer, term_ids: np.ndarray) -> None:
    out_path = OUT / "query_embeddings.f16.dat"
    if out_path.exists() and out_path.stat().st_size == len(term_ids) * DIM * 2:
        print("query embeddings already complete", flush=True)
        return
    terms = pd.read_csv(RAW / "terms.csv", usecols=["term_id", "query"]).drop_duplicates("term_id").set_index("term_id")
    text = terms.reindex(term_ids)["query"]
    if text.isna().any():
        raise RuntimeError(f"missing text for {int(text.isna().sum())} test terms")
    emb = model.encode(text.astype(str).tolist(), batch_size=512, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(len(term_ids), DIM))
    mm[:] = emb.astype(np.float16)
    mm.flush()
    print(f"wrote {out_path}", flush=True)


def encode_items(model: SentenceTransformer, item_ids: np.ndarray) -> None:
    out_path = OUT / "item_embeddings.f16.dat"
    done_path = OUT / "item_done.npy"
    expected = len(item_ids) * DIM * 2
    if out_path.exists() and out_path.stat().st_size != expected:
        raise RuntimeError(f"unexpected item embedding size: {out_path.stat().st_size} != {expected}")
    mm = np.memmap(out_path, dtype=np.float16, mode="r+" if out_path.exists() else "w+", shape=(len(item_ids), DIM))
    done = np.load(done_path) if done_path.exists() else np.zeros(len(item_ids), dtype=bool)
    item_index = pd.Index(item_ids)
    seen = 0
    started = time.time()
    cols = ["item_id", "title", "category", "brand"]
    for chunk in pd.read_csv(RAW / "items.csv", usecols=cols, chunksize=10_000, low_memory=False):
        pos = item_index.get_indexer(chunk["item_id"].astype(str))
        keep = pos >= 0
        if not keep.any():
            continue
        pos = pos[keep]
        frame = chunk.loc[keep]
        todo = ~done[pos]
        if todo.any():
            pos = pos[todo]
            frame = frame.loc[todo]
            emb = model.encode(product_text(frame), batch_size=256, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            mm[pos] = emb.astype(np.float16)
            done[pos] = True
        seen += int(keep.sum())
        if seen % 100_000 < 10_000:
            mm.flush(); np.save(done_path, done)
            rate = max(done.sum(), 1) / max(time.time() - started, 1e-6)
            print(f"items: {done.sum():,}/{len(done):,} complete ({rate:,.0f}/s)", flush=True)
    mm.flush(); np.save(done_path, done)
    if not done.all():
        raise RuntimeError(f"missing embeddings for {int((~done).sum())} candidate items")
    print("all item embeddings complete", flush=True)


def score_pairs(item_ids: np.ndarray, term_ids: np.ndarray, n_pairs: int) -> None:
    score_path = OUT / "v70_trendyol_pair_scores.npy"
    scores = np.lib.format.open_memmap(score_path, dtype=np.float32, mode="w+", shape=(n_pairs,))
    item_emb = np.memmap(OUT / "item_embeddings.f16.dat", dtype=np.float16, mode="r", shape=(len(item_ids), DIM))
    query_emb = np.memmap(OUT / "query_embeddings.f16.dat", dtype=np.float16, mode="r", shape=(len(term_ids), DIM))
    item_index, term_index = pd.Index(item_ids), pd.Index(term_ids)
    offset = 0
    for chunk in pd.read_csv(RAW / "submission_pairs.csv", usecols=["term_id", "item_id"], chunksize=25_000):
        ii = item_index.get_indexer(chunk["item_id"].astype(str))
        qi = term_index.get_indexer(chunk["term_id"].astype(str))
        if (ii < 0).any() or (qi < 0).any():
            raise RuntimeError("candidate id missing from embedding index")
        a = np.asarray(query_emb[qi], dtype=np.float32)
        b = np.asarray(item_emb[ii], dtype=np.float32)
        scores[offset:offset + len(chunk)] = np.einsum("ij,ij->i", a, b, optimize=True)
        offset += len(chunk)
        if offset % 250_000 < 25_000:
            scores.flush(); print(f"scores: {offset:,}/{n_pairs:,}", flush=True)
    scores.flush()
    if offset != n_pairs:
        raise RuntimeError(f"pair count changed: {offset} != {n_pairs}")
    meta = {
        "model": MODEL_ID, "revision": "00c030c9", "dimension": DIM,
        "query_format": "raw", "product_format": "Başlık + Kategori + Marka",
        "pair_order": "data/raw/submission_pairs.csv", "pairs": n_pairs,
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "score_mean": float(scores.mean()), "warning": "Scores are independent similarities, not calibrated probabilities."
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "index", "encode", "score"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "index"} or not (OUT / "item_ids.npy").exists():
        item_ids, term_ids, n_pairs = build_indices()
    else:
        item_ids, term_ids = np.load(OUT / "item_ids.npy"), np.load(OUT / "term_ids.npy")
        n_pairs = json.loads((OUT / "index.json").read_text(encoding="utf8"))["pairs"]
    if args.stage in {"all", "encode"}:
        model = load_model(); encode_queries(model, term_ids); encode_items(model, item_ids)
    if args.stage in {"all", "score"}:
        score_pairs(item_ids, term_ids, n_pairs)


if __name__ == "__main__":
    main()
