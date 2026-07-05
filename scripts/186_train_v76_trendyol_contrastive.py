"""Competition-positive contrastive adaptation of Trendyol's e-commerce encoder.

One product is sampled per query per epoch, preventing same-query positives from
becoming in-batch false negatives.  A deterministic query holdout is never used
for training and is written for candidate-shaped retrieval evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset


ROOT = Path(".")
MODEL_ID = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Trendyol--TY-ecomm-embed-multilingual-base-v1.2.0/snapshots/00c030c9a56bff9403f95c1b45f4b82e669e243c"
TRAIN = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"


def clean(x, limit):
    if pd.isna(x): return ""
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split())[:limit]


def product_text(r):
    # Exactly the format that produced the public-0.81 v71 signal.
    return f"Başlık: {clean(r.title,300)} | Kategori: {clean(r.category,220)} | Marka: {clean(r.brand,100)}"


def fold(term_id: str) -> int:
    return int(hashlib.md5(str(term_id).encode()).hexdigest()[:8], 16) % 10


class OnePositivePerQuery(Dataset):
    def __init__(self, terms, positives, query_map, item_map, seed):
        self.terms = list(terms); self.positives = positives
        self.query_map = query_map; self.item_map = item_map; self.rng = random.Random(seed)
    def __len__(self): return len(self.terms)
    def __getitem__(self, i):
        term = self.terms[i]
        item = self.rng.choice(self.positives[term])
        return InputExample(texts=[self.query_map[term], self.item_map[item]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="models/v76_trendyol_contrastive")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--temperature", type=float, default=.05)
    ap.add_argument("--seed", type=int, default=20260703)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(TRAIN, usecols=["term_id", "item_id"], dtype={"term_id": "string", "item_id": "string"}).drop_duplicates()
    pairs["fold"] = pairs.term_id.map(fold)
    holdout = sorted(pairs.loc[pairs.fold.eq(0), "term_id"].unique())
    train = pairs[~pairs.term_id.isin(holdout)].copy()
    positives = train.groupby("term_id").item_id.apply(list).to_dict()

    terms = pd.read_csv(TERMS, usecols=["term_id", "query"], dtype={"term_id": "string"})
    query_map = terms.set_index("term_id")["query"].fillna("").astype(str).to_dict()
    needed = set(train.item_id.astype(str)); parts = []
    for chunk in pd.read_csv(ITEMS, usecols=["item_id", "title", "category", "brand"], chunksize=100_000, low_memory=False):
        chunk.item_id = chunk.item_id.astype(str); keep = chunk.item_id.isin(needed)
        if keep.any(): parts.append(chunk.loc[keep].copy())
    items = pd.concat(parts, ignore_index=True).drop_duplicates("item_id")
    item_map = {str(r.item_id): product_text(r) for r in items.itertuples(index=False)}
    missing = needed - set(item_map)
    if missing: raise RuntimeError(f"missing item texts: {len(missing)}")

    (out / "holdout_terms.txt").write_text("\n".join(holdout), encoding="utf8")
    config = vars(args) | {
        "base_model": MODEL_ID, "base_revision": "00c030c9", "snapshot": str(SNAPSHOT),
        "train_queries": len(positives), "holdout_queries": len(holdout),
        "train_positive_pairs": len(train), "false_negative_control": "one sampled positive per query per epoch",
        "product_format": "Başlık | Kategori | Marka",
    }
    (out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = SentenceTransformer(str(SNAPSHOT), trust_remote_code=True, local_files_only=True, device="cuda")
    model.max_seq_length = 192
    dataset = OnePositivePerQuery(sorted(positives), positives, query_map, item_map, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    loss = losses.MultipleNegativesRankingLoss(model, scale=1.0 / args.temperature)
    steps = len(loader) * args.epochs
    model.fit(
        train_objectives=[(loader, loss)], epochs=args.epochs,
        warmup_steps=max(10, int(.08 * steps)), optimizer_params={"lr": args.lr},
        weight_decay=.01, use_amp=True, show_progress_bar=True,
        output_path=str(out / "final"), checkpoint_path=str(out / "checkpoints"),
        checkpoint_save_steps=500, checkpoint_save_total_limit=2,
    )
    print("saved", out / "final", flush=True)


if __name__ == "__main__":
    main()
