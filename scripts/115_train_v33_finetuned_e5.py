from pathlib import Path
import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup


ROOT = Path(".")

TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
TRAINING_PAIRS = ROOT / "data/raw/training_pairs.csv"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class PairDataset(Dataset):
    def __init__(self, queries, passages):
        self.queries = queries
        self.passages = passages

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.passages[idx]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def encode_batch(model, batch, tokenizer, max_len, device):
    enc = tokenizer(
        list(batch),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)
    return emb


def collate_fn(batch):
    q, p = zip(*batch)
    return list(q), list(p)


@torch.no_grad()
def evaluate_loss(model, loader, tokenizer, device, max_query_len, max_item_len, temperature, max_batches=80):
    model.eval()
    losses = []
    correct_q2p = 0
    total = 0

    for step, (q_batch, p_batch) in enumerate(loader):
        if step >= max_batches:
            break

        q_emb = encode_batch(model, q_batch, tokenizer, max_query_len, device)
        p_emb = encode_batch(model, p_batch, tokenizer, max_item_len, device)

        logits = torch.matmul(q_emb, p_emb.T) / temperature
        labels = torch.arange(logits.size(0), device=device)

        loss_qp = F.cross_entropy(logits, labels)
        loss_pq = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_qp + loss_pq)

        pred = logits.argmax(dim=1)
        correct_q2p += int((pred == labels).sum().item())
        total += int(labels.numel())
        losses.append(float(loss.item()))

    model.train()

    return {
        "val_loss": float(np.mean(losses)) if losses else None,
        "val_inbatch_top1": float(correct_q2p / max(1, total)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--output-dir", default="models/v33_e5_ft")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--max-query-len", type=int, default=48)
    parser.add_argument("--max-item-len", type=int, default=160)
    parser.add_argument("--val-size", type=int, default=12000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("loading csv files...")
    terms = pd.read_csv(TERMS)
    items = pd.read_csv(ITEMS, low_memory=False)
    pairs = pd.read_csv(TRAINING_PAIRS)

    terms["term_id"] = terms["term_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if c not in items.columns:
            items[c] = ""

    print("building texts...")
    terms["query_text"] = terms["query"].map(build_query_text)
    items["item_text"] = items.apply(build_item_text, axis=1)

    df = pairs[["term_id", "item_id"]].drop_duplicates().copy()
    df = df.merge(terms[["term_id", "query_text"]], on="term_id", how="left", validate="many_to_one")
    df = df.merge(items[["item_id", "item_text"]], on="item_id", how="left", validate="many_to_one")
    df = df.dropna(subset=["query_text", "item_text"]).reset_index(drop=True)

    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    val_size = min(args.val_size, max(1000, len(df) // 20))
    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()

    print("total positive pairs:", len(df))
    print("train:", len(train_df))
    print("val:", len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.train()

    train_ds = PairDataset(
        train_df["query_text"].tolist(),
        train_df["item_text"].tolist(),
    )
    val_ds = PairDataset(
        val_df["query_text"].tolist(),
        val_df["item_text"].tolist(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=collate_fn,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = vars(args).copy()
    config.update({
        "device": device,
        "total_positive_pairs": int(len(df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
    })

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("training starts")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    global_step = 0
    running_loss = []
    best_val = 999.0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    epoch = 0
    while global_step < total_steps:
        epoch += 1

        for micro_step, (q_batch, p_batch) in enumerate(train_loader):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                q_emb = encode_batch(model, q_batch, tokenizer, args.max_query_len, device)
                p_emb = encode_batch(model, p_batch, tokenizer, args.max_item_len, device)

                logits = torch.matmul(q_emb, p_emb.T) / args.temperature
                labels = torch.arange(logits.size(0), device=device)

                loss_qp = F.cross_entropy(logits, labels)
                loss_pq = F.cross_entropy(logits.T, labels)
                loss = 0.5 * (loss_qp + loss_pq)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (micro_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss.append(float(loss.item() * args.grad_accum))

                if global_step % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = float(np.mean(running_loss[-50:]))
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"step={global_step}/{total_steps} "
                        f"epoch={epoch} "
                        f"loss={avg_loss:.5f} "
                        f"lr={lr_now:.2e} "
                        f"elapsed_min={elapsed/60:.1f}"
                    )

                if global_step % 500 == 0 or global_step == total_steps:
                    val = evaluate_loss(
                        model=model,
                        loader=val_loader,
                        tokenizer=tokenizer,
                        device=device,
                        max_query_len=args.max_query_len,
                        max_item_len=args.max_item_len,
                        temperature=args.temperature,
                        max_batches=80,
                    )

                    print(
                        f"VAL step={global_step} "
                        f"val_loss={val['val_loss']:.5f} "
                        f"val_inbatch_top1={val['val_inbatch_top1']:.4f}"
                    )

                    with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "step": int(global_step),
                            "train_loss_last50": float(np.mean(running_loss[-50:])),
                            **val,
                        }, ensure_ascii=False) + "\n")

                    if val["val_loss"] is not None and val["val_loss"] < best_val:
                        best_val = val["val_loss"]
                        best_dir = out_dir / "best"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print("saved best:", best_dir)

                if args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt = out_dir / f"checkpoint_step_{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print("saved checkpoint:", ckpt)

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("done")
    print("final model:", final_dir)
    print("best model:", out_dir / "best")


if __name__ == "__main__":
    main()
