from pathlib import Path
import argparse
import json
import time
import random
import math
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm


ROOT = Path(".")
DATA = ROOT / "data/processed/v34/v34_hard_negative_pairs.parquet"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, df):
        self.query = df["query_text"].fillna("query: ").astype(str).tolist()
        self.item = df["item_text"].fillna("passage: ").astype(str).tolist()
        self.label = df["label"].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.query[idx], self.item[idx], self.label[idx]


def collate_fn(batch, tokenizer, max_len):
    q, item, y = zip(*batch)

    enc = tokenizer(
        list(q),
        list(item),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    labels = torch.tensor(y, dtype=torch.float32)
    return enc, labels


@torch.no_grad()
def evaluate(model, loader, device, max_batches=999999):
    model.eval()

    losses = []
    ys = []
    ps = []

    bce = torch.nn.BCEWithLogitsLoss()

    for step, (enc, labels) in enumerate(loader):
        if step >= max_batches:
            break

        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(**enc).logits.squeeze(-1)
            loss = bce(logits, labels)

        prob = torch.sigmoid(logits).detach().cpu().numpy()

        losses.append(float(loss.item()))
        ys.append(labels.detach().cpu().numpy())
        ps.append(prob)

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    pred = (p >= 0.5).astype(int)

    out = {
        "loss": float(np.mean(losses)),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "macro_f1_0p5": float(f1_score(y, pred, average="macro")),
        "precision_0p5": float(precision_score(y, pred, zero_division=0)),
        "recall_0p5": float(recall_score(y, pred, zero_division=0)),
        "pred_pos_ratio_0p5": float(pred.mean()),
        "true_pos_ratio": float(y.mean()),
    }

    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pp = (p >= t).astype(int)
        out[f"macro_f1_{str(t).replace('.', 'p')}"] = float(f1_score(y, pp, average="macro"))
        out[f"precision_{str(t).replace('.', 'p')}"] = float(precision_score(y, pp, zero_division=0))
        out[f"recall_{str(t).replace('.', 'p')}"] = float(recall_score(y, pp, zero_division=0))
        out[f"pred_pos_ratio_{str(t).replace('.', 'p')}"] = float(pp.mean())

    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--output-dir", default="models/v34_cross_encoder")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--val-terms", type=int, default=1800)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2026)
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

    print("loading data...")
    df = pd.read_parquet(DATA)
    df["term_id"] = df["term_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["label"] = df["label"].astype(int)

    # Validation split by term_id. Böylece aynı query train/val'e karışmıyor.
    terms = np.array(sorted(df["term_id"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(terms)

    val_terms = set(terms[:min(args.val_terms, len(terms) // 5)])
    val_df = df[df["term_id"].isin(val_terms)].copy()
    train_df = df[~df["term_id"].isin(val_terms)].copy()

    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print("train rows:", len(train_df), "pos_rate:", train_df["label"].mean(), "terms:", train_df["term_id"].nunique())
    print("val rows:", len(val_df), "pos_rate:", val_df["label"].mean(), "terms:", val_df["term_id"].nunique())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="single_label_classification",
    )
    model.to(device)
    model.train()

    train_ds = PairDataset(train_df)
    val_ds = PairDataset(val_df)

    def collate(batch):
        return collate_fn(batch, tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=collate,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)

    if args.max_steps > 0:
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

    pos = float(train_df["label"].sum())
    neg = float(len(train_df) - train_df["label"].sum())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    config = vars(args).copy()
    config.update({
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_pos_rate": float(train_df["label"].mean()),
        "val_pos_rate": float(val_df["label"].mean()),
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "pos_weight": float(pos_weight.item()),
    })

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(json.dumps(config, ensure_ascii=False, indent=2))

    best_ap = -1.0
    running = []
    global_step = 0
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        for micro_step, (enc, labels) in enumerate(train_loader):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(**enc).logits.squeeze(-1)
                loss = bce(logits, labels)
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
                running.append(float(loss.item() * args.grad_accum))

                if global_step % 100 == 0:
                    print(
                        f"step={global_step}/{total_steps} "
                        f"loss={np.mean(running[-100:]):.5f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"elapsed_min={(time.time()-start)/60:.1f}"
                    )

                if global_step % args.eval_every == 0 or global_step == total_steps:
                    val = evaluate(model, val_loader, device, max_batches=999999)

                    log_row = {
                        "step": int(global_step),
                        "train_loss_last100": float(np.mean(running[-100:])),
                        **val,
                    }

                    print("VAL", json.dumps(log_row, ensure_ascii=False, indent=2))

                    with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

                    if val["ap"] > best_ap:
                        best_ap = val["ap"]
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
    print("best:", out_dir / "best")
    print("final:", final_dir)


if __name__ == "__main__":
    main()
