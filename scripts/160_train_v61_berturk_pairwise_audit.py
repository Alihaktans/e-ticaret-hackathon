from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup


ROOT = Path(".")
DATA = ROOT / "data/processed/v61_berturk_pairwise_triplets.parquet"


class TripletDataset(Dataset):
    def __init__(self, frame):
        self.q = frame.query_text.astype(str).tolist()
        self.p = frame.pos_text.astype(str).tolist()
        self.n = frame.neg_text.astype(str).tolist()
        self.w = frame.confidence_weight.astype(np.float32).to_numpy()
        self.source = frame.source_type.astype(str).tolist()

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return self.q[i], self.p[i], self.n[i], self.w[i], self.source[i]


def collate(batch, tokenizer, max_len, random_swap=False):
    q, p, n, w, source = zip(*batch)
    left, right, y = list(p), list(n), np.ones(len(batch), dtype=np.float32)
    if random_swap:
        swap = np.random.random(len(batch)) < .5
        for i in np.flatnonzero(swap):
            left[i], right[i] = right[i], left[i]
            y[i] = 0.0
    all_q = list(q) + list(q)
    all_item = left + right
    enc = tokenizer(all_q, all_item, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc, torch.tensor(y), torch.tensor(w), list(source)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    margins, sources = [], []
    for enc, _, _, source in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**enc).logits.view(-1).float()
        half = len(source)
        margins.extend((logits[:half] - logits[half:]).cpu().numpy().tolist())
        sources.extend(source)
    d = pd.DataFrame({"margin": margins, "source": sources})
    result = {"n": len(d), "pair_accuracy": float((d.margin > 0).mean()), "margin_mean": float(d.margin.mean())}
    for source, g in d.groupby("source"):
        result[source] = {"n": len(g), "pair_accuracy": float((g.margin > 0).mean()), "margin_mean": float(g.margin.mean()), "margin_p10": float(g.margin.quantile(.1))}
    model.train()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--output-dir", default="models/v61_berturk_pairwise_audit")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--holdout-source", choices=["v29", "v33"], default="v33")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--holdout-all-fold0", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    d = pd.read_parquet(args.data)
    # One manual preference source is a strict holdout and never enters audit training.
    holdout_name = "manual_preference_" + args.holdout_source
    manual_val = d[d.source_type.eq(holdout_name)].copy()
    if args.holdout_all_fold0:
        clean_val = d[d.fold.eq(0) & ~d.source_type.str.startswith("manual_preference_")].copy()
        train = d[~d.source_type.eq(holdout_name) & ~(d.fold.eq(0) & ~d.source_type.str.startswith("manual_preference_"))].copy()
    else:
        clean_val = d[d.source_type.eq("clean_cross_root_v33_hard") & d.fold.eq(0)].copy()
        train = d[~d.source_type.eq(holdout_name) & ~(d.source_type.eq("clean_cross_root_v33_hard") & d.fold.eq(0))].copy()
    train = train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val = pd.concat([manual_val, clean_val], ignore_index=True)
    print("train", len(train), train.source_type.value_counts().to_dict())
    print("val", len(val), val.source_type.value_counts().to_dict())

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(device)
    model.train()
    tr_ds, va_ds = TripletDataset(train), TripletDataset(val)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda", drop_last=True, collate_fn=lambda x: collate(x, tok, args.max_len, True))
    va_loader = DataLoader(va_ds, batch_size=args.batch_size * 3, shuffle=False, num_workers=0, pin_memory=device == "cuda", collate_fn=lambda x: collate(x, tok, args.max_len, False))

    steps_epoch = math.ceil(len(tr_loader) / args.grad_accum)
    total_steps = int(steps_epoch * args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    sched = get_cosine_schedule_with_warmup(opt, int(.08 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    config = vars(args) | {"train_rows": len(train), "manual_holdout_rows": len(manual_val), "clean_val_rows": len(clean_val), "total_steps": total_steps}
    (out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    global_step = 0; best_manual = -1.; running = []; start = time.time(); opt.zero_grad(set_to_none=True)
    while global_step < total_steps:
        for micro, (enc, y, weight, _) in enumerate(tr_loader):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}; y = y.to(device); weight = weight.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits.view(-1).float(); half = len(y); margin = logits[:half] - logits[half:]
                loss = (bce(margin, y) * weight).mean() / args.grad_accum
            scaler.scale(loss).backward(); running.append(float(loss.item() * args.grad_accum))
            if (micro + 1) % args.grad_accum:
                continue
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); global_step += 1
            if global_step % 50 == 0:
                print("step", global_step, "/", total_steps, "loss", round(float(np.mean(running[-100:])), 4), "min", round((time.time()-start)/60, 1), flush=True)
            if global_step % args.eval_every == 0 or global_step == total_steps:
                metrics = evaluate(model, va_loader, device); metrics |= {"step": global_step, "train_loss_last100": float(np.mean(running[-100:]))}
                print("EVAL", json.dumps(metrics, ensure_ascii=False), flush=True)
                with (out / "train_log.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                manual_acc = metrics.get(holdout_name, {}).get("pair_accuracy", 0)
                if manual_acc > best_manual:
                    best_manual = manual_acc; best = out / "best"; best.mkdir(exist_ok=True); model.save_pretrained(best); tok.save_pretrained(best); (best / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            if global_step >= total_steps: break
    final = out / "final"; final.mkdir(exist_ok=True); model.save_pretrained(final); tok.save_pretrained(final)
    print("done best_manual", best_manual, "elapsed_min", (time.time()-start)/60)


if __name__ == "__main__":
    main()
