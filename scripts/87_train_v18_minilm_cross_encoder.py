from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(".")

DATA = ROOT / "data/processed/v18_reranker_train_pairs.csv"
OUT_MODEL = ROOT / "models/v18_minilm_cross_encoder"
REPORT = ROOT / "reports/manual_review/v18_minilm_train_report.csv"

SEED = 2026
MAX_ROWS = 360000
EPOCHS = 1
BATCH_SIZE = 16
LR = 2e-5
MAX_LENGTH = 192

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("loading train data...")
df = pd.read_csv(DATA)

df = df.dropna(subset=["query", "item_text", "label"]).copy()
df["label"] = df["label"].astype(int)

# Dengeli ama hard-negative ağırlıklı sample
pos = df[df["label"] == 1]
neg = df[df["label"] == 0]

target_pos = min(len(pos), MAX_ROWS // 4)
target_neg = min(len(neg), MAX_ROWS - target_pos)

pos_s = pos.sample(n=target_pos, random_state=SEED)
neg_s = neg.sample(n=target_neg, random_state=SEED)

df = pd.concat([pos_s, neg_s], ignore_index=True)
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print("used rows:", len(df))
print(df["label"].value_counts().to_string())
print(df["neg_type"].value_counts().to_string())

train_df, dev_df = train_test_split(
    df,
    test_size=0.08,
    random_state=SEED,
    stratify=df["label"],
)

train_examples = [
    InputExample(texts=[str(r.query), str(r.item_text)], label=float(r.label))
    for r in train_df.itertuples(index=False)
]

train_loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE,
)

model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

print("loading model:", model_name)
model = CrossEncoder(
    model_name,
    num_labels=1,
    max_length=MAX_LENGTH,
    automodel_args={"ignore_mismatched_sizes": True},
)

warmup_steps = int(len(train_loader) * EPOCHS * 0.08)

print("training...")
model.fit(
    train_dataloader=train_loader,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LR},
    use_amp=torch.cuda.is_available(),
    show_progress_bar=True,
    output_path=None,
)

print("saving model explicitly...")
OUT_MODEL.mkdir(parents=True, exist_ok=True)
model.save(str(OUT_MODEL))

print("evaluating dev...")
pairs = list(zip(dev_df["query"].astype(str), dev_df["item_text"].astype(str)))
scores = model.predict(pairs, batch_size=64, show_progress_bar=True)
scores = 1 / (1 + np.exp(-np.asarray(scores)))

y = dev_df["label"].to_numpy()

rows = []
for th in np.arange(0.05, 0.96, 0.025):
    pred = (scores >= th).astype(int)
    rows.append({
        "threshold": float(th),
        "macro_f1": f1_score(y, pred, average="macro"),
        "positive_f1": f1_score(y, pred, pos_label=1),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "pred_pos_ratio": float(pred.mean()),
        "true_pos_ratio": float(y.mean()),
    })

rep = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
rep["dev_auc"] = roc_auc_score(y, scores)
rep["train_rows"] = len(train_df)
rep["dev_rows"] = len(dev_df)
rep["model_path"] = str(OUT_MODEL)
rep.to_csv(REPORT, index=False)

print(rep.head(20).to_string(index=False))
print("saved model:", OUT_MODEL)
print("report:", REPORT)
