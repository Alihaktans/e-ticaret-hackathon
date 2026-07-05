from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers.cross_encoder import CrossEncoder

ROOT = Path(".")

MODEL = ROOT / "models/v18_minilm_cross_encoder"

CANDIDATE_PATHS = [
    ROOT / "data/processed/v13_cross_encoder_candidate_pairs.parquet",
    ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet",
    ROOT / "data/processed/v15_bge_reranker_candidate_scores.parquet",
]

SUB_PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"
REPORT = ROOT / "reports/manual_review/v18_minilm_candidate_score_report.csv"

BATCH_SIZE = 64
MAX_LENGTH = 192

def item_text(row):
    return (
        "title: " + str(row.get("title", "") or "")
        + " | category: " + str(row.get("category", "") or "")
        + " | brand: " + str(row.get("brand", "") or "")
        + " | gender: " + str(row.get("gender", "") or "")
        + " | age_group: " + str(row.get("age_group", "") or "")
        + " | attributes: " + str(row.get("attributes", "") or "")[:350]
    )

print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

cand_path = None
for p in CANDIDATE_PATHS:
    if p.exists():
        cand_path = p
        break

if cand_path is None:
    raise FileNotFoundError("No candidate file found.")

print("loading candidates:", cand_path)
cand = pd.read_parquet(cand_path)
cand["id"] = cand["id"].astype(str)

if "term_id" not in cand.columns or "item_id" not in cand.columns:
    print("candidate file lacks term_id/item_id; merging submission_pairs...")
    pairs = pd.read_csv(SUB_PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)
    cand = cand[["id"]].merge(pairs, on="id", how="left", validate="one_to_one")
else:
    cand = cand[["id", "term_id", "item_id"]].copy()
    cand["term_id"] = cand["term_id"].astype(str)
    cand["item_id"] = cand["item_id"].astype(str)

print("candidate rows:", len(cand))

terms = pd.read_csv(TERMS, usecols=["term_id", "query"])
terms["term_id"] = terms["term_id"].astype(str)

needed_items = set(cand["item_id"].unique())

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)
items = items[items["item_id"].isin(needed_items)].copy()
items["item_text"] = items.apply(item_text, axis=1)

df = (
    cand
    .merge(terms, on="term_id", how="left", validate="many_to_one")
    .merge(items[["item_id", "item_text"]], on="item_id", how="left", validate="many_to_one")
)

df["query"] = df["query"].fillna("").astype(str)
df["item_text"] = df["item_text"].fillna("").astype(str)

print("missing query:", int((df["query"] == "").sum()))
print("missing item_text:", int((df["item_text"] == "").sum()))

print("loading model:", MODEL)
model = CrossEncoder(str(MODEL), max_length=MAX_LENGTH)

pairs = list(zip(df["query"].tolist(), df["item_text"].tolist()))

print("scoring...")
raw = model.predict(
    pairs,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
)

raw = np.asarray(raw, dtype=np.float32)
sig = 1.0 / (1.0 + np.exp(-raw))

out = pd.DataFrame({
    "id": df["id"].to_numpy(),
    "v18_minilm_logit": raw,
    "v18_minilm_sigmoid": sig.astype(np.float32),
})

out.to_parquet(OUT, index=False)

report = pd.DataFrame([{
    "rows": len(out),
    "sigmoid_mean": float(out["v18_minilm_sigmoid"].mean()),
    "sigmoid_std": float(out["v18_minilm_sigmoid"].std()),
    "sigmoid_min": float(out["v18_minilm_sigmoid"].min()),
    "sigmoid_p01": float(out["v18_minilm_sigmoid"].quantile(0.01)),
    "sigmoid_p05": float(out["v18_minilm_sigmoid"].quantile(0.05)),
    "sigmoid_p25": float(out["v18_minilm_sigmoid"].quantile(0.25)),
    "sigmoid_p50": float(out["v18_minilm_sigmoid"].quantile(0.50)),
    "sigmoid_p75": float(out["v18_minilm_sigmoid"].quantile(0.75)),
    "sigmoid_p95": float(out["v18_minilm_sigmoid"].quantile(0.95)),
    "sigmoid_p99": float(out["v18_minilm_sigmoid"].quantile(0.99)),
    "sigmoid_max": float(out["v18_minilm_sigmoid"].max()),
}])

report.to_csv(REPORT, index=False)

print(report.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
