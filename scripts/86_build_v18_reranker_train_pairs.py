from pathlib import Path
import re
import random
import numpy as np
import pandas as pd

ROOT = Path(".")

TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT = ROOT / "data/processed/v18_reranker_train_pairs.csv"
REPORT = ROOT / "reports/manual_review/v18_reranker_train_pairs_report.csv"

RANDOM_SEED = 2026
MAX_POS = 160000
NEG_PER_POS = 3

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def root_category(x):
    s = "" if pd.isna(x) else str(x)
    for sep in [">", "/", "|"]:
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()

def item_text(row):
    return (
        "title: " + str(row.get("title", "") or "")
        + " | category: " + str(row.get("category", "") or "")
        + " | brand: " + str(row.get("brand", "") or "")
        + " | gender: " + str(row.get("gender", "") or "")
        + " | age_group: " + str(row.get("age_group", "") or "")
        + " | attributes: " + str(row.get("attributes", "") or "")[:350]
    )

print("loading data...")
train = pd.read_csv(TRAIN_PAIRS)
terms = pd.read_csv(TERMS)
items = pd.read_csv(ITEMS, low_memory=False)

train["term_id"] = train["term_id"].astype(str)
train["item_id"] = train["item_id"].astype(str)
terms["term_id"] = terms["term_id"].astype(str)
items["item_id"] = items["item_id"].astype(str)

items["root_category"] = items["category"].map(root_category)
items["item_text"] = items.apply(item_text, axis=1)
items["title_norm"] = items["title"].map(norm)

terms["query_norm"] = terms["query"].map(norm)

pos = (
    train[["term_id", "item_id", "label"]]
    .merge(terms[["term_id", "query", "query_norm"]], on="term_id", how="left")
    .merge(items[["item_id", "item_text", "root_category", "title_norm"]], on="item_id", how="left")
)

pos = pos.dropna(subset=["query", "item_text"]).copy()
pos = pos.sample(n=min(MAX_POS, len(pos)), random_state=RANDOM_SEED).reset_index(drop=True)

positive_pairs = set(zip(pos["term_id"], pos["item_id"]))

items_by_root = {
    k: g["item_id"].to_numpy()
    for k, g in items.groupby("root_category", sort=False)
}

all_item_ids = items["item_id"].to_numpy()
item_lookup = items.set_index("item_id")[["item_text", "root_category", "title_norm"]].to_dict("index")

rows = []

print("adding positives...")
for r in pos.itertuples(index=False):
    rows.append({
        "query": r.query,
        "item_text": r.item_text,
        "label": 1,
        "weight": 1.0,
        "neg_type": "positive",
        "term_id": r.term_id,
        "item_id": r.item_id,
    })

print("sampling negatives...")
for i, r in enumerate(pos.itertuples(index=False), start=1):
    q_tokens = set(str(r.query_norm).split())
    root_pool = items_by_root.get(r.root_category, all_item_ids)

    made = 0
    tried = 0

    while made < NEG_PER_POS and tried < 100:
        tried += 1

        if made == 0 and len(root_pool) > 2:
            cand_id = str(np.random.choice(root_pool))
            neg_type = "same_root_hard"
            weight = 1.15
        elif made == 1:
            cand_id = str(np.random.choice(all_item_ids))
            neg_type = "random_easy"
            weight = 0.65
        else:
            cand_id = str(np.random.choice(root_pool if len(root_pool) > 2 else all_item_ids))
            neg_type = "same_root_extra"
            weight = 1.0

        if cand_id == r.item_id:
            continue
        if (r.term_id, cand_id) in positive_pairs:
            continue

        item = item_lookup.get(cand_id)
        if item is None:
            continue

        title_tokens = set(str(item["title_norm"]).split())
        overlap = len(q_tokens & title_tokens)

        # Query neredeyse tamamen title içinde geçiyorsa bunu negatif yapma.
        if overlap >= max(3, len(q_tokens)):
            continue

        rows.append({
            "query": r.query,
            "item_text": item["item_text"],
            "label": 0,
            "weight": weight,
            "neg_type": neg_type,
            "term_id": r.term_id,
            "item_id": cand_id,
        })

        made += 1

    if i % 20000 == 0:
        print(f"processed {i}/{len(pos)}")

df = pd.DataFrame(rows)
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
df.to_csv(OUT, index=False, encoding="utf-8-sig")

report = pd.DataFrame([{
    "rows": len(df),
    "positives": int((df["label"] == 1).sum()),
    "negatives": int((df["label"] == 0).sum()),
    "pos_ratio": float(df["label"].mean()),
    "unique_terms": df["term_id"].nunique(),
    "unique_items": df["item_id"].nunique(),
}])

report.to_csv(REPORT, index=False)

print(report.to_string(index=False))
print(df["neg_type"].value_counts().to_string())
print("saved:", OUT)
print("report:", REPORT)
