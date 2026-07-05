from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

LABEL_PATH = Path("reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv")
SCORE_PATH = Path("data/processed/v5_e5base_full900_test_proba.parquet")
ITEMS_PATH = Path("data/raw/items.csv")
OUT = Path("reports/manual_review/v5_0p7_postprocess_rule_test.csv")

TH = 0.70

def norm(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def word_in(text, words):
    t = " " + norm(text) + " "
    return any((" " + norm(w) + " ") in t for w in words)

def metric(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro"),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

df = pd.read_csv(LABEL_PATH)
df["id"] = df["id"].astype(str)
df = df[df["assistant_label"].isin([0, 1, "0", "1"])].copy()
df["assistant_label"] = df["assistant_label"].astype(int)

scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)
df = df.merge(scores, on="id", how="left", validate="many_to_one")
df = df.dropna(subset=["proba_avg"]).copy()

items = pd.read_csv(ITEMS_PATH, usecols=["brand"])
brands = (
    items["brand"]
    .dropna()
    .astype(str)
    .map(norm)
    .value_counts()
)
brand_vocab = [
    b for b in brands.index
    if len(b) >= 3 and b not in {"diger", "other", "marka", "unisex", "no brand", "yok"}
]
brand_vocab = sorted(set(brand_vocab), key=len, reverse=True)

def query_brand_hits(q):
    qn = " " + norm(q) + " "
    hits = []
    for b in brand_vocab:
        if " " + b + " " in qn:
            hits.append(b)
            if len(hits) >= 3:
                break
    return hits

def brand_mismatch(row):
    hits = query_brand_hits(row.get("query", ""))
    if not hits:
        return False
    item_brand = norm(row.get("brand", ""))
    if not item_brand:
        return False
    return all(h != item_brand for h in hits)

male_words = ["erkek", "bay", "men", "male"]
female_words = ["kadin", "kadın", "bayan", "kiz", "kız", "women", "woman", "female"]

def gender_mismatch(row):
    q = row.get("query", "")
    item_text = " ".join([
        str(row.get("gender", "")),
        str(row.get("title", "")),
        str(row.get("category", "")),
    ])

    q_male = word_in(q, male_words)
    q_female = word_in(q, female_words)

    item_male = word_in(item_text, male_words)
    item_female = word_in(item_text, female_words)

    return (q_male and item_female) or (q_female and item_male)

df["base_pred"] = (df["proba_avg"] >= TH).astype(int)
df["brand_mismatch"] = df.apply(brand_mismatch, axis=1)
df["gender_mismatch"] = df.apply(gender_mismatch, axis=1)

rules = {}

rules["base_0p7"] = df["base_pred"].to_numpy()

p = df["base_pred"].copy()
p[(p == 1) & df["brand_mismatch"]] = 0
rules["minus_brand_mismatch"] = p.to_numpy()

p = df["base_pred"].copy()
p[(p == 1) & df["gender_mismatch"]] = 0
rules["minus_gender_mismatch"] = p.to_numpy()

p = df["base_pred"].copy()
p[(p == 1) & (df["brand_mismatch"] | df["gender_mismatch"])] = 0
rules["minus_brand_or_gender_mismatch"] = p.to_numpy()

rows = []
y = df["assistant_label"].to_numpy()

for name, pred in rules.items():
    row = {"rule": name, "n": len(df)}
    row.update(metric(y, pred))
    rows.append(row)

res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
res.to_csv(OUT, index=False)

print(res.to_string(index=False))
print()
print("brand_mismatch among base positives:", int(((df["base_pred"] == 1) & df["brand_mismatch"]).sum()))
print("gender_mismatch among base positives:", int(((df["base_pred"] == 1) & df["gender_mismatch"]).sum()))
print("saved:", OUT)
