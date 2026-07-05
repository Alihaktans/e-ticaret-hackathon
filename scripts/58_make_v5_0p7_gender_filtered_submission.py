from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
BASE_SUB = ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_RANDOM_VALIDATED.csv"
SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT = ROOT / "submissions/FINAL_CANDIDATE_v5_full900_threshold_0p7_GENDER_FILTERED.csv"
REPORT = ROOT / "reports/manual_review/v5_0p7_gender_filter_full_report.csv"

def norm_series(s):
    tr = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.translate(tr)
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def word_contains(series, words):
    pat = r"(^| )(" + "|".join(words) + r")( |$)"
    return series.str.contains(pat, regex=True, na=False)

base = pd.read_csv(BASE_SUB)
sample = pd.read_csv(SAMPLE)

assert base["id"].astype(str).reset_index(drop=True).equals(
    sample["id"].astype(str).reset_index(drop=True)
), "Base submission id order sample_submission ile aynı değil."

pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
assert pairs["id"].astype(str).reset_index(drop=True).equals(
    base["id"].astype(str).reset_index(drop=True)
), "submission_pairs id order base ile aynı değil."

pred = base["prediction"].to_numpy().astype(np.int8)
pos_idx = np.flatnonzero(pred == 1)

pos = pairs.iloc[pos_idx].copy()
pos["term_id"] = pos["term_id"].astype(str)
pos["item_id"] = pos["item_id"].astype(str)

terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)

items = pd.read_csv(ITEMS, usecols=["item_id", "title", "category", "gender"])
items["item_id"] = items["item_id"].astype(str)

pos = pos.merge(terms[["term_id", "query"]], on="term_id", how="left", validate="many_to_one")
pos = pos.merge(items, on="item_id", how="left", validate="many_to_one")

q = norm_series(pos["query"])
item_text = norm_series(
    pos["gender"].fillna("").astype(str) + " " +
    pos["title"].fillna("").astype(str) + " " +
    pos["category"].fillna("").astype(str)
)

male_words = ["erkek", "bay", "men", "male"]
female_words = ["kadin", "bayan", "kiz", "women", "woman", "female"]

q_male = word_contains(q, male_words)
q_female = word_contains(q, female_words)
item_male = word_contains(item_text, male_words)
item_female = word_contains(item_text, female_words)

# Güvenli gender mismatch:
# Query net erkek ama item net kadın/kız.
# Query net kadın/kız ama item net erkek.
# Unisex/ikisi birden geçen ürünleri silmiyoruz.
gender_mismatch = (
    (q_male & ~q_female & item_female & ~item_male)
    |
    (q_female & ~q_male & item_male & ~item_female)
)

flip_indices = pos_idx[gender_mismatch.to_numpy()]
pred[flip_indices] = 0

out = pd.DataFrame({
    "id": base["id"],
    "prediction": pred,
})
out.to_csv(OUT, index=False)

report = pd.DataFrame([{
    "base_file": str(BASE_SUB),
    "out_file": str(OUT),
    "rows": len(out),
    "base_ones": int(base["prediction"].sum()),
    "new_ones": int(pred.sum()),
    "flipped_1_to_0": int(len(flip_indices)),
    "base_pos_ratio": float(base["prediction"].mean()),
    "new_pos_ratio": float(pred.mean()),
    "id_order_ok": bool(out["id"].astype(str).reset_index(drop=True).equals(sample["id"].astype(str).reset_index(drop=True))),
}])
report.to_csv(REPORT, index=False)

print(report.to_string(index=False))
print("saved:", OUT)
print("report:", REPORT)
