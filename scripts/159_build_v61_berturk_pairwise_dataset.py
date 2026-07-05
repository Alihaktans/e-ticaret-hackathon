from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
V34 = ROOT / "data/processed/v34/v34_hard_negative_pairs.parquet"
ITEMS = ROOT / "data/raw/items.csv"
TERMS = ROOT / "data/raw/terms.csv"
LABELS = [
    ("v29", ROOT / "reports/manual_review/review_v29_qswap_targets_assistant_labeled.csv"),
    ("v33", ROOT / "reports/manual_review/review_v33_final_qswap_targets_assistant_labeled.csv"),
]
OUT = ROOT / "data/processed/v61_berturk_pairwise_triplets.parquet"
REPORT = ROOT / "reports/experiments/v61_berturk_pairwise_dataset_report.json"
SAMPLE = ROOT / "reports/manual_review/v61_berturk_pairwise_dataset_sample.csv"


TR = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c"})
STOP = {"ve", "ile", "icin", "bir", "bu", "set", "takim", "adet", "model", "urun", "renk"}


def norm(x):
    x = str(x or "")
    try:
        x = x.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    x = unicodedata.normalize("NFKD", x.translate(TR).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", x)).strip()


def tokens(x):
    return {t for t in norm(x).split() if len(t) > 1 and t not in STOP}


def overlap(query, title):
    q = tokens(query)
    return len(q & tokens(title)) / max(1, len(q))


def root_category(x):
    return norm(str(x).split("/")[0]) or "unknown"


def text_item(row):
    parts = []
    for col, label, limit in [
        ("title", "başlık", 260), ("category", "kategori", 180), ("brand", "marka", 80),
        ("gender", "cinsiyet", 50), ("age_group", "yaş", 50), ("attributes", "özellik", 350),
    ]:
        value = str(row.get(col, "") or "").replace("\n", " ")[:limit].strip()
        if value and value.lower() != "nan":
            parts.append(f"{label}: {value}")
    return " | ".join(parts)


def fold_of(term_id):
    return int(hashlib.md5(str(term_id).encode()).hexdigest()[:8], 16) % 10


def build_clean_cross_root():
    cols = ["term_id", "item_id", "label", "source", "v33_retrieval_score", "query", "title", "category"]
    d = pd.read_parquet(V34, columns=cols)
    d["term_id"] = d.term_id.astype(str)
    d["item_id"] = d.item_id.astype(str)
    d["root"] = d.category.map(root_category)
    d["title_overlap"] = [overlap(q, t) for q, t in zip(d["query"], d["title"])]

    pos = d[d.label.eq(1)].copy()
    neg = d[d.label.eq(0)].copy()
    roots = pos.groupby("term_id").root.apply(set).to_dict()
    neg["cross_root"] = [r not in roots.get(t, set()) for t, r in zip(neg.term_id, neg.root)]
    neg = neg[neg.cross_root].copy()

    # Select a real positive with the strongest direct query/title evidence.
    best_pos = pos.sort_values(["term_id", "title_overlap"], ascending=[True, False]).drop_duplicates("term_id")
    best_pos = best_pos[["term_id", "item_id", "title_overlap"]].rename(
        columns={"item_id": "item_id_pos", "title_overlap": "pos_title_overlap"}
    )

    # Keep hard negatives first, but cap each query to prevent a few queries dominating.
    neg["v33_retrieval_score"] = pd.to_numeric(neg.v33_retrieval_score, errors="coerce").fillna(-99)
    neg = neg.sort_values(["term_id", "v33_retrieval_score"], ascending=[True, False]).groupby("term_id", sort=False).head(6)
    out = neg.merge(best_pos, on="term_id", how="inner", validate="many_to_one")
    out = out.rename(columns={"item_id": "item_id_neg", "title_overlap": "neg_title_overlap"})
    out["source_type"] = "clean_cross_root_v33_hard"
    out["confidence_weight"] = np.where(out.neg_title_overlap >= .5, 1.15, 1.0).astype(np.float32)
    return out[["term_id", "query", "item_id_pos", "item_id_neg", "source_type", "confidence_weight", "pos_title_overlap", "neg_title_overlap"]]


def build_manual_preferences():
    parts = []
    for source, path in LABELS:
        d = pd.read_csv(path)
        label = pd.to_numeric(d.assistant_swap_label, errors="coerce")
        recheck = pd.to_numeric(d.get("needs_recheck", 0), errors="coerce").fillna(1)
        conf = d.get("assistant_confidence", "").astype(str).str.lower()
        keep = label.isin([0, 1]) & recheck.eq(0) & conf.isin(["high", "medium"])
        d = d.loc[keep].copy()
        y = label.loc[keep].astype(int).to_numpy()
        d["item_id_pos"] = np.where(y == 1, d.item_id_add.astype(str), d.item_id_drop.astype(str))
        d["item_id_neg"] = np.where(y == 1, d.item_id_drop.astype(str), d.item_id_add.astype(str))
        d["source_type"] = "manual_preference_" + source
        d["confidence_weight"] = np.where(conf.loc[keep].eq("high"), 3.0, 2.0).astype(np.float32)
        d["pos_title_overlap"] = np.nan
        d["neg_title_overlap"] = np.nan
        parts.append(d[["term_id", "query", "item_id_pos", "item_id_neg", "source_type", "confidence_weight", "pos_title_overlap", "neg_title_overlap"]])
    return pd.concat(parts, ignore_index=True)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    clean = build_clean_cross_root()
    manual = build_manual_preferences()
    triplets = pd.concat([clean, manual], ignore_index=True)
    triplets["term_id"] = triplets.term_id.astype(str)
    triplets["item_id_pos"] = triplets.item_id_pos.astype(str)
    triplets["item_id_neg"] = triplets.item_id_neg.astype(str)
    triplets = triplets[triplets.item_id_pos.ne(triplets.item_id_neg)].drop_duplicates(["term_id", "item_id_pos", "item_id_neg"])

    needed = set(triplets.item_id_pos) | set(triplets.item_id_neg)
    items = pd.read_csv(ITEMS, low_memory=False)
    items["item_id"] = items.item_id.astype(str)
    small = items[items.item_id.isin(needed)].copy().drop_duplicates("item_id")
    small["item_text"] = small.apply(text_item, axis=1)
    text_map = small.set_index("item_id").item_text
    triplets["pos_text"] = triplets.item_id_pos.map(text_map)
    triplets["neg_text"] = triplets.item_id_neg.map(text_map)
    triplets["query_text"] = "sorgu: " + triplets["query"].fillna("").astype(str).str.slice(0, 180)
    triplets["fold"] = triplets.term_id.map(fold_of).astype(np.int8)
    triplets["is_manual"] = triplets.source_type.str.startswith("manual").astype(np.int8)
    triplets = triplets.dropna(subset=["pos_text", "neg_text", "query_text"]).reset_index(drop=True)
    triplets.to_parquet(OUT, index=False)

    sample = pd.concat([
        triplets[triplets.is_manual.eq(1)].sample(min(100, int(triplets.is_manual.sum())), random_state=42),
        triplets[triplets.is_manual.eq(0)].sample(min(100, int((triplets.is_manual.eq(0)).sum())), random_state=42),
    ]).sample(frac=1, random_state=42)
    sample.to_csv(SAMPLE, index=False)

    report = {
        "rows": int(len(triplets)),
        "unique_terms": int(triplets.term_id.nunique()),
        "unique_positive_items": int(triplets.item_id_pos.nunique()),
        "unique_negative_items": int(triplets.item_id_neg.nunique()),
        "source_counts": triplets.source_type.value_counts().to_dict(),
        "fold_counts": triplets.fold.value_counts().sort_index().to_dict(),
        "manual_rows": int(triplets.is_manual.sum()),
        "cross_root_rows": int((triplets.is_manual.eq(0)).sum()),
        "warning": "Cross-root negatives are high-confidence preferences, not proven hidden-test negatives. Manual-source holdout remains mandatory.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", OUT, SAMPLE, REPORT)


if __name__ == "__main__":
    main()
