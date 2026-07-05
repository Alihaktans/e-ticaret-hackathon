"""Build leakage-reduced lexical features for the v34 hard-negative pool."""
from pathlib import Path
import runpy
import time

import pandas as pd


ROOT = Path(".")
V21 = runpy.run_path(str(ROOT / "scripts/95_build_v21_feature_tables.py"))
OUT = ROOT / "data/processed/v82_v34_features.parquet"


def main() -> None:
    t0 = time.time()
    train_pairs = pd.read_csv(ROOT / "data/raw/training_pairs.csv")
    train_pairs["term_id"] = train_pairs["term_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)
    if "label" in train_pairs:
        train_pairs = train_pairs[train_pairs["label"].astype(int) == 1].copy()

    item_pos = train_pairs["item_id"].value_counts().rename("item_train_pos_count").reset_index()
    item_pos.columns = ["item_id", "item_train_pos_count"]
    item_pos["item_id"] = item_pos["item_id"].astype(str)

    terms = pd.read_csv(ROOT / "data/raw/terms.csv")
    terms["term_id"] = terms["term_id"].astype(str)
    terms["query_norm"] = terms["query"].map(V21["norm"])

    item_cols = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    items = pd.read_csv(ROOT / "data/raw/items.csv", usecols=lambda c: c in item_cols, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)
    for c in item_cols:
        if c not in items:
            items[c] = ""

    items["title_norm"] = items["title"].map(V21["norm"])
    items["category_norm"] = items["category"].map(V21["norm"])
    items["brand_norm"] = items["brand"].map(V21["norm"])
    items["gender_norm"] = items["gender"].map(V21["norm"])
    items["age_group_norm"] = items["age_group"].map(V21["norm"])
    items["attr_norm"] = items["attributes"].fillna("").astype(str).str.slice(0, 600).map(V21["norm"])
    items["root_key"] = items["category"].map(V21["root_category"]).map(V21["norm"])
    items["brand_key"] = items["brand_norm"]
    items["category_key"] = items["category_norm"].str.slice(0, 120)
    items = items.merge(item_pos, on="item_id", how="left")
    items["item_train_pos_count"] = items["item_train_pos_count"].fillna(0).astype("int32")

    brand_pos = items.groupby("brand_norm")["item_train_pos_count"].sum().rename("brand_train_pos_count").reset_index()
    root_pos = items.groupby("root_key")["item_train_pos_count"].sum().rename("root_train_pos_count").reset_index()
    items = items.merge(brand_pos, on="brand_norm", how="left").merge(root_pos, on="root_key", how="left")
    items["brand_train_pos_count"] = items["brand_train_pos_count"].fillna(0).astype("int32")
    items["root_train_pos_count"] = items["root_train_pos_count"].fillna(0).astype("int32")

    brand_map = V21["build_brand_detector"](items)
    terms["query_brand_detected"] = terms["query_norm"].map(lambda x: V21["detect_query_brand"](x, brand_map))
    terms_feat = terms[["term_id", "query_norm", "query_brand_detected"]]
    items_feat = items[[
        "item_id", "title_norm", "category_norm", "brand_norm", "gender_norm",
        "age_group_norm", "attr_norm", "root_key", "brand_key", "category_key",
        "item_train_pos_count", "brand_train_pos_count", "root_train_pos_count",
    ]]

    candidates = pd.read_parquet(ROOT / "data/processed/v34/v34_hard_negative_pairs.parquet",
                                 columns=["term_id", "item_id", "label", "source"])
    candidates = candidates.rename(columns={"source": "candidate_source"})
    candidates["term_id"] = candidates["term_id"].astype(str)
    candidates["item_id"] = candidates["item_id"].astype(str)
    features = V21["build_one"](candidates, "v82-v34", terms_feat, items_feat)
    features.to_parquet(OUT, index=False)
    print({"out": str(OUT), "shape": features.shape, "minutes": (time.time() - t0) / 60})


if __name__ == "__main__":
    main()
