from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(".")

REVIEW = ROOT / "reports/manual_review/review_v43_pop_brand_category_swaps.csv"
ITEMS = ROOT / "data/raw/items.csv"
TERMS = ROOT / "data/raw/terms.csv"

OUT_ALL = ROOT / "reports/manual_review/review_v43_popbc_enriched_all.csv"
OUT_TOP = ROOT / "reports/manual_review/review_v43_popbc_enriched_top_v43_7000.csv"
OUT_RISK = ROOT / "reports/manual_review/review_v43_popbc_enriched_risky_v43_7000.csv"
OUT_GOOD = ROOT / "reports/manual_review/review_v43_popbc_enriched_good_v43_7000.csv"

TARGET_VARIANT = "v43_v41_meta_same_quota_v43_meta_prob_impact_cap7000"

TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "bir", "adet", "set", "takim", "model", "uyumlu",
    "orjinal", "orijinal", "yeni", "renk", "boy", "numara", "beden",
    "cm", "mm", "lt", "kg", "gr", "ml", "x", "no", "olan", "veya",
    "plus", "pro", "max", "mini", "urun", "da", "de", "mi", "mı", "mu", "mü",
}


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def toks(x):
    return [t for t in norm_text(x).split() if len(t) >= 2 and t not in STOP]


def detect_col(cols, exacts=(), contains=()):
    lower = {c.lower(): c for c in cols}
    for e in exacts:
        if e.lower() in lower:
            return lower[e.lower()]
    for pat in contains:
        for c in cols:
            if pat.lower() in c.lower():
                return c
    return None


def overlap_ratio(query, text):
    q = toks(query)
    if not q:
        return 0.0
    s = set(toks(text))
    return len([t for t in q if t in s]) / len(q)


def any_brand_in_query(brand, query):
    brand = norm_text(brand)
    query = norm_text(query)
    return int(bool(brand) and brand in query)


def number_tokens(x):
    return set(re.findall(r"\b\d+[a-z]?\b", norm_text(x)))


def number_coverage(query, text):
    qn = number_tokens(query)
    if not qn:
        return 1.0
    tn = number_tokens(text)
    return len(qn & tn) / len(qn)


def special_tokens(x):
    out = []
    for t in toks(x):
        if re.search(r"\d", t) or len(t) >= 7:
            out.append(t)
    return set(out)


def special_coverage(query, text):
    q = special_tokens(query)
    if not q:
        return 1.0
    tt = set(toks(text))
    return len(q & tt) / len(q)


def load_items():
    items = pd.read_csv(ITEMS)
    cols = list(items.columns)

    item_col = detect_col(cols, exacts=("item_id", "product_id", "id"), contains=("item_id",))
    title_col = detect_col(cols, exacts=("title", "name", "product_name", "item_name", "urun_adi", "ürün_adı"), contains=("title", "name", "urun", "ürün"))
    brand_col = detect_col(cols, exacts=("brand", "marka"), contains=("brand", "marka"))
    cat_col = detect_col(cols, exacts=("category", "kategori", "category_name", "leaf_category", "cat"), contains=("category", "kategori", "cat"))

    if item_col is None:
        raise RuntimeError(f"item id column not found; columns={cols}")

    out = pd.DataFrame({"item_id": items[item_col].astype(str)})
    out["title"] = items[title_col].astype(str) if title_col else ""
    out["brand"] = items[brand_col].astype(str) if brand_col else ""
    out["category"] = items[cat_col].astype(str) if cat_col else ""
    return out


def load_terms():
    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    if "query" not in terms.columns:
        cand = [c for c in terms.columns if c != "term_id" and terms[c].dtype == "object"]
        if not cand:
            raise RuntimeError(f"query column not found; columns={list(terms.columns)}")
        terms = terms.rename(columns={cand[0]: "query"})
    return terms[["term_id", "query"]].copy()


def main():
    if not REVIEW.exists():
        raise FileNotFoundError(REVIEW)

    review = pd.read_csv(REVIEW)
    review["term_id"] = review["term_id"].astype(str)
    review["item_id_add"] = review["item_id_add"].astype(str)
    review["item_id_drop"] = review["item_id_drop"].astype(str)

    items = load_items()
    terms = load_terms()

    add = items.rename(columns={
        "item_id": "item_id_add",
        "title": "add_title",
        "brand": "add_brand",
        "category": "add_category",
    })

    drop = items.rename(columns={
        "item_id": "item_id_drop",
        "title": "drop_title",
        "brand": "drop_brand",
        "category": "drop_category",
    })

    df = review.merge(terms, on="term_id", how="left")
    df = df.merge(add, on="item_id_add", how="left")
    df = df.merge(drop, on="item_id_drop", how="left")

    for c in ["query", "add_title", "add_brand", "add_category", "drop_title", "drop_brand", "drop_category"]:
        df[c] = df[c].fillna("").astype(str)

    df["add_title_overlap"] = [overlap_ratio(q, t) for q, t in zip(df["query"], df["add_title"])]
    df["drop_title_overlap"] = [overlap_ratio(q, t) for q, t in zip(df["query"], df["drop_title"])]
    df["add_category_overlap"] = [overlap_ratio(q, t) for q, t in zip(df["query"], df["add_category"])]
    df["drop_category_overlap"] = [overlap_ratio(q, t) for q, t in zip(df["query"], df["drop_category"])]
    df["add_brand_in_query"] = [any_brand_in_query(b, q) for b, q in zip(df["add_brand"], df["query"])]
    df["drop_brand_in_query"] = [any_brand_in_query(b, q) for b, q in zip(df["drop_brand"], df["query"])]
    df["add_number_cov"] = [number_coverage(q, t + " " + b + " " + c) for q, t, b, c in zip(df["query"], df["add_title"], df["add_brand"], df["add_category"])]
    df["drop_number_cov"] = [number_coverage(q, t + " " + b + " " + c) for q, t, b, c in zip(df["query"], df["drop_title"], df["drop_brand"], df["drop_category"])]
    df["add_special_cov"] = [special_coverage(q, t + " " + b + " " + c) for q, t, b, c in zip(df["query"], df["add_title"], df["add_brand"], df["add_category"])]
    df["drop_special_cov"] = [special_coverage(q, t + " " + b + " " + c) for q, t, b, c in zip(df["query"], df["drop_title"], df["drop_brand"], df["drop_category"])]

    df["title_overlap_gain"] = df["add_title_overlap"] - df["drop_title_overlap"]
    df["category_overlap_gain"] = df["add_category_overlap"] - df["drop_category_overlap"]
    df["number_cov_gain"] = df["add_number_cov"] - df["drop_number_cov"]
    df["special_cov_gain"] = df["add_special_cov"] - df["drop_special_cov"]

    # Risk flags for V44 rule design.
    df["risk_drop_title_better"] = ((df["drop_title_overlap"] >= df["add_title_overlap"] + 0.20) & (df["drop_title_overlap"] >= 0.25)).astype(int)
    df["risk_add_category_only"] = ((df["add_title_overlap"] <= 0.12) & (df["add_category_overlap"] >= 0.30)).astype(int)
    df["risk_number_loss"] = ((df["drop_number_cov"] > df["add_number_cov"]) & (df["drop_number_cov"] >= 0.99)).astype(int)
    df["risk_special_loss"] = ((df["drop_special_cov"] > df["add_special_cov"]) & (df["drop_special_cov"] >= 0.99)).astype(int)
    df["risk_brand_loss"] = ((df["drop_brand_in_query"] > df["add_brand_in_query"])).astype(int)

    df["v44_risk_score"] = (
        2.0 * df["risk_drop_title_better"]
        + 1.5 * df["risk_add_category_only"]
        + 1.5 * df["risk_number_loss"]
        + 1.2 * df["risk_special_loss"]
        + 1.0 * df["risk_brand_loss"]
    )

    df["v44_good_score"] = (
        2.0 * (df["add_title_overlap"] >= df["drop_title_overlap"] + 0.20).astype(int)
        + 1.2 * (df["add_title_overlap"] >= 0.30).astype(int)
        + 1.0 * (df["add_brand_in_query"] >= df["drop_brand_in_query"]).astype(int)
        + 1.0 * (df["add_number_cov"] >= df["drop_number_cov"]).astype(int)
        + 1.0 * (df["add_special_cov"] >= df["drop_special_cov"]).astype(int)
    )

    # Useful column order for manual reading.
    front = [
        "variant", "term_id", "pair_rank", "query",
        "id_add", "item_id_add", "add_title", "add_brand", "add_category",
        "id_drop", "item_id_drop", "drop_title", "drop_brand", "drop_category",
        "v43_swap_support_score", "prob_gain", "score_gain",
        "add_title_overlap", "drop_title_overlap", "title_overlap_gain",
        "add_category_overlap", "drop_category_overlap", "category_overlap_gain",
        "add_brand_in_query", "drop_brand_in_query",
        "add_number_cov", "drop_number_cov", "number_cov_gain",
        "add_special_cov", "drop_special_cov", "special_cov_gain",
        "v44_risk_score", "v44_good_score",
        "risk_drop_title_better", "risk_add_category_only", "risk_number_loss", "risk_special_loss", "risk_brand_loss",
    ]

    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    df.to_csv(OUT_ALL, index=False)

    target = df[df["variant"].eq(TARGET_VARIANT)].copy()
    if len(target):
        # Top by original v43 confidence, now enriched.
        top = target.sort_values(["v43_swap_support_score", "prob_gain"], ascending=False).head(300)
        top.to_csv(OUT_TOP, index=False)

        # Risky examples: likely to reveal traps.
        risk = target.sort_values(["v44_risk_score", "v43_swap_support_score"], ascending=[False, False]).head(300)
        risk.to_csv(OUT_RISK, index=False)

        # Good examples: likely clean swaps.
        good = target.sort_values(["v44_good_score", "v43_swap_support_score"], ascending=[False, False]).head(300)
        good.to_csv(OUT_GOOD, index=False)

    print("rows:", len(df))
    print("target_rows:", len(target))
    print("risk_rate_target:", float((target["v44_risk_score"] > 0).mean()) if len(target) else None)
    print("good_mean_target:", float(target["v44_good_score"].mean()) if len(target) else None)
    print("wrote:")
    print(OUT_ALL)
    print(OUT_TOP)
    print(OUT_RISK)
    print(OUT_GOOD)

    if len(target):
        print("\nTarget risk flags:")
        print(target[[
            "risk_drop_title_better", "risk_add_category_only", "risk_number_loss", "risk_special_loss", "risk_brand_loss"
        ]].mean().to_string())


if __name__ == "__main__":
    main()
