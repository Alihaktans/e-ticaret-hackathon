"""Mine repeated intent-error hard negatives: brand, number, gender and category bait."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(".")
V34 = ROOT / "data/processed/v34/v34_hard_negative_pairs.parquet"
BASE = ROOT / "data/processed/v62_berturk_pairwise_triplets.parquet"
OUT = ROOT / "data/processed/v88_intent_pairwise_triplets.parquet"
REPORT = ROOT / "reports/experiments/v88_intent_hardnegative_report.json"
STOP = {"ve", "ile", "icin", "bir", "bu", "set", "takim", "adet", "model", "urun", "renk"}


def repair(x):
    x = "" if pd.isna(x) else str(x)
    try:
        return x.encode("latin1").decode("utf8")
    except (UnicodeError, UnicodeEncodeError):
        return x


def norm(x):
    x = repair(x).lower().replace("ı", "i")
    x = unicodedata.normalize("NFKD", x)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", x)).strip()


def toks(x):
    return {t for t in norm(x).split() if len(t) > 1 and t not in STOP}


def coverage(q, x):
    qt = toks(q)
    return len(qt & toks(x)) / max(1, len(qt))


def digits(x):
    return set(re.findall(r"\d+", norm(x)))


def phrase_in(text, phrase):
    return bool(phrase) and f" {phrase} " in f" {text} "


def main():
    cols = ["term_id", "item_id", "label", "v33_retrieval_score", "query", "title",
            "category", "brand", "gender", "item_text"]
    d = pd.read_parquet(V34, columns=cols)
    d["term_id"] = d["term_id"].astype(str); d["item_id"] = d["item_id"].astype(str)
    for c in ["query", "title", "category", "brand", "gender"]:
        d[c + "_n"] = d[c].map(norm)
    d["title_cov"] = [coverage(q, x) for q, x in zip(d["query"], d["title"])]
    d["cat_cov"] = [coverage(q, x) for q, x in zip(d["query"], d["category"])]
    d["root"] = d.category_n.str.split().str[0].fillna("")
    d["leaf"] = d.category_n
    d["q_digits"] = d["query"].map(digits); d["item_digits"] = (d["title"].fillna("") + " " + d["item_text"].fillna("")).map(digits)

    # Brands occurring often enough are safe query constraints, not incidental tokens.
    brand_counts = d[["item_id", "brand_n"]].drop_duplicates().brand_n.value_counts()
    brands = sorted([b for b, n in brand_counts.items() if n >= 20 and len(b) >= 3], key=len, reverse=True)
    query_brand = {}
    for q in d.query_n.drop_duplicates():
        query_brand[q] = next((b for b in brands if phrase_in(q, b)), "")
    d["query_brand"] = d.query_n.map(query_brand)

    pos = d[d.label.eq(1)].copy(); neg = d[d.label.eq(0)].copy()
    pos_roots = pos.groupby("term_id").root.apply(set).to_dict()
    pos_leaves = pos.groupby("term_id").leaf.apply(set).to_dict()
    pos_best = pos.sort_values(["term_id", "title_cov"], ascending=[True, False]).drop_duplicates("term_id").set_index("term_id")

    rows = []
    genders = {"erkek": {"kadin", "bayan", "kiz"}, "kadin": {"erkek", "bay"},
               "bebek": {"erkek", "kadin"}, "cocuk": {"erkek", "kadin"}}
    for term_id, g in neg.groupby("term_id", sort=False):
        if term_id not in pos_best.index:
            continue
        p = pos_best.loc[term_id]
        qtok = toks(p["query"]); head = list(norm(p["query"]).split())[-1] if norm(p["query"]) else ""
        qb = p.query_brand
        qd = p.q_digits
        candidates = []
        for n in g.itertuples(index=False):
            same_root = n.root in pos_roots.get(term_id, set())
            types = []
            if n.cat_cov >= .50 and n.title_cov <= .25 and p.title_cov >= n.title_cov + .25:
                types.append(("category_bait", .90))
            if qb and not phrase_in(n.brand_n, qb) and (phrase_in(p.brand_n, qb) or phrase_in(p.title_n, qb)):
                types.append(("brand_mismatch", 1.15))
            if qd and not (qd & n.item_digits) and (qd & p.item_digits):
                types.append(("numeric_mismatch", 1.15))
            qwords = set(norm(p["query"]).split())
            for wanted, wrong in genders.items():
                if wanted in qwords and wrong & set((n.title_n + " " + n.gender_n).split()):
                    types.append(("gender_mismatch", 1.05)); break
            if same_root and head and head in toks(p.title) and head not in toks(n.title) and p.title_cov >= n.title_cov + .20:
                types.append(("head_noun_mismatch", .95))
            if not types:
                continue
            priority = float(n.v33_retrieval_score) if pd.notna(n.v33_retrieval_score) else -1
            for typ, weight in types:
                candidates.append((typ, weight, priority, n))

        for typ in {x[0] for x in candidates}:
            chosen = sorted((x for x in candidates if x[0] == typ), key=lambda x: x[2], reverse=True)[:3]
            for _, weight, _, n in chosen:
                rows.append({
                    "term_id": term_id, "query": repair(p["query"]), "item_id_pos": p.item_id,
                    "item_id_neg": n.item_id, "source_type": "intent_" + typ,
                    "confidence_weight": np.float32(weight), "pos_title_overlap": np.float32(p.title_cov),
                    "neg_title_overlap": np.float32(n.title_cov), "pos_text": repair(p.item_text),
                    "neg_text": repair(n.item_text), "query_text": "sorgu: " + repair(p["query"])[:180],
                    "fold": np.int8(int(hashlib.md5(term_id.encode()).hexdigest()[:8], 16) % 10),
                    "is_manual": np.int8(0),
                })

    target = pd.DataFrame(rows).drop_duplicates(["term_id", "item_id_pos", "item_id_neg"])
    base = pd.read_parquet(BASE)
    out = pd.concat([base, target], ignore_index=True).drop_duplicates(["term_id", "item_id_pos", "item_id_neg", "source_type"])
    OUT.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(OUT, index=False)
    report = {"base_rows": len(base), "targeted_rows": len(target), "total_rows": len(out),
              "targeted_terms": int(target.term_id.nunique()),
              "targeted_sources": target.source_type.value_counts().to_dict()}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
