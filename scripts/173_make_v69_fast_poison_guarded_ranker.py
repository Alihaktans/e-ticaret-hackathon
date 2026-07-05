from pathlib import Path
import re
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V68_FEATURES = ROOT / "data/processed/v68_independent_query_rank_features.parquet"
V67_FEATURES = ROOT / "data/processed/v67_global_aggressive_blend_scores.parquet"

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v69_fast"

OUT_FEATURES = ROOT / "data/processed/v69_fast_poison_guarded_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v69_fast_poison_guarded_candidate_summary.csv"
OUT_AUDIT = OUT_DIR / "review_v69_fast_poison_guarded_audit.csv"
OUT_BAD = OUT_DIR / "v69_fast_auto_bad_examples.csv"

REFERENCE_FILES = [
    ROOT / "submissions/final_candidates_v68/FINAL_CANDIDATE_v68_IND_GLOBAL_ratio0p34_ff3bda60.csv",
    ROOT / "submissions/final_candidates_v68/FINAL_CANDIDATE_v68_IND_GLOBAL_ratio0p31684_cdd371b9.csv",
    ROOT / "submissions/final_candidates_v66/FINAL_CANDIDATE_v66_intent_ultra_swap_cap75_699710c5.csv",
]

GLOBAL_RATIOS = [0.30, 0.31684, 0.32, 0.34, 0.36, 0.37, 0.40]
QUERY_RATIOS = [0.26, 0.30, 0.34, 0.38]
CONSENSUS = [
    ("g58_t45_top1", 0.58, 0.45, 1),
    ("g60_t50_top1", 0.60, 0.50, 1),
    ("g63_t50_top1", 0.63, 0.50, 1),
    ("g65_t55_top1", 0.65, 0.55, 1),
]
HYBRID = [
    ("union_g32_q24", 0.32, 0.24, "union"),
    ("union_g34_q26", 0.34, 0.26, "union"),
    ("inter_g37_q34", 0.37, 0.34, "inter"),
    ("inter_g40_q38", 0.40, 0.38, "inter"),
]

TR_TRANS = str.maketrans({
    "ı": "i", "İ": "i", "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
    "ş": "s", "Ş": "s", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]

def clean_series(s):
    return (
        s.fillna("")
         .astype(str)
         .str.lower()
         .str.translate(TR_TRANS)
         .str.replace(r"[^a-z0-9%./+ -]+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

def contains_word(s, pat):
    return s.str.contains(rf"(^|[^a-z0-9])(?:{pat})([^a-z0-9]|$)", regex=True, na=False)

def contains_any(s, words):
    pats = []
    for w in words:
        w = w.lower().translate(TR_TRANS)
        pats.append(re.escape(w))
    return contains_word(s, "|".join(pats))

def load_base_features():
    src = V68_FEATURES if V68_FEATURES.exists() else V67_FEATURES
    if not src.exists():
        raise FileNotFoundError("V68/V67 feature file yok. Önce 171 veya 170 çalışmalı.")
    print("Loading features:", src, flush=True)
    df = pd.read_parquet(src)
    for c in ["id", "term_id", "item_id"]:
        df[c] = df[c].astype(str)

    # If V68 fields are missing, create fallback from V67.
    if "v68_ind_score" not in df.columns:
        print("Creating fallback independent score from V67", flush=True)
        df["global_pct"] = pd.Series(df["v67_final_score"]).rank(pct=True)
        df["term_rank"] = df.groupby("term_id")["v67_final_score"].rank(method="first", ascending=False).astype(np.int32)
        df["term_n"] = df.groupby("term_id")["id"].transform("size").astype(np.int32)
        df["term_top_pct"] = 1.0 - ((df["term_rank"] - 1) / np.maximum(1, df["term_n"] - 1))
        g = df.groupby("term_id")["v67_final_score"]
        df["term_z"] = ((df["v67_final_score"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0)
        df["v68_ind_score"] = (
            0.46 * df["global_pct"] + 0.31 * df["term_top_pct"] +
            0.15 * (1 / (1 + np.exp(-np.clip(df["term_z"], -20, 20)))) +
            0.08 * df["v67_final_score"]
        )
    return df

def join_text(df):
    print("Joining terms/items text", flush=True)

    terms = pd.read_csv(TERMS)
    term_text_cols = [c for c in terms.columns if c != "term_id" and re.search(r"(query|term|text|name|arama)", c, re.I)]
    if "term_id" not in terms.columns or not term_text_cols:
        raise RuntimeError("terms.csv içinde term_id + query/term text kolonu bulunamadı.")
    terms = terms[["term_id", term_text_cols[0]]].rename(columns={term_text_cols[0]: "query_text"})
    terms["term_id"] = terms["term_id"].astype(str)
    df = df.merge(terms, on="term_id", how="left")

    items = pd.read_csv(ITEMS)
    title_cols = [c for c in items.columns if re.search(r"(title|name|urun|ürün|product)", c, re.I)]
    brand_cols = [c for c in items.columns if re.search(r"(brand|marka)", c, re.I)]
    cat_cols = [c for c in items.columns if re.search(r"(category|kategori|cat)", c, re.I)]

    use = ["item_id"]
    ren = {}
    if title_cols:
        use.append(title_cols[0]); ren[title_cols[0]] = "item_title"
    if brand_cols:
        use.append(brand_cols[0]); ren[brand_cols[0]] = "item_brand"
    if cat_cols:
        use.append(cat_cols[0]); ren[cat_cols[0]] = "item_category"
    items = items[use].rename(columns=ren)
    items["item_id"] = items["item_id"].astype(str)
    df = df.merge(items, on="item_id", how="left")
    return df

def apply_fast_poison(df):
    print("Normalizing text columns", flush=True)
    q = clean_series(df["query_text"])
    title = clean_series(df.get("item_title", pd.Series("", index=df.index)))
    brand = clean_series(df.get("item_brand", pd.Series("", index=df.index)))
    cat = clean_series(df.get("item_category", pd.Series("", index=df.index)))
    item = (title + " " + brand + " " + cat).str.replace(r"\s+", " ", regex=True)

    n = len(df)
    penalty = np.zeros(n, dtype=np.float32)
    reason = np.full(n, "", dtype=object)

    def add(mask, val, r):
        nonlocal penalty, reason
        m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        penalty[m] += val
        empty = (reason == "") & m
        reason[empty] = r
        both = (reason != "") & m & (~empty)
        reason[both] = np.char.add(np.char.add(reason[both].astype(str), "|"), r)

    print("Applying vectorized poison rules", flush=True)

    male = ["erkek", "bay", "mens", "men"]
    female = ["kadin", "bayan", "women", "woman", "girl", "girls"]
    kid = ["cocuk", "kids", "kid", "jr", "junior", "bebek", "baby", "kiz"]
    accessory = ["kilif", "case", "kapak", "aksesuar", "aparat", "yedek", "parca", "askisi", "kayisi", "kablo", "adapter", "adaptör", "sarj", "şarj", "stand"]
    main_elec = ["iphone", "ayfon", "telefon", "tablet", "laptop", "notebook", "bilgisayar", "televizyon", "tv", "playstation", "ps4"]

    q_male = contains_any(q, male)
    q_female = contains_any(q, female)
    i_male = contains_any(item, male)
    i_female = contains_any(item, female)
    q_kid = contains_any(q, kid)
    i_kid = contains_any(item, kid)
    i_accessory = contains_any(item, accessory)
    q_accessory = contains_any(q, accessory)
    q_main_elec = contains_any(q, main_elec) & (~q_accessory)

    add(q_male & i_female & (~i_male), 0.42, "gender_male_item_female")
    add(q_female & i_male & (~i_female), 0.42, "gender_female_item_male")
    add(q_kid & i_male & (~i_kid), 0.24, "child_query_adult_male_item")
    add(q_main_elec & i_accessory, 0.55, "main_product_query_accessory_item")

    # Color mismatch. Only common colors from audit.
    color_map = {
        "siyah": ["siyah", "black"],
        "beyaz": ["beyaz", "white"],
        "kirmizi": ["kirmizi", "red"],
        "mavi": ["mavi", "blue"],
        "yesil": ["yesil", "green"],
        "sari": ["sari", "yellow"],
        "pembe": ["pembe", "pink"],
        "gri": ["gri", "gray", "grey"],
        "lacivert": ["lacivert"],
        "kahverengi": ["kahverengi"],
        "bej": ["bej"],
        "krem": ["krem"],
        "taba": ["taba"],
        "vizon": ["vizon"],
        "gold": ["gold", "altin"],
        "gumus": ["gumus", "silver"],
    }
    q_color_any = pd.Series(False, index=df.index)
    i_color_any = pd.Series(False, index=df.index)
    same_color = pd.Series(False, index=df.index)
    for cname, vals in color_map.items():
        q_c = contains_any(q, vals)
        i_c = contains_any(item, vals)
        q_color_any |= q_c
        i_color_any |= i_c
        same_color |= (q_c & i_c)
    add(q_color_any & i_color_any & (~same_color), 0.18, "color_mismatch")

    # Brand missing from item.
    brands = [
        "nike", "adidas", "puma", "skechers", "lacoste", "salomon", "slazenger",
        "pierre cardin", "karaca", "loreal", "l'oreal", "la roche", "new balance",
        "u.s. polo", "us polo", "polo assn", "tamer tanca", "hotic", "hotiç",
        "scooter", "tommy", "jack jones", "jack & jones", "gillette", "oral b",
        "apple", "samsung", "xiaomi", "crocs", "ray ban", "ray-ban",
    ]
    for b in brands:
        bnorm = b.lower().translate(TR_TRANS)
        q_b = q.str.contains(re.escape(bnorm), regex=True, na=False)
        # loose item check: remove punctuation tokens
        loose = re.escape(bnorm.replace(".", "").replace("&", " ").replace("'", ""))
        i_b = item.str.contains(re.escape(bnorm), regex=True, na=False) | item.str.contains(loose, regex=True, na=False)
        add(q_b & (~i_b), 0.22, "brand_missing")

    # Known audit poison rules from V68.
    known_rules = [
        ("cocuk deodorant", ["men", "erkek"], 0.60, "known_child_deodorant_men"),
        ("scooter kadin", ["erkek", "men"], 0.60, "known_scooter_woman_men"),
        ("kadin esofman", ["erkek", "men"], 0.60, "known_women_tracksuit_men"),
        ("salomon erkek", ["kadin", "women", " w "], 0.55, "known_salomon_men_women"),
        ("lacoste kadin", ["erkek", "men"], 0.55, "known_lacoste_women_men"),
        ("paten kask", ["paten"], 0.45, "known_skate_helmet_item_skate"),
        ("kucuk dikis makinesi", ["ayak", "aksesuar", "parca"], 0.50, "known_sewing_machine_accessory"),
        ("galatasaray termos", ["stanley"], 0.35, "known_gs_thermos_missing_gs"),
        ("mavi omuz cantasi", ["siyah"], 0.35, "known_blue_bag_black"),
    ]
    for qpat, iw, val, r in known_rules:
        qm = q.str.contains(re.escape(qpat), regex=True, na=False)
        im = contains_any(item, iw)
        add(qm & im, val, r)

    penalty = np.minimum(penalty, 0.95)
    df["v69_poison_penalty"] = penalty
    df["v69_poison_reason"] = reason

    df["v69_guarded_score"] = (
        df["v68_ind_score"].astype(float)
        - 0.60 * df["v69_poison_penalty"].astype(float)
        + 0.05 * df["global_pct"].astype(float)
        + 0.03 * df["term_top_pct"].astype(float)
    )

    print("Re-ranking guarded scores", flush=True)
    df["v69_global_pct"] = pd.Series(df["v69_guarded_score"]).rank(pct=True)
    df["v69_term_rank"] = df.groupby("term_id")["v69_guarded_score"].rank(method="first", ascending=False).astype(np.int32)
    df["v69_term_n"] = df.groupby("term_id")["id"].transform("size").astype(np.int32)
    df["v69_term_top_pct"] = 1.0 - ((df["v69_term_rank"] - 1) / np.maximum(1, df["v69_term_n"] - 1))
    return df

def find_reference(sample):
    for p in REFERENCE_FILES:
        if p.exists():
            b = pd.read_csv(p)
            b["id"] = b["id"].astype(str)
            if b["id"].equals(sample["id"]):
                return p, b["prediction"].astype(np.int8).to_numpy()
    return None, None

def save_variant(name, pred, sample_ids, df, ref_pred, rows, audits, family, meta):
    file = SUB_DIR / f"FINAL_CANDIDATE_{name}_{short_hash(name)}.csv"
    pd.DataFrame({"id": sample_ids, "prediction": pred.astype(np.int8)}).to_csv(file, index=False)

    score = df["v69_guarded_score"].to_numpy()
    pen = df["v69_poison_penalty"].to_numpy()

    row = {
        "variant": name,
        "file": str(file),
        "family": family,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "score_mean_pred1": float(score[pred == 1].mean()) if pred.sum() else np.nan,
        "score_mean_pred0": float(score[pred == 0].mean()) if (pred == 0).sum() else np.nan,
        "poison_rate_pred1": float((pen[pred == 1] > 0).mean()) if pred.sum() else np.nan,
        "poison_penalty_mean_pred1": float(pen[pred == 1].mean()) if pred.sum() else np.nan,
        **meta,
    }
    if ref_pred is not None:
        row["diff_vs_reference"] = int((pred != ref_pred).sum())
    rows.append(row)
    print("saved", file, row, flush=True)

    if ref_pred is not None:
        add = np.where((pred == 1) & (ref_pred == 0))[0]
        drop = np.where((pred == 0) & (ref_pred == 1))[0]
        if len(add):
            x = df.iloc[add].sort_values("v69_guarded_score", ascending=False).head(160).copy()
            x["variant"] = name
            x["action"] = "add_vs_reference"
            audits.append(x)
        if len(drop):
            x = df.iloc[drop].sort_values("v69_guarded_score", ascending=True).head(160).copy()
            x["variant"] = name
            x["action"] = "drop_vs_reference"
            audits.append(x)

def main():
    import sys
    force = "--force" in set(sys.argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    sample_ids = sample["id"].to_numpy()

    if OUT_FEATURES.exists() and not force:
        print("Using existing", OUT_FEATURES, flush=True)
        df = pd.read_parquet(OUT_FEATURES)
    else:
        df = load_base_features()
        df = join_text(df)
        df = apply_fast_poison(df)

        keep = [
            "id", "term_id", "item_id",
            "query_text", "item_title", "item_brand", "item_category",
            "v68_ind_score", "v69_guarded_score", "v69_poison_penalty", "v69_poison_reason",
            "global_pct", "term_rank", "term_n", "term_top_pct", "term_z",
            "v69_global_pct", "v69_term_rank", "v69_term_n", "v69_term_top_pct",
        ]
        keep = [c for c in keep if c in df.columns]
        df[keep].to_parquet(OUT_FEATURES, index=False)
        bad = df[df["v69_poison_penalty"] > 0].sort_values("v69_poison_penalty", ascending=False)
        bad.head(8000).to_csv(OUT_BAD, index=False)
        print("saved", OUT_FEATURES, flush=True)
        print("saved", OUT_BAD, "bad_rows", len(bad), flush=True)
        df = df[keep]

    df["id"] = df["id"].astype(str)
    if not df["id"].equals(sample["id"]):
        print("Aligning to sample order", flush=True)
        df = sample.merge(df, on="id", how="left")
        for c in ["v69_guarded_score", "v69_poison_penalty", "v69_global_pct", "v69_term_top_pct"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
        for c in ["v69_term_rank", "v69_term_n"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(999999).astype(np.int32)

    ref_path, ref_pred = find_reference(sample)
    print("Reference for diff only:", ref_path, flush=True)

    n = len(df)
    score = df["v69_guarded_score"].to_numpy()
    order = np.argsort(-score, kind="mergesort")

    rows = []
    audits = []

    for r in GLOBAL_RATIOS:
        k = int(round(n * r))
        pred = np.zeros(n, dtype=np.int8)
        pred[order[:k]] = 1
        save_variant(
            f"v69_FAST_GLOBAL_ratio{str(r).replace('.', 'p')}",
            pred, sample_ids, df, ref_pred, rows, audits,
            "fast_guarded_global_topk",
            {"target_ratio": float(r), "k": int(k)}
        )

    tr = df["v69_term_rank"].astype(int).to_numpy()
    tn = df["v69_term_n"].astype(int).to_numpy()
    g = df["v69_global_pct"].to_numpy()
    tt = df["v69_term_top_pct"].to_numpy()

    for r in QUERY_RATIOS:
        keep = np.maximum(1, np.ceil(tn * r).astype(int))
        pred = (tr <= keep).astype(np.int8)
        save_variant(
            f"v69_FAST_QUERY_ratio{str(r).replace('.', 'p')}",
            pred, sample_ids, df, ref_pred, rows, audits,
            "fast_guarded_query_ratio",
            {"query_ratio": float(r)}
        )

    for name, gmin, tmin, top_always in CONSENSUS:
        pred = (((g >= gmin) & (tt >= tmin)) | (tr <= top_always)).astype(np.int8)
        save_variant(
            f"v69_FAST_CONSENSUS_{name}",
            pred, sample_ids, df, ref_pred, rows, audits,
            "fast_guarded_consensus",
            {"global_pct_min": float(gmin), "term_top_pct_min": float(tmin), "top_rank_always": int(top_always)}
        )

    for name, gr, qr, mode in HYBRID:
        k = int(round(n * gr))
        gm = np.zeros(n, dtype=bool)
        gm[order[:k]] = True
        qkeep = np.maximum(1, np.ceil(tn * qr).astype(int))
        qm = tr <= qkeep
        if mode == "union":
            pred = (gm | qm).astype(np.int8)
        else:
            pred = (gm & qm).astype(np.int8)
            pred = np.maximum(pred, (tr <= 1).astype(np.int8))
        save_variant(
            f"v69_FAST_HYBRID_{name}",
            pred, sample_ids, df, ref_pred, rows, audits,
            "fast_guarded_hybrid",
            {"global_ratio": float(gr), "query_ratio": float(qr), "hybrid_mode": mode}
        )

    sm = pd.DataFrame(rows)
    sm["v69_fast_diagnostic_score"] = (
        0.33 * sm["score_mean_pred1"]
        - 0.12 * sm["score_mean_pred0"]
        - 0.18 * sm["poison_penalty_mean_pred1"]
        - 0.06 * sm["poison_rate_pred1"]
        - 0.16 * (sm["pos_ratio"] - 0.34).abs()
    )
    if "diff_vs_reference" in sm.columns:
        sm["v69_fast_diagnostic_score"] += 0.10 * np.minimum(1.0, np.log1p(sm["diff_vs_reference"]) / np.log1p(n))

    sm = sm.sort_values(["v69_fast_diagnostic_score", "pos_ratio"], ascending=[False, False])
    sm.to_csv(OUT_SUMMARY, index=False)

    if audits:
        audit = pd.concat(audits, ignore_index=True)
        topv = set(sm.head(12)["variant"])
        audit = audit[audit["variant"].isin(topv)].copy()
        audit.to_csv(OUT_AUDIT, index=False)

    cols = [
        "variant", "file", "family", "ones", "pos_ratio", "diff_vs_reference",
        "score_mean_pred1", "score_mean_pred0",
        "poison_rate_pred1", "poison_penalty_mean_pred1",
        "target_ratio", "query_ratio", "global_pct_min", "term_top_pct_min", "hybrid_mode",
        "v69_fast_diagnostic_score",
    ]
    print("\nTOP SUMMARY", flush=True)
    print(sm[[c for c in cols if c in sm.columns]].head(100).to_string(index=False), flush=True)

    print("\noutputs:", flush=True)
    print(OUT_FEATURES, flush=True)
    print(OUT_BAD, flush=True)
    print(OUT_SUMMARY, flush=True)
    print(OUT_AUDIT, flush=True)
    print(SUB_DIR, flush=True)

if __name__ == "__main__":
    main()
