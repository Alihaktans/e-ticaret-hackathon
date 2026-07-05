from pathlib import Path
import re
import hashlib
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V68_FEATURES = ROOT / "data/processed/v68_independent_query_rank_features.parquet"
V67_FEATURES = ROOT / "data/processed/v67_global_aggressive_blend_scores.parquet"

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v69"

OUT_FEATURES = ROOT / "data/processed/v69_poison_guarded_independent_scores.parquet"
OUT_SUMMARY = OUT_DIR / "v69_poison_guarded_candidate_summary.csv"
OUT_AUDIT = OUT_DIR / "review_v69_poison_guarded_audit.csv"
OUT_BAD_EXAMPLES = OUT_DIR / "v69_auto_bad_examples.csv"

BASES = [
    ROOT / "submissions/final_candidates_v68/FINAL_CANDIDATE_v68_IND_GLOBAL_ratio0p34_ff3bda60.csv",
    ROOT / "submissions/final_candidates_v68/FINAL_CANDIDATE_v68_IND_GLOBAL_ratio0p31684_cdd371b9.csv",
    ROOT / "submissions/final_candidates_v66/FINAL_CANDIDATE_v66_intent_ultra_swap_cap75_699710c5.csv",
    ROOT / "submissions/final_candidates_v60/v60_public080_ultra_cap250.csv",
]

# Risky but guarded.
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

TR_CHARS = str.maketrans({
    "ı": "i", "İ": "i", "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
    "ş": "s", "Ş": "s", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

COLORS = {
    "siyah", "beyaz", "kirmizi", "mavi", "lacivert", "yesil", "sari", "mor",
    "pembe", "gri", "kahverengi", "bej", "ekru", "krem", "turuncu", "bordo",
    "taba", "vizon", "gold", "gumus", "altin", "haki", "fuşya", "fucsia",
}
COLOR_SYNONYMS = {
    "altin": {"gold", "altin"},
    "gold": {"gold", "altin"},
    "gumus": {"gumus", "silver"},
    "siyah": {"siyah", "black"},
    "beyaz": {"beyaz", "white"},
    "kirmizi": {"kirmizi", "red"},
    "mavi": {"mavi", "blue"},
    "yesil": {"yesil", "green"},
    "sari": {"sari", "yellow"},
    "pembe": {"pembe", "pink"},
    "gri": {"gri", "gray", "grey"},
}

GENDER_MALE = {"erkek", "bay", "mens", "men", "man"}
GENDER_FEMALE = {"kadin", "bayan", "girl", "girls", "woman", "women"}
KID = {"cocuk", "cocuklar", "kids", "kid", "jr", "junior", "bebek", "baby", "kiz", "oglan"}
ADULT = {"erkek", "kadin", "bay", "bayan", "mens", "women", "man", "woman"}

ACCESSORY_WORDS = {
    "kilif", "kılıf", "case", "kapak", "stand", "koruyucu", "ekran koruyucu",
    "aksesuar", "aparat", "yedek", "parca", "parça", "askisi", "kayisi", "kablo",
    "adaptör", "adapter", "sarj", "şarj",
}
MAIN_ELECTRONICS = {
    "iphone", "ayfon", "telefon", "tablet", "laptop", "notebook", "bilgisayar",
    "televizyon", "tv", "ps4", "playstation", "kamera",
}

# Query intent groups. If query asks one group but item category/title screams another unrelated group -> penalty.
GROUPS = {
    "phone": {"telefon", "iphone", "ayfon", "samsung galaxy", "xiaomi", "ios cep"},
    "tablet_case": {"tablet kilif", "tablet kılıf", "tab kilif", "tab kılıf"},
    "shoe": {"ayakkabi", "sneaker", "spor ayakkabi", "loafer", "bot", "cizme", "terlik", "sandalet", "krampon"},
    "bag": {"canta", "çanta", "sirt cantasi", "omuz cantasi", "postaci canta"},
    "clothing": {"mont", "ceket", "pantolon", "esofman", "sweatshirt", "t-shirt", "elbise", "gomlek", "kaban"},
    "book": {"kitap", "roman", "kpss", "tyt", "ayt", "paragraf", "dunya klasikleri", "psikoloji"},
    "cosmetic": {"deodorant", "parfum", "gunes kremi", "sampuan", "allik", "kapatıcı", "ruj", "dudak"},
    "home": {"avize", "ayna", "hali", "yolluk", "perde", "kirlent", "masa ortusu", "tencere", "kahvaltilik", "dolap"},
    "pet": {"kedi", "kopek", "pet", "mama"},
    "tire": {"lastik", "195 55 16", "195/55"},
}

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().translate(TR_CHARS)
    x = re.sub(r"[^a-z0-9%./+ -]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def has_word(text, word):
    word = norm(word)
    if not word:
        return False
    if " " in word:
        return word in text
    return re.search(rf"(^|[^a-z0-9]){re.escape(word)}([^a-z0-9]|$)", text) is not None

def any_word(text, words):
    return any(has_word(text, w) for w in words)

def extract_colors(text):
    out = set()
    for c in COLORS:
        if has_word(text, c):
            out.add(c)
    # normalize aliases into canonical if possible
    normed = set()
    for c in out:
        found = False
        for k, vals in COLOR_SYNONYMS.items():
            if c in vals:
                normed.add(k)
                found = True
                break
        if not found:
            normed.add(c)
    return normed

def group_hits(text):
    hits = set()
    for g, words in GROUPS.items():
        if any_word(text, words):
            hits.add(g)
    return hits

def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]

def load_text_tables(df):
    out = df.copy()

    if TERMS.exists():
        terms = pd.read_csv(TERMS)
        term_col = "term_id" if "term_id" in terms.columns else None
        text_cols = [c for c in terms.columns if c != term_col and re.search(r"(query|term|text|name|arama)", c, re.I)]
        if term_col and text_cols:
            tmp = terms[[term_col, text_cols[0]]].rename(columns={text_cols[0]: "query_text"})
            tmp["term_id"] = tmp["term_id"].astype(str)
            out = out.merge(tmp, on="term_id", how="left")

    if ITEMS.exists():
        items = pd.read_csv(ITEMS)
        if "item_id" in items.columns:
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
            tmp = items[use].rename(columns=ren)
            tmp["item_id"] = tmp["item_id"].astype(str)
            out = out.merge(tmp, on="item_id", how="left")
    return out

def poison_score_row(q, title, brand, cat):
    qn = norm(q)
    tn = norm(title)
    bn = norm(brand)
    cn = norm(cat)
    all_item = " ".join([tn, bn, cn])

    reasons = []
    penalty = 0.0

    # 1) Hard gender/age mismatch.
    q_male = any_word(qn, GENDER_MALE)
    q_female = any_word(qn, GENDER_FEMALE)
    item_male = any_word(all_item, GENDER_MALE)
    item_female = any_word(all_item, GENDER_FEMALE)

    if q_male and item_female and not item_male:
        penalty += 0.42; reasons.append("gender_query_male_item_female")
    if q_female and item_male and not item_female:
        penalty += 0.42; reasons.append("gender_query_female_item_male")

    q_kid = any_word(qn, KID)
    item_kid = any_word(all_item, KID)
    q_adult = any_word(qn, ADULT)
    item_adult = any_word(all_item, ADULT)

    # child/adult mismatch. Softer because "çocuk" can be broad.
    if q_kid and item_adult and not item_kid:
        penalty += 0.25; reasons.append("age_query_child_item_adult")
    if q_adult and item_kid and not q_kid:
        penalty += 0.25; reasons.append("age_query_adult_item_child")

    # 2) Accessory vs main product mismatch.
    q_main_elec = any_word(qn, MAIN_ELECTRONICS) and not any_word(qn, ACCESSORY_WORDS)
    item_accessory = any_word(all_item, ACCESSORY_WORDS)
    if q_main_elec and item_accessory:
        penalty += 0.55; reasons.append("main_electronic_query_item_accessory")

    q_accessory = any_word(qn, ACCESSORY_WORDS)
    item_main_phone = any_word(all_item, {"akilli cep telefonu", "ios cep telefonu", "cep telefonu", "telefon"}) and not item_accessory
    if q_accessory and item_main_phone:
        penalty += 0.30; reasons.append("accessory_query_item_main_product")

    # 3) Color mismatch when query explicitly names a color and item explicitly names a different common color.
    qc = extract_colors(qn)
    ic = extract_colors(all_item)
    if qc and ic and qc.isdisjoint(ic):
        # Lower penalty if query has many non-color tokens and item still matches category.
        penalty += 0.18; reasons.append("color_mismatch")

    # 4) Group mismatch for common hard groups. Do not punish broad groups too much.
    qg = group_hits(qn)
    ig = group_hits(all_item)
    if qg and ig:
        # If query's most concrete groups do not appear in item group at all.
        hard_qg = qg - {"home", "clothing"}
        if hard_qg and hard_qg.isdisjoint(ig):
            penalty += 0.30; reasons.append("intent_group_mismatch")
        elif qg.isdisjoint(ig) and len(qn.split()) >= 2:
            penalty += 0.16; reasons.append("soft_group_mismatch")

    # 5) Known bad audit patterns from V68/V67.
    if has_word(qn, "cocuk deodorant") and any_word(all_item, {"men", "erkek"}):
        penalty += 0.60; reasons.append("known_bad_child_deodorant_men")
    if has_word(qn, "scooter kadin") and any_word(all_item, {"erkek", "men"}):
        penalty += 0.60; reasons.append("known_bad_scooter_woman_men")
    if has_word(qn, "kadin esofman") and any_word(all_item, {"erkek", "men"}):
        penalty += 0.60; reasons.append("known_bad_women_tracksuit_men")
    if has_word(qn, "salomon erkek") and any_word(all_item, {"kadin", "women", "w "}) and not any_word(all_item, {"erkek", "men"}):
        penalty += 0.55; reasons.append("known_bad_salomon_men_women")
    if has_word(qn, "lacoste kadin") and any_word(all_item, {"erkek", "men"}):
        penalty += 0.55; reasons.append("known_bad_lacoste_women_men")
    if has_word(qn, "mavi omuz cantasi") and has_word(all_item, "siyah") and not has_word(all_item, "mavi"):
        penalty += 0.35; reasons.append("known_bad_blue_bag_black")
    if has_word(qn, "paten kask") and has_word(all_item, "paten") and not has_word(all_item, "kask"):
        penalty += 0.45; reasons.append("known_bad_skate_helmet_item_skate")
    if has_word(qn, "kucuk dikis makinesi") and any_word(all_item, {"ayak", "aksesuar", "parca"}) and not has_word(all_item, "makinesi"):
        penalty += 0.50; reasons.append("known_bad_sewing_machine_accessory")
    if has_word(qn, "galatasaray termos") and not any_word(all_item, {"galatasaray", "gsstore", "gs"}):
        penalty += 0.35; reasons.append("known_bad_missing_galatasaray")
    if has_word(qn, "crocs cocuk") and any_word(all_item, {"kiz", "girl"}) and not any_word(qn, {"kiz", "girl"}):
        # not always bad, just slight if query says general child and item specific girl
        penalty += 0.08; reasons.append("soft_child_gender_specific")

    # 6) Missing explicit brand in query. If query has known brand token and item brand/title lacks it -> penalty.
    # Lightweight generic list based on observed data.
    brands = [
        "nike", "adidas", "puma", "skechers", "lacoste", "salomon", "slazenger",
        "pierre cardin", "karaca", "loreal", "l'oreal", "la roche", "new balance",
        "u.s. polo", "us polo", "polo assn", "tamer tanca", "hotic", "hotiç",
        "scooter", "tommy", "jack jones", "jack & jones", "gillette", "oral b",
        "apple", "samsung", "xiaomi", "crocs", "ray ban", "ray-ban",
    ]
    for b in brands:
        if has_word(qn, b):
            bnorm = norm(b)
            alt = bnorm.replace(".", "").replace("&", " ").replace("'", "")
            if bnorm not in all_item and alt not in all_item:
                penalty += 0.22; reasons.append("brand_missing_" + re.sub(r"[^a-z0-9]+", "_", bnorm).strip("_"))
            break

    penalty = min(0.95, penalty)
    return penalty, "|".join(reasons)

def build_scores(force=False):
    if OUT_FEATURES.exists() and not force:
        print("Using existing", OUT_FEATURES)
        return pd.read_parquet(OUT_FEATURES)

    src = V68_FEATURES if V68_FEATURES.exists() else V67_FEATURES
    if not src.exists():
        raise FileNotFoundError("V68/V67 score features missing. Run 171 or 170 first.")

    print("Loading", src)
    df = pl.read_parquet(src).to_pandas()
    for c in ["id", "term_id", "item_id"]:
        df[c] = df[c].astype(str)

    if "v68_ind_score" not in df.columns:
        # fallback if V68 was not built
        n = len(df)
        df["global_pct"] = pd.Series(df["v67_final_score"]).rank(pct=True)
        df["term_rank"] = df.groupby("term_id")["v67_final_score"].rank(method="first", ascending=False).astype(int)
        df["term_n"] = df.groupby("term_id")["id"].transform("size").astype(int)
        df["term_top_pct"] = 1.0 - ((df["term_rank"] - 1) / np.maximum(1, df["term_n"] - 1))
        g = df.groupby("term_id")["v67_final_score"]
        df["term_z"] = ((df["v67_final_score"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0)
        df["v68_ind_score"] = (
            0.46 * df["global_pct"] + 0.31 * df["term_top_pct"] +
            0.15 * (1 / (1 + np.exp(-np.clip(df["term_z"], -20, 20)))) +
            0.08 * df["v67_final_score"]
        )

    print("Joining text columns for poison guard")
    df = load_text_tables(df)

    print("Scoring poison penalties")
    penalties = []
    reasons = []
    for row in df[["query_text", "item_title", "item_brand", "item_category"]].itertuples(index=False):
        p, r = poison_score_row(row.query_text, row.item_title, row.item_brand, row.item_category)
        penalties.append(p)
        reasons.append(r)
    df["v69_poison_penalty"] = np.asarray(penalties, dtype=np.float32)
    df["v69_poison_reason"] = reasons

    # Guarded score. Penalty is strong but not absolute; truly high score can survive unless poison is severe.
    df["v69_guarded_score"] = (
        df["v68_ind_score"].astype(float)
        - 0.60 * df["v69_poison_penalty"].astype(float)
        + 0.05 * df["global_pct"].astype(float)
        + 0.03 * df["term_top_pct"].astype(float)
    )

    # Recompute ranks after guard.
    n = len(df)
    df["v69_global_pct"] = pd.Series(df["v69_guarded_score"]).rank(pct=True)
    df["v69_term_rank"] = df.groupby("term_id")["v69_guarded_score"].rank(method="first", ascending=False).astype(int)
    df["v69_term_n"] = df.groupby("term_id")["id"].transform("size").astype(int)
    df["v69_term_top_pct"] = 1.0 - ((df["v69_term_rank"] - 1) / np.maximum(1, df["v69_term_n"] - 1))

    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        "id", "term_id", "item_id", "query_text", "item_title", "item_brand", "item_category",
        "v68_ind_score", "v69_guarded_score", "v69_poison_penalty", "v69_poison_reason",
        "global_pct", "term_rank", "term_n", "term_top_pct", "term_z",
        "v69_global_pct", "v69_term_rank", "v69_term_n", "v69_term_top_pct",
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_parquet(OUT_FEATURES, index=False)

    bad = df[df["v69_poison_penalty"] > 0].sort_values("v69_poison_penalty", ascending=False)
    bad.head(5000).to_csv(OUT_BAD_EXAMPLES, index=False)
    print("saved", OUT_FEATURES, df.shape)
    print("saved bad examples", OUT_BAD_EXAMPLES, bad.shape)
    return df[keep]

def find_base(sample):
    for p in BASES:
        if p.exists():
            try:
                b = pd.read_csv(p)
                b["id"] = b["id"].astype(str)
                if b["id"].equals(sample["id"]):
                    return p, b["prediction"].astype(np.int8).to_numpy()
            except Exception:
                pass
    return None, None

def save_variant(name, pred, sample_ids, df, base_pred, summary, audit_frames, family, meta):
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
    if base_pred is not None:
        row["diff_vs_reference"] = int((pred != base_pred).sum())
    summary.append(row)
    print("saved", file, row)

    if base_pred is not None:
        add = np.where((pred == 1) & (base_pred == 0))[0]
        drop = np.where((pred == 0) & (base_pred == 1))[0]
        if len(add):
            x = df.iloc[add].sort_values("v69_guarded_score", ascending=False).head(200).copy()
            x["variant"] = name; x["action"] = "add_vs_reference"
            audit_frames.append(x)
        if len(drop):
            x = df.iloc[drop].sort_values("v69_guarded_score", ascending=True).head(200).copy()
            x["variant"] = name; x["action"] = "drop_vs_reference"
            audit_frames.append(x)
    else:
        pos = np.where(pred == 1)[0]
        x = df.iloc[pos].sort_values("v69_guarded_score", ascending=False).head(250).copy()
        x["variant"] = name; x["action"] = "top_positive"
        audit_frames.append(x)

def main():
    import sys
    force = "--force" in set(sys.argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    sample_ids = sample["id"].to_numpy()

    df = build_scores(force=force)
    df["id"] = df["id"].astype(str)
    if not df["id"].equals(sample["id"]):
        df = sample.merge(df, on="id", how="left")
        for c in ["v69_guarded_score", "v69_poison_penalty", "v69_global_pct", "v69_term_top_pct"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
        for c in ["v69_term_rank", "v69_term_n"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(999999).astype(int)

    base_path, base_pred = find_base(sample)
    print("Reference for diff only:", base_path)

    score = df["v69_guarded_score"].to_numpy()
    order = np.argsort(-score, kind="mergesort")
    n = len(df)

    summary = []
    audit_frames = []

    # Global top-K guarded.
    for r in GLOBAL_RATIOS:
        k = int(round(n * r))
        pred = np.zeros(n, dtype=np.int8)
        pred[order[:k]] = 1
        save_variant(
            f"v69_GUARDED_GLOBAL_ratio{str(r).replace('.', 'p')}",
            pred, sample_ids, df, base_pred, summary, audit_frames,
            "guarded_global_topk",
            {"target_ratio": float(r), "k": int(k)}
        )

    # Query ratio guarded.
    tr = df["v69_term_rank"].astype(int).to_numpy()
    tn = df["v69_term_n"].astype(int).to_numpy()
    g = df["v69_global_pct"].to_numpy()
    tt = df["v69_term_top_pct"].to_numpy()

    for r in QUERY_RATIOS:
        keep = np.maximum(1, np.ceil(tn * r).astype(int))
        pred = (tr <= keep).astype(np.int8)
        save_variant(
            f"v69_GUARDED_QUERY_ratio{str(r).replace('.', 'p')}",
            pred, sample_ids, df, base_pred, summary, audit_frames,
            "guarded_query_ratio",
            {"query_ratio": float(r)}
        )

    # Consensus guarded.
    for name, gmin, tmin, top_always in CONSENSUS:
        pred = (((g >= gmin) & (tt >= tmin)) | (tr <= top_always)).astype(np.int8)
        save_variant(
            f"v69_GUARDED_CONSENSUS_{name}",
            pred, sample_ids, df, base_pred, summary, audit_frames,
            "guarded_consensus",
            {"global_pct_min": float(gmin), "term_top_pct_min": float(tmin), "top_rank_always": int(top_always)}
        )

    # Hybrid guarded.
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
            f"v69_GUARDED_HYBRID_{name}",
            pred, sample_ids, df, base_pred, summary, audit_frames,
            "guarded_hybrid",
            {"global_ratio": float(gr), "query_ratio": float(qr), "hybrid_mode": mode}
        )

    sm = pd.DataFrame(summary)
    sm["v69_diagnostic_score"] = (
        0.33 * sm["score_mean_pred1"]
        - 0.12 * sm["score_mean_pred0"]
        - 0.18 * sm["poison_penalty_mean_pred1"]
        - 0.06 * sm["poison_rate_pred1"]
        - 0.16 * (sm["pos_ratio"] - 0.34).abs()
    )
    if "diff_vs_reference" in sm.columns:
        sm["v69_diagnostic_score"] += 0.10 * np.minimum(1.0, np.log1p(sm["diff_vs_reference"]) / np.log1p(n))

    sm = sm.sort_values(["v69_diagnostic_score", "pos_ratio"], ascending=[False, False])
    sm.to_csv(OUT_SUMMARY, index=False)

    if audit_frames:
        audit = pd.concat(audit_frames, ignore_index=True)
        topv = sm.head(12)["variant"].tolist()
        audit = audit[audit["variant"].isin(topv)].copy()
        audit.to_csv(OUT_AUDIT, index=False)

    cols = [
        "variant", "file", "family", "ones", "pos_ratio", "diff_vs_reference",
        "score_mean_pred1", "score_mean_pred0",
        "poison_rate_pred1", "poison_penalty_mean_pred1",
        "target_ratio", "query_ratio", "global_pct_min", "term_top_pct_min", "hybrid_mode",
        "v69_diagnostic_score",
    ]
    print("\nTOP SUMMARY")
    print(sm[[c for c in cols if c in sm.columns]].head(100).to_string(index=False))

    print("\noutputs:")
    print(OUT_FEATURES)
    print(OUT_BAD_EXAMPLES)
    print(OUT_SUMMARY)
    print(OUT_AUDIT)
    print(SUB_DIR)

if __name__ == "__main__":
    main()
