from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
SPARSE_SCORE = ROOT / "data/processed/v26_sparse_text_scores.parquet"

OUT_EVAL = ROOT / "reports/manual_review/v28_sparse_safety_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v28_sparse_safety_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v28_sparse_safety_full_summary.csv"
OUT_RULES = ROOT / "reports/manual_review/v28_sparse_safety_rule_report.csv"
OUT_FLAGGED = ROOT / "reports/manual_review/v28_sparse_safety_flagged_additions.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
    "v26_sparse": ROOT / "reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv",
}

PATHS = {
    "v22_base": [
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv",
        ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v22_v21_vote_full_risky.csv",
    ],
    "v24_anchor": [
        ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v26_v24_anchor.csv",
    ],
    "strict2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget2000.csv",
    ],
    "strict3000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget3000.csv",
    ],
    "strict5000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget5000.csv",
    ],
    "supported2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget2000.csv",
    ],
    "supported3000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget3000.csv",
    ],
    "supported5000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget5000.csv",
    ],
}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def clean_name(x):
    return (
        str(x)
        .replace(".", "p")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace(":", "")
    )

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = (
        x.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def load_pred(name, sample, required=True):
    p = first_existing(PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(name)
        print("missing optional:", name)
        return None

    print("loading", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)

    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {name}")

    return d["prediction"].astype(np.int8).to_numpy()

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1]),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def contains_any(text, words):
    if not text:
        return False
    padded = " " + text + " "
    return any((" " + w + " ") in padded for w in words)

def detect_grade(text):
    if not text:
        return set()

    out = set()
    t = " " + text + " "

    for m in re.finditer(r"\b([1-9]|1[0-2])\s*(sinif|sınıf|class)\b", t):
        out.add(int(m.group(1)))

    for m in re.finditer(r"\b([1-9]|1[0-2])\s*\.\s*(sinif|sınıf|class)\b", t):
        out.add(int(m.group(1)))

    for m in re.finditer(r"\b([1-9]|1[0-2])inci\s+sinif\b", t):
        out.add(int(m.group(1)))

    return out

def make_gender_flags(q, item_text, gender):
    q_male = contains_any(q, ["erkek", "bay"])
    q_female = contains_any(q, ["kadin", "bayan", "kiz"])
    q_child = contains_any(q, ["cocuk", "bebek"])

    g = gender or ""
    item_male = contains_any(g, ["erkek", "bay"]) or contains_any(item_text, ["erkek"])
    item_female = contains_any(g, ["kadin", "bayan", "kiz"]) or contains_any(item_text, ["kadin", "bayan", "kiz"])
    item_unisex = contains_any(g, ["unisex"]) or contains_any(item_text, ["unisex"])

    if item_unisex:
        return False

    if q_male and item_female:
        return True
    if q_female and item_male:
        return True

    return False

def product_type_mismatch(q, item_text):
    phone_words = ["telefon", "iphone", "samsung", "xiaomi", "oppo", "realme", "redmi"]
    phone_accessory = ["kilif", "kılıf", "kapak", "cam", "koruyucu", "sarj", "şarj", "kablo", "adaptör", "adapter", "stand"]

    laptop_words = ["laptop", "notebook", "bilgisayar"]
    laptop_accessory = ["canta", "çanta", "kilif", "kılıf", "stand", "sogutucu", "soğutucu", "mousepad"]

    book_words = ["kitap", "test", "soru", "deneme", "tyt", "ayt", "lgs", "kpss"]

    # Telefon arıyor ama ürün aksesuar ise riskli.
    if contains_any(q, phone_words) and not contains_any(q, phone_accessory):
        if contains_any(item_text, phone_accessory):
            return True

    # Kılıf arıyor ama ürün ana telefon gibi görünüyorsa riskli.
    if contains_any(q, ["kilif", "kılıf", "kapak"]):
        if contains_any(item_text, ["cep telefonu", "akilli telefon", "akıllı telefon"]) and not contains_any(item_text, ["kilif", "kılıf", "kapak"]):
            return True

    # Laptop arıyor ama ürün aksesuar ise riskli.
    if contains_any(q, laptop_words) and not contains_any(q, laptop_accessory):
        if contains_any(item_text, laptop_accessory):
            return True

    # Sınıf/kitap araması ama ürün kitap değilse riskli.
    q_grade = detect_grade(q)
    if q_grade and contains_any(q, book_words):
        if not contains_any(item_text, book_words):
            return True

    return False

def broad_group_mismatch(q, item_text):
    groups = {
        "shoe": ["ayakkabi", "sneaker", "bot", "terlik", "sandalet", "loafer", "çizme", "cizme"],
        "bag": ["canta", "çanta", "sirt cantasi", "sırt çantası", "valiz"],
        "dress": ["elbise", "abiye"],
        "pants": ["pantolon", "jean", "tayt"],
        "top": ["tisort", "tshirt", "sweat", "hoodie", "gomlek", "gömlek", "kazak"],
        "perfume": ["parfum", "parfüm", "koku"],
        "watch": ["saat", "akilli saat", "akıllı saat"],
    }

    q_groups = {k for k, ws in groups.items() if contains_any(q, ws)}
    item_groups = {k for k, ws in groups.items() if contains_any(item_text, ws)}

    if not q_groups or not item_groups:
        return False

    if q_groups.isdisjoint(item_groups):
        # "saat" bazen kitap/ölçü birimi gibi geçebilir; sadece net kıyafet/aksesuar gruplarında uygula.
        return True

    return False

def save_variant(variants, name, pred, ids):
    variants[name] = pred.astype(np.int8)
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v28_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": pred.astype(np.int8)}).to_csv(out, index=False)
    print("saved", out, "ones", int(pred.sum()), "pos_ratio", float(pred.mean()))

print("loading sample")
sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
ids = sample["id"].to_numpy()
n = len(sample)
id_to_idx = pd.Series(np.arange(n), index=sample["id"])

pred_base = load_pred("v22_base", sample)
pred_v24 = load_pred("v24_anchor", sample)

pred_strict2000 = load_pred("strict2000", sample)
pred_strict3000 = load_pred("strict3000", sample, required=False)
pred_strict5000 = load_pred("strict5000", sample, required=False)

pred_supported2000 = load_pred("supported2000", sample)
pred_supported3000 = load_pred("supported3000", sample, required=False)
pred_supported5000 = load_pred("supported5000", sample, required=False)

if pred_strict3000 is None:
    pred_strict3000 = pred_strict2000.copy()
if pred_strict5000 is None:
    pred_strict5000 = pred_strict3000.copy()
if pred_supported3000 is None:
    pred_supported3000 = pred_supported2000.copy()
if pred_supported5000 is None:
    pred_supported5000 = pred_supported3000.copy()

print("loading sparse scores")
score = pd.read_parquet(SPARSE_SCORE)
score["id"] = score["id"].astype(str)

if not score["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
    print("sparse score order mismatch, merging by id")
    score = sample.merge(score, on="id", how="left", validate="one_to_one")

score["term_id"] = score["term_id"].astype(str)
score["item_id"] = score["item_id"].astype(str)

# ------------------------------------------------------------
# Safety flags for added rows only
# ------------------------------------------------------------
add_union = (
    (pred_v24 == 0)
    & (
        (pred_strict5000 == 1)
        | (pred_supported5000 == 1)
        | (pred_strict3000 == 1)
        | (pred_supported3000 == 1)
    )
)

add_idx = np.where(add_union)[0]
print("add union rows:", len(add_idx))

safety = score.iloc[add_idx][[
    "id", "term_id", "item_id", "sparse_word", "sparse_char",
    "sparse_blend", "sparse_rank", "sparse_pct_rank", "sparse_delta_from_max"
]].copy()
safety["idx"] = add_idx

print("loading metadata")
terms = pd.read_csv(TERMS)
terms["term_id"] = terms["term_id"].astype(str)
terms["query_clean"] = terms["query"].map(clean_text)

items = pd.read_csv(ITEMS, low_memory=False)
items["item_id"] = items["item_id"].astype(str)

for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

items["brand_clean"] = items["brand"].map(clean_text)
items["title_clean"] = items["title"].map(clean_text)
items["cat_clean"] = items["category"].map(clean_text)
items["gender_clean"] = items["gender"].map(clean_text)
items["attr_clean"] = items["attributes"].map(clean_text)
items["item_text_clean"] = (
    items["title_clean"] + " " +
    items["cat_clean"] + " " +
    items["brand_clean"] + " " +
    items["gender_clean"] + " " +
    items["attr_clean"].str.slice(0, 500)
).str.strip()

# Brand dictionary
generic_brand_tokens = {
    "mavi", "siyah", "beyaz", "yesil", "yeşil", "kirmizi", "kırmızı", "sari", "sarı",
    "mor", "pembe", "gri", "lacivert", "bej", "gold", "silver", "blue", "red",
    "new", "style", "sport", "spor", "moda", "basic", "classic", "trend", "home",
}

brand_counts = items["brand_clean"].value_counts()
brand_counts = brand_counts[brand_counts >= 20]

single_brand = {}
multi_brands = []

for br in brand_counts.index:
    if not br or br in generic_brand_tokens:
        continue
    toks = br.split()
    if len(toks) == 1:
        tok = toks[0]
        if len(tok) >= 3 and tok not in generic_brand_tokens:
            single_brand.setdefault(tok, set()).add(br)
    elif 2 <= len(toks) <= 4 and len(br) >= 5:
        multi_brands.append(br)

multi_brands = sorted(multi_brands, key=len, reverse=True)[:2500]

def query_brand_set(q):
    toks = set(q.split())
    out = set()

    for tok in toks:
        if tok in single_brand:
            out.update(single_brand[tok])

    padded = " " + q + " "
    for br in multi_brands:
        if (" " + br + " ") in padded:
            out.add(br)

    return out

def brand_mismatch(q, item_brand, item_text):
    qbrands = query_brand_set(q)
    if not qbrands:
        return False, ""

    ib = item_brand or ""
    it = item_text or ""

    for br in qbrands:
        if br == ib or br in ib or ib in br or br in it:
            return False, "|".join(sorted(qbrands))

    return True, "|".join(sorted(qbrands))

safety = safety.merge(terms[["term_id", "query", "query_clean"]], on="term_id", how="left", validate="many_to_one")
safety = safety.merge(
    items[[
        "item_id", "title", "category", "brand", "gender", "age_group", "attributes",
        "brand_clean", "gender_clean", "item_text_clean"
    ]],
    on="item_id",
    how="left",
    validate="many_to_one",
)

brand_bad = []
query_brands = []
grade_bad = []
gender_bad = []
device_bad = []
group_bad = []

for _, r in safety.iterrows():
    q = r.get("query_clean", "") or ""
    item_text = r.get("item_text_clean", "") or ""
    item_brand = r.get("brand_clean", "") or ""
    gender = r.get("gender_clean", "") or ""

    bm, qbs = brand_mismatch(q, item_brand, item_text)
    brand_bad.append(bool(bm))
    query_brands.append(qbs)

    qg = detect_grade(q)
    ig = detect_grade(item_text)
    grade_bad.append(bool(qg and ig and qg.isdisjoint(ig)))

    gender_bad.append(make_gender_flags(q, item_text, gender))
    device_bad.append(product_type_mismatch(q, item_text))
    group_bad.append(broad_group_mismatch(q, item_text))

safety["query_brand_detected"] = query_brands
safety["bad_brand_mismatch"] = brand_bad
safety["bad_grade_mismatch"] = grade_bad
safety["bad_gender_mismatch"] = gender_bad
safety["bad_device_accessory_mismatch"] = device_bad
safety["bad_broad_group_mismatch"] = group_bad

safety["bad_light"] = (
    safety["bad_brand_mismatch"]
    | safety["bad_grade_mismatch"]
    | safety["bad_gender_mismatch"]
    | safety["bad_device_accessory_mismatch"]
)

safety["bad_hard"] = (
    safety["bad_light"]
    | safety["bad_broad_group_mismatch"]
)

bad_light = np.zeros(n, dtype=bool)
bad_hard = np.zeros(n, dtype=bool)
bad_brand = np.zeros(n, dtype=bool)

bad_light[safety["idx"].to_numpy()] = safety["bad_light"].to_numpy()
bad_hard[safety["idx"].to_numpy()] = safety["bad_hard"].to_numpy()
bad_brand[safety["idx"].to_numpy()] = safety["bad_brand_mismatch"].to_numpy()

rule_rows = []
for col in [
    "bad_brand_mismatch",
    "bad_grade_mismatch",
    "bad_gender_mismatch",
    "bad_device_accessory_mismatch",
    "bad_broad_group_mismatch",
    "bad_light",
    "bad_hard",
]:
    rule_rows.append({
        "rule": col,
        "flagged_rows": int(safety[col].sum()),
        "flagged_rate_in_add_union": float(safety[col].mean()),
    })

rule_df = pd.DataFrame(rule_rows)
rule_df.to_csv(OUT_RULES, index=False)

flagged = safety[safety["bad_hard"]].copy()
flagged.to_csv(OUT_FLAGGED, index=False, encoding="utf-8-sig")

print("\nRULE REPORT")
print(rule_df.to_string(index=False))

# ------------------------------------------------------------
# Build candidates
# ------------------------------------------------------------
variants = {}

save_variant(variants, "v22_base_public080", pred_base, ids)
save_variant(variants, "v24_anchor", pred_v24, ids)
save_variant(variants, "v26_strict2000", pred_strict2000, ids)
save_variant(variants, "v26_supported2000", pred_supported2000, ids)

def remove_bad_from_additions(base_pred, bad_mask):
    pred = base_pred.copy()
    remove = (pred_v24 == 0) & (pred == 1) & bad_mask
    pred[remove] = 0
    return pred

save_variant(variants, "v28_strict2000_remove_brand", remove_bad_from_additions(pred_strict2000, bad_brand), ids)
save_variant(variants, "v28_strict2000_remove_light", remove_bad_from_additions(pred_strict2000, bad_light), ids)
save_variant(variants, "v28_strict2000_remove_hard", remove_bad_from_additions(pred_strict2000, bad_hard), ids)

save_variant(variants, "v28_supported2000_remove_light", remove_bad_from_additions(pred_supported2000, bad_light), ids)
save_variant(variants, "v28_supported2000_remove_hard", remove_bad_from_additions(pred_supported2000, bad_hard), ids)

save_variant(variants, "v28_strict3000_remove_light", remove_bad_from_additions(pred_strict3000, bad_light), ids)
save_variant(variants, "v28_strict3000_remove_hard", remove_bad_from_additions(pred_strict3000, bad_hard), ids)

save_variant(variants, "v28_supported3000_remove_light", remove_bad_from_additions(pred_supported3000, bad_light), ids)
save_variant(variants, "v28_supported3000_remove_hard", remove_bad_from_additions(pred_supported3000, bad_hard), ids)

# Add extra candidates from 2000->3000 only if safety pass
quality = (
    score["sparse_blend"].astype("float32").to_numpy()
    + 0.20 * (1.0 - score["sparse_pct_rank"].astype("float32").to_numpy())
).astype("float32")

def add_extra(base_pred, pool_mask, budget, name):
    pred = base_pred.copy()
    idx = np.where(pool_mask & (~bad_light) & (~bad_hard))[0]
    idx = idx[np.argsort(-quality[idx])]
    take = idx[:min(budget, len(idx))]
    pred[take] = 1
    save_variant(variants, name, pred, ids)

strict_extra_2k_3k = (pred_strict3000 == 1) & (pred_strict2000 == 0)
supported_extra_2k_3k = (pred_supported3000 == 1) & (pred_supported2000 == 0)
supported_not_strict_2k = (pred_supported2000 == 1) & (pred_strict2000 == 0)
strict_not_supported_2k = (pred_strict2000 == 1) & (pred_supported2000 == 0)

base_light = remove_bad_from_additions(pred_strict2000, bad_light)
base_hard = remove_bad_from_additions(pred_strict2000, bad_hard)

for budget in [100, 250, 500, 750, 1000]:
    add_extra(base_light, strict_extra_2k_3k, budget, f"v28_strict2000_light_plus_strict_extra{budget}")

for budget in [100, 250, 500]:
    add_extra(base_light, supported_extra_2k_3k, budget, f"v28_strict2000_light_plus_supported_extra{budget}")

for budget in [100, 250, 500]:
    add_extra(base_hard, strict_extra_2k_3k, budget, f"v28_strict2000_hard_plus_strict_extra{budget}")

for budget in [100, 250]:
    add_extra(base_light, supported_not_strict_2k, budget, f"v28_strict2000_light_plus_supported_notstrict{budget}")

# ------------------------------------------------------------
# Full summary
# ------------------------------------------------------------
full_rows = []
for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_base": int((pred != pred_base).sum()),
        "diff_vs_v24": int((pred != pred_v24).sum()),
        "diff_vs_strict2000": int((pred != pred_strict2000).sum()),
        "diff_vs_supported2000": int((pred != pred_supported2000).sum()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------
eval_rows = []

for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label file:", label_set, path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)

    if "assistant_label" not in lab.columns:
        continue

    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)

    subsets = {"all": lab}

    if "needs_recheck" in lab.columns:
        nr = lab["needs_recheck"].fillna(0).astype(str).str.replace(".0", "", regex=False)
        nr = pd.to_numeric(nr, errors="coerce").fillna(0).astype(int)
        subsets["clean"] = lab[nr == 0].copy()

    if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        nr = lab["needs_recheck"].fillna(0).astype(str).str.replace(".0", "", regex=False)
        nr = pd.to_numeric(nr, errors="coerce").fillna(0).astype(int)
        subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()
        subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()

    if label_set in ["v20_active", "v21_active", "v26_sparse"] and "review_bucket" in lab.columns:
        for b, g in lab.groupby("review_bucket"):
            if len(g) >= 20 and g["assistant_label"].nunique() >= 2:
                subsets[f"bucket_{b}"] = g.copy()

    for subset_name, part in subsets.items():
        if len(part) < 30 or part["assistant_label"].nunique() < 2:
            continue

        mapped = part["id"].map(id_to_idx)
        ok = mapped.notna()
        part = part.loc[ok].copy()
        idx = mapped.loc[ok].astype(int).to_numpy()
        y = part["assistant_label"].to_numpy()

        for name, pred in variants.items():
            p = pred[idx]
            row = {
                "label_set": label_set,
                "subset": subset_name,
                "eval_key": f"{label_set}_{subset_name}",
                "variant": name,
                "n": len(part),
            }
            row.update(metrics(y, p))
            eval_rows.append(row)

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT_EVAL, index=False)

legacy_weights = {
    "random_clean_v2_all": 0.30,
    "random_clean_v2_clean": 0.30,
    "manual_v1_clean": 0.35,
    "manual_v1_high_clean": 0.20,
    "manual_v1_all": 0.15,
    "review_v15_vs_v13_clean": 0.20,
    "review_v15_vs_v13_high_medium_clean": 0.20,
    "review_v13_vs_v5_clean": 0.10,
    "review_v13_vs_v5_high_medium_clean": 0.10,
}

active_weights = {
    "v20_active_clean": 0.04,
    "v20_active_high_medium_clean": 0.03,
    "v21_active_clean": 0.12,
    "v21_active_high_medium_clean": 0.10,
    "v21_active_bucket_v21_add5k_added_over_v20": 0.08,
    "v21_active_bucket_v21_vote_added_over_v20": 0.12,
    "v21_active_bucket_v21_vote_removed_from_v20": 0.12,
}

sparse_weights = {
    "v26_sparse_clean": 0.22,
    "v26_sparse_high_medium_clean": 0.18,
    "v26_sparse_bucket_supported_0_500_added": 0.12,
    "v26_sparse_bucket_supported_500_1000_extra": 0.12,
    "v26_sparse_bucket_supported_1000_2000_extra": 0.18,
    "v26_sparse_bucket_supported_2000_3000_extra": 0.12,
    "v26_sparse_bucket_supported_3000_5000_extra_risky": 0.18,
    "v26_sparse_bucket_strictpct3_2000_extra_not_supported": 0.18,
    "v26_sparse_bucket_supported_2000_extra_not_strictpct3": 0.18,
    "v26_sparse_bucket_raw_sparse_5000_extra_not_supported": 0.12,
}

agg_rows = []
v24_pos = float(pred_v24.mean())

for name, g in eval_df.groupby("variant"):
    legacy_score = legacy_wsum = 0.0
    active_score = active_wsum = 0.0
    sparse_score = sparse_wsum = 0.0
    legacy_vals = []
    active_vals = []
    sparse_vals = []

    for _, r in g.iterrows():
        key = r["eval_key"]

        lw = legacy_weights.get(key, 0.0)
        if lw:
            legacy_score += lw * r["macro_f1"]
            legacy_wsum += lw
            legacy_vals.append(r["macro_f1"])

        aw = active_weights.get(key, 0.0)
        if aw:
            active_score += aw * r["macro_f1"]
            active_wsum += aw
            active_vals.append(r["macro_f1"])

        sw = sparse_weights.get(key, 0.0)
        if sw:
            sparse_score += sw * r["macro_f1"]
            sparse_wsum += sw
            sparse_vals.append(r["macro_f1"])

    if legacy_wsum == 0:
        continue

    legacy_weighted = legacy_score / legacy_wsum
    active_weighted = active_score / active_wsum if active_wsum else np.nan
    sparse_weighted = sparse_score / sparse_wsum if sparse_wsum else np.nan

    if sparse_wsum and active_wsum:
        combined_score = 0.58 * legacy_weighted + 0.18 * active_weighted + 0.24 * sparse_weighted
    elif active_wsum:
        combined_score = 0.72 * legacy_weighted + 0.28 * active_weighted
    else:
        combined_score = legacy_weighted

    f = full[full["variant"] == name].iloc[0]
    pos_ratio = float(f["pos_ratio"])
    diff_v24 = int(f["diff_vs_v24"])

    pos_penalty = abs(pos_ratio - v24_pos) * 1.35
    diff_penalty = max(0, diff_v24 - 7000) / 850000.0
    public_safe_score = combined_score - pos_penalty - diff_penalty

    main_eval = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "public_safe_score": public_safe_score,
        "combined_score": combined_score,
        "legacy_weighted": legacy_weighted,
        "active_weighted": active_weighted,
        "sparse_weighted": sparse_weighted,
        "legacy_min_macro": float(np.min(legacy_vals)) if legacy_vals else np.nan,
        "active_min_macro": float(np.min(active_vals)) if active_vals else np.nan,
        "sparse_min_macro": float(np.min(sparse_vals)) if sparse_vals else np.nan,
        "main_min_macro": float(main_eval["macro_f1"].min()) if len(main_eval) else np.nan,
        "main_mean_macro": float(main_eval["macro_f1"].mean()) if len(main_eval) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows)
agg = agg.merge(full, on="variant", how="left")
agg = agg.sort_values(["public_safe_score", "combined_score", "legacy_weighted"], ascending=False)
agg.to_csv(OUT_AGG, index=False)

print("\nTOP PUBLIC SAFE")
print(agg.head(80).to_string(index=False))

print("\nTOP COMBINED")
print(agg.sort_values("combined_score", ascending=False).head(80).to_string(index=False))

print("\nFULL SUMMARY")
print(full.sort_values("diff_vs_v24").to_string(index=False))

print("\nRULE REPORT")
print(rule_df.to_string(index=False))

print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
print("saved:", OUT_RULES)
print("saved:", OUT_FLAGGED)
