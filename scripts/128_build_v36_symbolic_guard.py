from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

CAND_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv",
]

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
    "v26_sparse": ROOT / "reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv",
}

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_SWAPS = OUT_DIR / "v36_symbolic_guard_scored_swaps.csv"
OUT_SUMMARY = OUT_DIR / "v36_symbolic_guard_summary.csv"
OUT_EVAL = OUT_DIR / "v36_symbolic_guard_eval.csv"
OUT_REVIEW = OUT_DIR / "review_v36_symbolic_guard_decision_sample.csv"


# -----------------------------
# Turkish-ish text normalization
# -----------------------------
TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "bir", "adet", "li", "lii", "set", "takim",
    "model", "uyumlu", "orjinal", "orijinal", "yeni", "renk", "boy",
    "numara", "beden", "cm", "mm", "lt", "kg", "gr", "ml", "x", "no",
    "erkek", "kadin", "unisex", "cocuk", "bebek", "kiz", "kız",
}

COLOR = {
    "siyah", "beyaz", "kirmizi", "kırmızı", "mavi", "lacivert", "yesil", "yeşil",
    "sari", "sarı", "pembe", "mor", "turuncu", "gri", "antrasit", "kahverengi",
    "bej", "krem", "gold", "gumus", "gümüş", "silver", "renkli", "bordo",
}

GENDER_TOKENS = {
    "erkek": "erkek",
    "kadin": "kadin",
    "kadın": "kadin",
    "bayan": "kadin",
    "kiz": "kiz",
    "kız": "kiz",
    "unisex": "unisex",
}

AGE_TOKENS = {
    "bebek": "bebek",
    "cocuk": "cocuk",
    "çocuk": "cocuk",
    "kids": "cocuk",
    "genc": "genc",
    "genç": "genc",
    "yetiskin": "yetiskin",
    "yetişkin": "yetiskin",
}

# High-confidence query family rules.
# These are not trying to classify everything; they only veto obvious category traps.
FAMILY_RULES = [
    ("ayakkabi", {"ayakkabi", "bot", "sneaker", "terlik", "sandalet", "cizme", "çizme", "spor ayakkabi", "loafer"}, {"giyim"}),
    ("giyim", {"pantolon", "gomlek", "gömlek", "elbise", "etek", "kazak", "mont", "ceket", "tshirt", "tişört", "sort", "şort", "tayt", "hırka", "hirka", "sweatshirt", "bluz"}, {"giyim"}),
    ("ic_giyim", {"sutyen", "sütyen", "kulot", "külot", "boxer", "corap", "çorap", "pijama"}, {"giyim"}),
    ("canta", {"canta", "çanta", "valiz", "bavul", "beslenme cantasi", "beslenme çantası", "sirt cantasi", "sırt çantası"}, {"aksesuar", "giyim"}),
    ("kozmetik", {"sampuan", "şampuan", "krem", "serum", "parfum", "parfüm", "ruj", "maskara", "fondoten", "wax", "sac", "saç", "cilt", "gunes kremi", "güneş kremi"}, {"kozmetik"}),
    ("elektronik", {"telefon", "tablet", "laptop", "bilgisayar", "kulaklik", "kulaklık", "samsung", "iphone", "xiaomi", "powerbank", "sarj", "şarj", "kamera"}, {"elektronik"}),
    ("ev_mobilya", {"koltuk", "sandalye", "masa", "hali", "halı", "perde", "yorgan", "battaniye", "nevresim", "dolap", "sehpa", "mutfak"}, {"ev", "mobilya", "ev & mobilya"}),
    ("otomotiv", {"oto", "araba", "motor", "motosiklet", "cam suyu", "antifriz", "lastik", "jant", "silecek", "far"}, {"otomotiv", "yapi market", "yapı market"}),
    ("kitap", {"kitap", "roman", "tyt", "ayt", "kpss", "yks", "defter"}, {"kitap", "kirtasiye", "kırtasiye"}),
    ("oyuncak", {"oyuncak", "lego", "puzzle", "bebek arabasi", "akulu araba", "akülü araba"}, {"anne", "bebek", "oyuncak"}),
    ("spor_outdoor", {"tufek", "tüfek", "olta", "kamp", "matara", "dambıl", "dambil", "fitness", "sporcu"}, {"spor", "outdoor"}),
    ("yapi_market", {"musluk", "batarya", "vida", "matkap", "boya", "ampul", "avize", "lavabo", "evye", "duy"}, {"yapi market", "yapı market", "ev & mobilya"}),
]

HARD_TRAPS = [
    # query tokens, banned category roots/tokens, note
    ({"tufek", "tüfek"}, {"elektronik", "bilgisayar", "tablet", "telefon", "aksesuar"}, "tufek_not_electronics"),
    ({"cam", "suyu"}, {"giyim", "ayakkabi", "aksesuar"}, "cam_suyu_not_fashion"),
    ({"bandaj"}, {"ev", "mobilya", "yapi", "yapı", "elektronik"}, "bandaj_not_home_electronics"),
    ({"beslenme", "cantasi"}, {"kozmetik", "elektronik"}, "beslenme_cantasi_not_cosmetic_electronics"),
]


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def toks(x):
    x = norm_text(x)
    out = []
    for t in x.split():
        if len(t) < 2:
            continue
        if t in STOP:
            continue
        out.append(t)
    return set(out)


def root_category(cat):
    c = norm_text(cat)
    if not c:
        return ""
    first = c.split("/")[0].strip()
    return first


def cat_tokens(cat):
    return toks(str(cat).replace("/", " "))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def containment(a, b):
    # fraction of query tokens contained in target tokens
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def extract_colors(tokens):
    return {t for t in tokens if t in {norm_text(c) for c in COLOR}}


def extract_gender(tokens, text=""):
    out = set()
    all_text = norm_text(text)
    for k, v in GENDER_TOKENS.items():
        nk = norm_text(k)
        if nk in tokens or re.search(rf"\b{re.escape(nk)}\b", all_text):
            out.add(v)
    return out


def extract_age(tokens, text=""):
    out = set()
    all_text = norm_text(text)
    for k, v in AGE_TOKENS.items():
        nk = norm_text(k)
        if nk in tokens or re.search(rf"\b{re.escape(nk)}\b", all_text):
            out.add(v)
    return out


def expected_families(q_norm):
    q_tokens = set(q_norm.split())
    families = []
    allowed = set()
    for name, keys, cats in FAMILY_RULES:
        key_norms = {norm_text(k) for k in keys}
        hit = False
        for k in key_norms:
            if not k:
                continue
            # phrase or token match
            if " " in k:
                if k in q_norm:
                    hit = True
                    break
            elif k in q_tokens:
                hit = True
                break
        if hit:
            families.append(name)
            allowed |= {norm_text(c) for c in cats}
    return families, allowed


def category_family_match(q_norm, cat):
    families, allowed = expected_families(q_norm)
    if not families:
        return 0.0, 0.0, ""
    c = norm_text(cat)
    root = root_category(cat)
    ok = 0.0
    for a in allowed:
        if a and (a in c or a in root):
            ok = 1.0
            break
    mismatch = 1.0 - ok
    return ok, mismatch, ",".join(families)


def hard_trap(q_tokens, cat, title):
    c = cat_tokens(cat) | toks(title) | {root_category(cat)}
    notes = []
    for q_need, banned, note in HARD_TRAPS:
        qn = {norm_text(x) for x in q_need}
        bn = {norm_text(x) for x in banned}
        if qn.issubset(q_tokens) and len(c & bn) > 0:
            notes.append(note)
    return notes


def exact_phrase_hit(query, text):
    qn = norm_text(query)
    tn = norm_text(text)
    if not qn or not tn:
        return 0.0
    # For short search queries, exact phrase in title is very powerful.
    if len(qn.split()) <= 4 and qn in tn:
        return 1.0
    return 0.0


def numeric_tokens(x):
    return set(re.findall(r"\d+", norm_text(x)))


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def align_score(score_path, sample, cols):
    d = pd.read_parquet(score_path)
    d["id"] = d["id"].astype(str)
    if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        return d[["id"] + cols].copy()
    return sample[["id"]].merge(d[["id"] + cols], on="id", how="left", validate="one_to_one")


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1], zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_swaps(anchor, cand, sample, pairs, terms, items, v33, v34):
    changed_add = (anchor == 0) & (cand == 1)
    changed_drop = (anchor == 1) & (cand == 0)

    add = pairs.loc[changed_add, ["id", "term_id", "item_id"]].copy()
    drop = pairs.loc[changed_drop, ["id", "term_id", "item_id"]].copy()

    add = add.merge(v33, on="id", how="left").merge(v34, on="id", how="left")
    drop = drop.merge(v33, on="id", how="left").merge(v34, on="id", how="left")

    for d in [add, drop]:
        d["v33_rs"] = (1.0 - d["v33_ft_pct_rank"]).clip(0, 1)
        d["v34_rs"] = (1.0 - d["v34_ce_pct_rank"]).clip(0, 1)
        d["v33_zsig"] = 1.0 / (1.0 + np.exp(-np.clip(d["v33_ft_term_z"].fillna(0).to_numpy(np.float32) / 1.8, -12, 12)))
        d["v34_zsig"] = 1.0 / (1.0 + np.exp(-np.clip(d["v34_ce_term_z"].fillna(0).to_numpy(np.float32) / 1.8, -12, 12)))
        d["sem_score"] = (
            0.42 * d["v34_rs"].to_numpy(np.float32) +
            0.33 * d["v33_rs"].to_numpy(np.float32) +
            0.11 * d["v34_zsig"].to_numpy(np.float32) +
            0.08 * d["v33_zsig"].to_numpy(np.float32)
        ).astype(np.float32)

    add = add.sort_values(["term_id", "sem_score"], ascending=[True, False])
    drop = drop.sort_values(["term_id", "sem_score"], ascending=[True, True])
    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    swaps = add.merge(drop, on=["term_id", "pair_rank"], suffixes=("_add", "_drop"), how="inner")
    swaps["sem_gain"] = swaps["sem_score_add"] - swaps["sem_score_drop"]
    swaps = swaps.sort_values("sem_gain", ascending=False).reset_index(drop=True)
    swaps["swap_rank"] = np.arange(1, len(swaps) + 1)

    terms_small = terms[["term_id", "query"]].copy()
    swaps = swaps.merge(terms_small, on="term_id", how="left", validate="many_to_one")

    for c in ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]:
        if c not in items.columns:
            items[c] = ""

    item_cols = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    items_small = items[item_cols].copy()

    swaps = swaps.merge(items_small.add_suffix("_add"), left_on="item_id_add", right_on="item_id_add", how="left")
    swaps = swaps.merge(items_small.add_suffix("_drop"), left_on="item_id_drop", right_on="item_id_drop", how="left")

    return swaps


def item_symbolic_features(query, title, category, brand, gender, age_group, attributes):
    q_norm = norm_text(query)
    title_norm = norm_text(title)
    cat_norm = norm_text(category)
    brand_norm = norm_text(brand)
    gender_norm = norm_text(gender)
    age_norm = norm_text(age_group)
    attr_norm = norm_text(attributes)

    qt = toks(query)
    title_t = toks(title)
    cat_t = cat_tokens(category)
    brand_t = toks(brand)
    all_t = toks(" ".join([str(title), str(category), str(brand), str(gender), str(age_group), str(attributes)]))

    title_cont = containment(qt, title_t)
    cat_cont = containment(qt, cat_t)
    all_cont = containment(qt, all_t)
    title_jac = jaccard(qt, title_t)
    cat_jac = jaccard(qt, cat_t)

    phrase = exact_phrase_hit(query, title)

    fam_match, fam_mismatch, fam_name = category_family_match(q_norm, category)

    q_colors = extract_colors(qt)
    item_colors = extract_colors(all_t)
    color_required = 1.0 if q_colors else 0.0
    color_match = 0.0
    color_mismatch = 0.0
    if q_colors:
        color_match = 1.0 if len(q_colors & item_colors) > 0 else 0.0
        color_mismatch = 1.0 - color_match

    q_gender = extract_gender(qt, query)
    item_gender = extract_gender(all_t, " ".join([str(title), str(category), str(gender), str(attributes)]))
    gender_required = 1.0 if q_gender else 0.0
    gender_match = 0.0
    gender_conflict = 0.0
    if q_gender:
        if "unisex" in item_gender:
            gender_match = 0.8
        elif len(q_gender & item_gender) > 0:
            gender_match = 1.0
        elif item_gender:
            gender_conflict = 1.0

    q_age = extract_age(qt, query)
    item_age = extract_age(all_t, " ".join([str(title), str(category), str(age_group), str(attributes)]))
    age_required = 1.0 if q_age else 0.0
    age_match = 0.0
    age_conflict = 0.0
    if q_age:
        if len(q_age & item_age) > 0:
            age_match = 1.0
        elif item_age:
            age_conflict = 1.0

    q_nums = numeric_tokens(query)
    item_nums = numeric_tokens(" ".join([str(title), str(category), str(attributes)]))
    numeric_required = 1.0 if q_nums else 0.0
    numeric_match = 0.0
    numeric_mismatch = 0.0
    if q_nums:
        numeric_match = 1.0 if len(q_nums & item_nums) > 0 else 0.0
        # avoid too strong penalty: size numbers may be absent in title
        numeric_mismatch = 0.5 if numeric_match == 0 else 0.0

    brand_in_query = 0.0
    brand_match = 0.0
    if brand_norm and brand_norm in q_norm:
        brand_in_query = 1.0
        brand_match = 1.0
    elif brand_t and len(qt & brand_t) > 0:
        brand_in_query = 1.0
        brand_match = 1.0

    traps = hard_trap(qt, category, title)
    hard_veto = 1.0 if traps else 0.0

    # Low lexical support while model score is high = likely semantic trap.
    lexical_support = max(title_cont, all_cont, phrase)
    category_support = max(cat_cont, fam_match)

    symbolic_quality = (
        0.28 * title_cont +
        0.16 * all_cont +
        0.14 * cat_cont +
        0.15 * fam_match +
        0.08 * phrase +
        0.05 * color_match -
        0.06 * color_mismatch +
        0.04 * gender_match -
        0.08 * gender_conflict +
        0.04 * age_match -
        0.08 * age_conflict +
        0.03 * numeric_match -
        0.03 * numeric_mismatch +
        0.03 * brand_match -
        0.22 * fam_mismatch -
        0.35 * hard_veto
    )

    return {
        "q_tokens": " ".join(sorted(qt)),
        "title_cont": float(title_cont),
        "cat_cont": float(cat_cont),
        "all_cont": float(all_cont),
        "title_jaccard": float(title_jac),
        "cat_jaccard": float(cat_jac),
        "phrase_hit": float(phrase),
        "family_name": fam_name,
        "family_match": float(fam_match),
        "family_mismatch": float(fam_mismatch),
        "color_required": float(color_required),
        "color_match": float(color_match),
        "color_mismatch": float(color_mismatch),
        "gender_required": float(gender_required),
        "gender_match": float(gender_match),
        "gender_conflict": float(gender_conflict),
        "age_required": float(age_required),
        "age_match": float(age_match),
        "age_conflict": float(age_conflict),
        "numeric_required": float(numeric_required),
        "numeric_match": float(numeric_match),
        "numeric_mismatch": float(numeric_mismatch),
        "brand_in_query": float(brand_in_query),
        "brand_match": float(brand_match),
        "lexical_support": float(lexical_support),
        "category_support": float(category_support),
        "hard_veto": float(hard_veto),
        "trap_notes": ",".join(traps),
        "symbolic_quality": float(symbolic_quality),
    }


def add_guard_features(swaps):
    add_feats = []
    drop_feats = []

    for r in swaps.itertuples(index=False):
        add_feats.append(item_symbolic_features(
            r.query, r.title_add, r.category_add, r.brand_add, r.gender_add, r.age_group_add, r.attributes_add
        ))
        drop_feats.append(item_symbolic_features(
            r.query, r.title_drop, r.category_drop, r.brand_drop, r.gender_drop, r.age_group_drop, r.attributes_drop
        ))

    add_df = pd.DataFrame(add_feats).add_suffix("_add")
    drop_df = pd.DataFrame(drop_feats).add_suffix("_drop")

    out = pd.concat([swaps.reset_index(drop=True), add_df, drop_df], axis=1)

    out["symbolic_gain"] = out["symbolic_quality_add"] - out["symbolic_quality_drop"]

    out["semantic_trap_add"] = (
        (out["v34_rs_add"] >= 0.82)
        & (out["lexical_support_add"] < 0.18)
        & (out["category_support_add"] < 0.30)
    ).astype(int)

    out["drop_seems_good"] = (
        (out["symbolic_quality_drop"] >= 0.22)
        & (out["lexical_support_drop"] >= out["lexical_support_add"])
        & (out["category_support_drop"] >= out["category_support_add"])
    ).astype(int)

    out["guard_score"] = (
        0.56 * out["symbolic_gain"].astype(float)
        + 0.24 * out["sem_gain"].astype(float)
        + 0.10 * (out["lexical_support_add"] - out["lexical_support_drop"]).astype(float)
        + 0.10 * (out["category_support_add"] - out["category_support_drop"]).astype(float)
        - 0.35 * out["hard_veto_add"].astype(float)
        - 0.18 * out["semantic_trap_add"].astype(float)
        - 0.16 * out["drop_seems_good"].astype(float)
    )

    out["guard_reason"] = ""
    reasons = []
    for r in out.itertuples(index=False):
        rr = []
        if r.hard_veto_add:
            rr.append(f"ADD_HARD_VETO:{r.trap_notes_add}")
        if r.semantic_trap_add:
            rr.append("ADD_SEMANTIC_TRAP_LOW_LEXICAL_CATEGORY")
        if r.drop_seems_good:
            rr.append("DROP_STILL_LOOKS_GOOD")
        if r.family_mismatch_add:
            rr.append(f"ADD_FAMILY_MISMATCH:{r.family_name_add}")
        if r.family_match_add:
            rr.append(f"ADD_FAMILY_OK:{r.family_name_add}")
        if r.phrase_hit_add:
            rr.append("ADD_EXACT_PHRASE")
        if r.color_mismatch_add:
            rr.append("ADD_COLOR_MISSING")
        if r.gender_conflict_add:
            rr.append("ADD_GENDER_CONFLICT")
        if r.age_conflict_add:
            rr.append("ADD_AGE_CONFLICT")
        reasons.append(";".join(rr))
    out["guard_reason"] = reasons

    return out


def build_variant(anchor, swaps, accept_mask, name, max_swaps=None):
    take = swaps[accept_mask].copy()
    take = take.sort_values(["guard_score", "sem_gain"], ascending=False)
    if max_swaps is not None:
        take = take.head(max_swaps).copy()

    pred = anchor.copy()
    # id indices were not kept; map ids later in main via global id_to_idx
    return take, pred


def evaluate_variants(variants, sample):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    rows = []
    for label_set, path in LABEL_FILES.items():
        if not path.exists():
            print("missing label:", label_set, path)
            continue

        lab = pd.read_csv(path)
        if "id" not in lab.columns or "assistant_label" not in lab.columns:
            continue

        lab["id"] = lab["id"].astype(str)
        lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
        if len(lab) < 30:
            continue
        lab["assistant_label"] = lab["assistant_label"].astype(int)

        subsets = {"all": lab}
        if "needs_recheck" in lab.columns:
            nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
            subsets["clean"] = lab[nr == 0].copy()

        if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
            conf = lab["assistant_confidence"].astype(str).str.lower()
            nr = pd.to_numeric(lab["needs_recheck"], errors="coerce").fillna(0).astype(int)
            subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()
            subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()

        for subset_name, part in subsets.items():
            if len(part) < 30 or part["assistant_label"].nunique() < 2:
                continue
            idx = part["id"].map(id_to_idx)
            ok = idx.notna()
            if ok.sum() < 30:
                continue

            idx = idx[ok].astype(int).to_numpy()
            y = part.loc[ok, "assistant_label"].astype(int).to_numpy()

            for name, pred in variants.items():
                p = pred[idx]
                row = {
                    "label_set": label_set,
                    "subset": subset_name,
                    "eval_key": f"{label_set}_{subset_name}",
                    "variant": name,
                    "n": int(len(y)),
                }
                row.update(metrics(y, p))
                rows.append(row)

    return pd.DataFrame(rows)


def weighted_score(eval_df):
    weights = {
        "random_clean_v2_all": 0.26,
        "random_clean_v2_clean": 0.26,
        "manual_v1_clean": 0.25,
        "manual_v1_high_clean": 0.12,
        "manual_v1_high_medium_clean": 0.10,
        "review_v15_vs_v13_clean": 0.13,
        "review_v15_vs_v13_high_medium_clean": 0.11,
        "review_v13_vs_v5_clean": 0.05,
        "v20_active_clean": 0.04,
        "v21_active_clean": 0.09,
        "v21_active_high_medium_clean": 0.07,
        "v26_sparse_clean": 0.07,
        "v26_sparse_high_medium_clean": 0.05,
    }

    rows = []
    if len(eval_df) == 0:
        return pd.DataFrame()

    for name, g in eval_df.groupby("variant"):
        wsum = 0.0
        val = 0.0
        used = []
        for _, r in g.iterrows():
            w = weights.get(r["eval_key"], 0.0)
            if w:
                wsum += w
                val += w * r["macro_f1"]
                used.append(float(r["macro_f1"]))
        if wsum == 0:
            continue
        rows.append({
            "variant": name,
            "weighted_macro": float(val / wsum),
            "used_min_macro": float(np.min(used)) if used else np.nan,
            "eval_count": int(len(g)),
            "mean_precision": float(g["precision"].mean()),
            "mean_recall": float(g["recall"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("submission_pairs order mismatch sample")

    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)

    items = pd.read_csv(ITEMS, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)

    v33 = align_score(V33, sample, ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"])
    v34 = align_score(V34, sample, ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"])

    anchor_path = first_existing(ANCHOR_PATHS)
    cand_path = first_existing(CAND_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor v33 qprob2000 not found")
    if cand_path is None:
        raise FileNotFoundError("candidate v35 b5000 not found")

    print("anchor:", anchor_path)
    print("candidate:", cand_path)

    anchor = load_pred(anchor_path, sample)
    cand = load_pred(cand_path, sample)

    swaps = make_swaps(anchor, cand, sample, pairs, terms, items, v33, v34)
    print("swaps:", len(swaps))
    swaps = add_guard_features(swaps)

    # Decision masks. These are intentionally different from semantic model thresholds.
    masks = {
        "v36_guard_ultra_strict": (
            (swaps["guard_score"] >= 0.18)
            & (swaps["symbolic_gain"] >= 0.00)
            & (swaps["hard_veto_add"] == 0)
            & (swaps["semantic_trap_add"] == 0)
            & (swaps["drop_seems_good"] == 0)
            & (swaps["lexical_support_add"] >= 0.20)
        ),
        "v36_guard_strict": (
            (swaps["guard_score"] >= 0.08)
            & (swaps["symbolic_gain"] >= -0.04)
            & (swaps["hard_veto_add"] == 0)
            & (swaps["semantic_trap_add"] == 0)
        ),
        "v36_guard_balanced": (
            (swaps["guard_score"] >= -0.02)
            & (swaps["hard_veto_add"] == 0)
            & (swaps["semantic_trap_add"] == 0)
            & ~((swaps["family_mismatch_add"] == 1) & (swaps["lexical_support_add"] < 0.25))
        ),
        "v36_guard_loose": (
            (swaps["guard_score"] >= -0.10)
            & (swaps["hard_veto_add"] == 0)
            & ~((swaps["semantic_trap_add"] == 1) & (swaps["lexical_support_add"] < 0.15))
        ),
    }

    # Max swap caps prevent symbolic score from over-accepting.
    caps = {
        "v36_guard_ultra_strict": 2000,
        "v36_guard_strict": 3000,
        "v36_guard_balanced": 4000,
        "v36_guard_loose": 5000,
    }

    variants = {
        "anchor_v33_qprob2000": anchor.copy(),
        "raw_v35_b5000": cand.copy(),
    }

    summary_rows = []
    take_tables = {}

    for name, mask in masks.items():
        take = swaps[mask].copy()
        take = take.sort_values(["guard_score", "sem_gain"], ascending=False).head(caps[name]).copy()

        pred = anchor.copy()
        add_idx = take["id_add"].astype(str).map(id_to_idx)
        drop_idx = take["id_drop"].astype(str).map(id_to_idx)
        if add_idx.isna().any() or drop_idx.isna().any():
            raise RuntimeError(f"id mapping failed for {name}")

        pred[add_idx.astype(int).to_numpy()] = 1
        pred[drop_idx.astype(int).to_numpy()] = 0
        variants[name] = pred
        take_tables[name] = take

        out = SUB_DIR / f"FINAL_CANDIDATE_{name}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

        summary_rows.append({
            "variant": name,
            "accepted_swaps": int(len(take)),
            "candidate_file": str(out),
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != cand).sum()),
            "guard_score_mean": float(take["guard_score"].mean()) if len(take) else np.nan,
            "guard_score_min": float(take["guard_score"].min()) if len(take) else np.nan,
            "symbolic_gain_mean": float(take["symbolic_gain"].mean()) if len(take) else np.nan,
            "sem_gain_mean": float(take["sem_gain"].mean()) if len(take) else np.nan,
            "hard_veto_rate": float(take["hard_veto_add"].mean()) if len(take) else np.nan,
            "semantic_trap_rate": float(take["semantic_trap_add"].mean()) if len(take) else np.nan,
        })

        print("saved:", out, "accepted_swaps:", len(take), "diff_vs_anchor:", int((pred != anchor).sum()))

    # save final main aliases for top practical choices
    for alias in ["v36_guard_strict", "v36_guard_balanced"]:
        if alias in variants:
            out_main = SUB_DIR / f"FINAL_MAIN_{alias}.csv"
            pd.DataFrame({"id": sample["id"], "prediction": variants[alias].astype(np.int8)}).to_csv(out_main, index=False)
            print("saved main alias:", out_main)

    swaps.to_csv(OUT_SWAPS, index=False)

    # Review sample: accepted/rejected borderline and veto examples
    review_parts = []
    for name, take in take_tables.items():
        x = take.copy()
        x["decision_variant"] = name
        x["decision"] = "accepted"
        review_parts.append(x.head(150))
        review_parts.append(x.tail(100))

    rejected = swaps[
        (swaps["hard_veto_add"] == 1)
        | (swaps["semantic_trap_add"] == 1)
        | (swaps["drop_seems_good"] == 1)
        | (swaps["guard_score"].between(-0.12, 0.12))
    ].copy()
    rejected["decision_variant"] = "diagnostic_rejected"
    rejected["decision"] = "rejected_or_borderline"
    review_parts.append(rejected.sort_values(["hard_veto_add", "semantic_trap_add", "guard_score"], ascending=[False, False, True]).head(700))

    review = pd.concat(review_parts, ignore_index=True)
    keep_cols = [
        "decision_variant", "decision", "swap_rank", "term_id", "query",
        "id_add", "title_add", "category_add", "brand_add", "gender_add", "age_group_add",
        "id_drop", "title_drop", "category_drop", "brand_drop", "gender_drop", "age_group_drop",
        "sem_gain", "symbolic_gain", "guard_score", "guard_reason",
        "v33_ft_score_add", "v34_ce_score_add", "v33_ft_score_drop", "v34_ce_score_drop",
        "lexical_support_add", "category_support_add", "lexical_support_drop", "category_support_drop",
        "hard_veto_add", "semantic_trap_add", "drop_seems_good",
    ]
    review[keep_cols].to_csv(OUT_REVIEW, index=False)

    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)
    w = weighted_score(eval_df)

    summary = pd.DataFrame(summary_rows)
    base_rows = []
    for name, pred in variants.items():
        if name.startswith("v36_"):
            continue
        base_rows.append({
            "variant": name,
            "accepted_swaps": -1,
            "candidate_file": "",
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != cand).sum()),
            "guard_score_mean": np.nan,
            "guard_score_min": np.nan,
            "symbolic_gain_mean": np.nan,
            "sem_gain_mean": np.nan,
            "hard_veto_rate": np.nan,
            "semantic_trap_rate": np.nan,
        })
    summary = pd.concat([pd.DataFrame(base_rows), summary], ignore_index=True)

    if len(w):
        summary = summary.merge(w, on="variant", how="left")
    summary = summary.sort_values(["weighted_macro", "accepted_swaps"], ascending=[False, False])
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nSUMMARY")
    show_cols = [
        "variant", "accepted_swaps", "weighted_macro", "used_min_macro",
        "mean_precision", "mean_recall", "diff_vs_anchor", "diff_vs_raw_v35",
        "ones", "pos_ratio", "guard_score_mean", "symbolic_gain_mean", "candidate_file",
    ]
    print(summary[[c for c in show_cols if c in summary.columns]].to_string(index=False))

    print("\noutputs:")
    print(OUT_SWAPS)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
