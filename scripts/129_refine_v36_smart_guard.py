from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]
RAW_V35_PATHS = [
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv",
]
SWAPS = ROOT / "reports/manual_review/v36_symbolic_guard_scored_swaps.csv"

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

OUT_SWAPS = OUT_DIR / "v36p1_smart_guard_scored_swaps.csv"
OUT_SUMMARY = OUT_DIR / "v36p1_smart_guard_summary.csv"
OUT_EVAL = OUT_DIR / "v36p1_smart_guard_eval.csv"
OUT_REVIEW = OUT_DIR / "review_v36p1_smart_guard_sample.csv"


TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "için", "bir", "adet", "set", "takim", "takım",
    "model", "uyumlu", "orjinal", "orijinal", "yeni", "renk", "boy",
    "numara", "beden", "cm", "mm", "lt", "kg", "gr", "ml", "x", "no",
    "var", "misin", "mısın", "mi", "mı", "mu", "mü", "de", "da",
}

GENERIC_TOKENS = {
    "var", "misin", "mısın", "3lu", "3lü", "2li", "2li", "buyuk", "büyük",
    "kucuk", "küçük", "orta", "sade", "renkli", "set", "adet", "model"
}

PHONE_ACCESSORY_WORDS = {"kilif", "kılıf", "kapak", "case", "ekran", "koruyucu", "cam", "sarj", "şarj", "kablo", "adaptör", "adapter"}
PHONE_WORDS = {"telefon", "iphone", "samsung", "xiaomi", "redmi", "oppo", "realme", "huawei", "cep"}
CAR_WORDS = {"oto", "arac", "araç", "otomobil", "motosiklet", "araba"}
TIRE_WORDS = {"lastik", "r15", "r16", "r17", "r18", "r19", "r20", "jant"}
HAIR_ACCESSORY_WORDS = {"toka", "sac", "saç", "lastik toka", "bilek lastik"}

COLOR_WORDS = {
    "siyah", "beyaz", "kirmizi", "kırmızı", "mavi", "lacivert", "yesil", "yeşil",
    "sari", "sarı", "pembe", "mor", "turuncu", "gri", "antrasit", "kahverengi",
    "bej", "krem", "gold", "gumus", "gümüş", "silver", "bordo"
}

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def toks(x):
    return {t for t in norm(x).split() if len(t) >= 2 and t not in STOP}

def has_any(text, words):
    t = norm(text)
    toks_text = set(t.split())
    for w in words:
        nw = norm(w)
        if " " in nw:
            if nw in t:
                return True
        elif nw in toks_text:
            return True
    return False

def cat_canon(cat):
    c = norm(cat)
    if "telefon" in c and ("kilif" in c or "kapak" in c or "aksesuar" in c):
        return "phone_accessory"
    if "telefon" in c and "aksesuar" not in c:
        return "phone"
    if "otomobil" in c or "motosiklet" in c or "lastik" in c or "jant" in c:
        return "auto"
    if "ayakkabi" in c:
        return "shoe"
    if "giyim" in c:
        return "clothing"
    if "aksesuar" in c:
        return "accessory"
    if "kozmetik" in c or "kisisel bakim" in c:
        return "cosmetic"
    if "elektronik" in c:
        return "electronics"
    if "ev mobilya" in c or "mobilya" in c:
        return "home"
    if "yapi" in c or "hirdavat" in c or "banyo yapi" in c:
        return "hardware"
    if "supermarket" in c or "gida" in c:
        return "supermarket"
    if "kitap" in c:
        return "book"
    if "kirtasiye" in c or "ofis" in c:
        return "stationery"
    if "spor" in c or "outdoor" in c:
        return "sport"
    return "other"

def exact_phrase(query, title):
    q = norm(query)
    t = norm(title)
    if not q or not t:
        return False
    # 1-word query exact hit is too weak; phrase queries exact hit is strong.
    return len(q.split()) >= 2 and q in t

def query_is_generic(query):
    q = toks(query)
    if len(q) == 0:
        return True
    if len(q) <= 1 and next(iter(q)) in {norm(x) for x in GENERIC_TOKENS}:
        return True
    if len(q) <= 2 and len(q & {norm(x) for x in GENERIC_TOKENS}) == len(q):
        return True
    return False

def phone_mismatch(query, title, category):
    # User searches actual phone, add is case/accessory.
    q = norm(query)
    if not has_any(query, PHONE_WORDS):
        return False
    query_mentions_accessory = has_any(query, PHONE_ACCESSORY_WORDS)
    cat = cat_canon(category)
    title_is_accessory = has_any(title, PHONE_ACCESSORY_WORDS)
    if not query_mentions_accessory and (cat == "phone_accessory" or title_is_accessory):
        return True
    return False

def car_paspas_mismatch(query, title, category):
    q = norm(query)
    if "paspas" not in q:
        return False
    # If query is paspas, add should contain paspas or mat; car vacuum/filter is bad.
    t = norm(title)
    c = norm(category)
    if ("paspas" not in t) and ("mat" not in t) and ("paspas" not in c):
        if "supurge" in t or "filtre" in t or "ekran" in t or "telefon" in c:
            return True
    return False

def cam_suyu_mismatch(query, title, category):
    q = norm(query)
    if not ("cam" in q and "suyu" in q):
        return False
    t = norm(title)
    c = norm(category)
    if "cam suyu" in t or "antifriz" in t or "oto" in c or "otomobil" in c:
        return False
    return True

def weapon_air_mismatch(query, title, category):
    q = norm(query)
    if "tufek" not in q and "tabanca" not in q:
        return False
    c = cat_canon(category)
    if c in {"electronics", "phone", "phone_accessory", "home"}:
        return True
    return False

def tire_bonus_or_veto(query, title, category):
    q = norm(query)
    t = norm(title)
    c = norm(category)
    if "lastik" not in q:
        return 0.0, False
    # Plain "lastik" often means tire in this dataset; accept real auto tire, veto obvious hair band.
    if cat_canon(category) == "auto" and ("lastik" in t or re.search(r"\b\d{3}\s*/\s*\d{2}", t)):
        return 0.18, False
    if has_any(title + " " + category, HAIR_ACCESSORY_WORDS):
        return -0.25, True
    return 0.0, False

def category_intent_bonus(query, title, category):
    q = norm(query)
    c = cat_canon(category)
    bonus = 0.0

    if has_any(query, PHONE_WORDS) and c == "phone":
        bonus += 0.18
    if has_any(query, CAR_WORDS) and c == "auto":
        bonus += 0.12
    if any(w in q for w in ["ayakkabi", "bot", "cizme", "sneaker", "terlik"]) and c == "shoe":
        bonus += 0.12
    if any(w in q for w in ["pantolon", "gomlek", "elbise", "kazak", "mont", "ceket", "jean", "tayt"]) and c == "clothing":
        bonus += 0.10
    if any(w in q for w in ["fondoten", "ruj", "sampuan", "sac", "krem", "parfum"]) and c == "cosmetic":
        bonus += 0.10
    if exact_phrase(query, title):
        bonus += 0.12
    return bonus

def generic_veto(query, title, category, lexical_support, category_support):
    if not query_is_generic(query):
        return False
    if exact_phrase(query, title):
        return False
    if lexical_support >= 0.50 or category_support >= 0.60:
        return False
    return True

def color_mismatch(query, title):
    qt = toks(query)
    tt = toks(title)
    qcolors = qt & {norm(x) for x in COLOR_WORDS}
    if not qcolors:
        return False
    return len(qcolors & tt) == 0

def targeted_rules(row):
    q = row.get("query", "")
    ta = row.get("title_add", "")
    ca = row.get("category_add", "")

    lexical = float(row.get("lexical_support_add", 0) or 0)
    cat_support = float(row.get("category_support_add", 0) or 0)

    veto_notes = []
    bonus = 0.0
    penalty = 0.0

    b, tire_veto = tire_bonus_or_veto(q, ta, ca)
    bonus += max(0, b)
    penalty += max(0, -b)
    if tire_veto:
        veto_notes.append("TIRE_QUERY_HAIR_ACCESSORY_ADD")

    if phone_mismatch(q, ta, ca):
        veto_notes.append("PHONE_QUERY_ACCESSORY_ADD")
    if car_paspas_mismatch(q, ta, ca):
        veto_notes.append("PASPAS_QUERY_NON_PASPAS_ADD")
    if cam_suyu_mismatch(q, ta, ca):
        veto_notes.append("CAM_SUYU_WRONG_CATEGORY_ADD")
    if weapon_air_mismatch(q, ta, ca):
        veto_notes.append("WEAPON_QUERY_ELECTRONIC_HOME_ADD")
    if generic_veto(q, ta, ca, lexical, cat_support):
        veto_notes.append("GENERIC_QUERY_LOW_SUPPORT_SWAP")

    bonus += category_intent_bonus(q, ta, ca)

    # Existing V36 sometimes flagged true positives as family mismatch because aliases were incomplete.
    # Here we repair that by adding bonus for canonical category fit.
    if cat_canon(ca) == "auto" and ("lastik" in norm(q) or has_any(q, CAR_WORDS)):
        bonus += 0.08

    if color_mismatch(q, ta):
        penalty += 0.05

    return pd.Series({
        "targeted_veto": int(len(veto_notes) > 0),
        "targeted_notes": ";".join(veto_notes),
        "targeted_bonus": float(bonus),
        "targeted_penalty": float(penalty),
        "cat_canon_add": cat_canon(ca),
    })

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
                row = {
                    "label_set": label_set,
                    "subset": subset_name,
                    "eval_key": f"{label_set}_{subset_name}",
                    "variant": name,
                    "n": int(len(y)),
                }
                row.update(metrics(y, pred[idx]))
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
        val = 0.0
        wsum = 0.0
        used = []
        for _, r in g.iterrows():
            w = weights.get(r["eval_key"], 0.0)
            if w:
                val += w * float(r["macro_f1"])
                wsum += w
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

    anchor_path = first_existing(ANCHOR_PATHS)
    raw_path = first_existing(RAW_V35_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor v33 qprob2000 missing")
    if raw_path is None:
        raise FileNotFoundError("raw v35 b5000 missing")

    anchor = load_pred(anchor_path, sample)
    raw = load_pred(raw_path, sample)

    swaps = pd.read_csv(SWAPS)
    swaps["id_add"] = swaps["id_add"].astype(str)
    swaps["id_drop"] = swaps["id_drop"].astype(str)

    print("loaded swaps:", len(swaps))
    print("anchor:", anchor_path)
    print("raw:", raw_path)

    extra = swaps.apply(targeted_rules, axis=1)
    swaps = pd.concat([swaps, extra], axis=1)

    swaps["smart_guard_score"] = (
        swaps["guard_score"].astype(float)
        + swaps["targeted_bonus"].astype(float)
        - swaps["targeted_penalty"].astype(float)
        - 0.55 * swaps["targeted_veto"].astype(float)
    )

    # If old semantic_trap was the only issue but categorical/phrase evidence is strong, allow repair.
    swaps["semantic_trap_repaired"] = (
        (swaps["semantic_trap_add"].astype(int) == 1)
        & (swaps["targeted_veto"].astype(int) == 0)
        & (
            (swaps["targeted_bonus"].astype(float) >= 0.18)
            | (swaps["phrase_hit_add"].astype(float) == 1)
            | (swaps["lexical_support_add"].astype(float) >= 0.42)
        )
    ).astype(int)

    # Masks
    masks = {
        "v36p1_ultra": (
            (swaps["targeted_veto"] == 0)
            & ((swaps["semantic_trap_add"] == 0) | (swaps["semantic_trap_repaired"] == 1))
            & (swaps["drop_seems_good"] == 0)
            & (swaps["smart_guard_score"] >= 0.22)
        ),
        "v36p1_strict": (
            (swaps["targeted_veto"] == 0)
            & ((swaps["semantic_trap_add"] == 0) | (swaps["semantic_trap_repaired"] == 1))
            & (swaps["smart_guard_score"] >= 0.12)
            & ~((swaps["drop_seems_good"] == 1) & (swaps["smart_guard_score"] < 0.28))
        ),
        "v36p1_balanced": (
            (swaps["targeted_veto"] == 0)
            & ((swaps["semantic_trap_add"] == 0) | (swaps["semantic_trap_repaired"] == 1))
            & (swaps["smart_guard_score"] >= 0.04)
        ),
        "v36p1_impact": (
            (swaps["targeted_veto"] == 0)
            & (swaps["smart_guard_score"] >= -0.02)
            & ~((swaps["semantic_trap_add"] == 1) & (swaps["semantic_trap_repaired"] == 0))
        ),
    }

    caps = {
        "v36p1_ultra": 1500,
        "v36p1_strict": 2300,
        "v36p1_balanced": 3200,
        "v36p1_impact": 4200,
    }

    variants = {
        "anchor_v33_qprob2000": anchor.copy(),
        "raw_v35_b5000": raw.copy(),
    }

    summary_rows = []

    for name, mask in masks.items():
        take = swaps[mask].copy()
        take = take.sort_values(["smart_guard_score", "sem_gain"], ascending=False).head(caps[name]).copy()

        pred = anchor.copy()
        add_idx = take["id_add"].map(id_to_idx)
        drop_idx = take["id_drop"].map(id_to_idx)
        if add_idx.isna().any() or drop_idx.isna().any():
            raise RuntimeError(f"id map failed {name}")

        pred[add_idx.astype(int).to_numpy()] = 1
        pred[drop_idx.astype(int).to_numpy()] = 0
        variants[name] = pred

        out = SUB_DIR / f"FINAL_CANDIDATE_{name}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

        if name in ["v36p1_strict", "v36p1_balanced"]:
            main = SUB_DIR / f"FINAL_MAIN_{name}.csv"
            pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(main, index=False)

        summary_rows.append({
            "variant": name,
            "accepted_swaps": int(len(take)),
            "candidate_file": str(out),
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != raw).sum()),
            "smart_guard_mean": float(take["smart_guard_score"].mean()) if len(take) else np.nan,
            "smart_guard_min": float(take["smart_guard_score"].min()) if len(take) else np.nan,
            "old_guard_mean": float(take["guard_score"].mean()) if len(take) else np.nan,
            "symbolic_gain_mean": float(take["symbolic_gain"].mean()) if len(take) else np.nan,
            "sem_gain_mean": float(take["sem_gain"].mean()) if len(take) else np.nan,
            "targeted_veto_rate": float(take["targeted_veto"].mean()) if len(take) else np.nan,
            "semantic_trap_rate": float(take["semantic_trap_add"].mean()) if len(take) else np.nan,
            "semantic_trap_repaired_rate": float(take["semantic_trap_repaired"].mean()) if len(take) else np.nan,
            "drop_seems_good_rate": float(take["drop_seems_good"].mean()) if len(take) else np.nan,
        })

        print("saved", out, "accepted", len(take), "diff_vs_anchor", int((pred != anchor).sum()))

    # baselines
    for name, pred in [("anchor_v33_qprob2000", anchor), ("raw_v35_b5000", raw)]:
        summary_rows.append({
            "variant": name,
            "accepted_swaps": -1,
            "candidate_file": "",
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != raw).sum()),
            "smart_guard_mean": np.nan,
            "smart_guard_min": np.nan,
            "old_guard_mean": np.nan,
            "symbolic_gain_mean": np.nan,
            "sem_gain_mean": np.nan,
            "targeted_veto_rate": np.nan,
            "semantic_trap_rate": np.nan,
            "semantic_trap_repaired_rate": np.nan,
            "drop_seems_good_rate": np.nan,
        })

    swaps.to_csv(OUT_SWAPS, index=False)

    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)

    summary = pd.DataFrame(summary_rows)
    w = weighted_score(eval_df)
    if len(w):
        summary = summary.merge(w, on="variant", how="left")
    summary = summary.sort_values(["weighted_macro", "accepted_swaps"], ascending=[False, False])
    summary.to_csv(OUT_SUMMARY, index=False)

    # diagnostic review sample
    review_parts = []
    for nm in masks:
        acc = swaps[masks[nm]].copy()
        acc = acc.sort_values(["smart_guard_score", "sem_gain"], ascending=False).head(caps[nm])
        acc["decision_variant"] = nm
        acc["decision"] = "accepted"
        review_parts.append(acc.head(120))
        review_parts.append(acc.tail(80))

    vetoed = swaps[swaps["targeted_veto"] == 1].copy()
    vetoed["decision_variant"] = "targeted_veto"
    vetoed["decision"] = "rejected"
    review_parts.append(vetoed.head(400))

    borderline = swaps[swaps["smart_guard_score"].between(-0.05, 0.15)].copy()
    borderline["decision_variant"] = "borderline"
    borderline["decision"] = "borderline"
    review_parts.append(borderline.sample(min(500, len(borderline)), random_state=2026) if len(borderline) else borderline)

    review = pd.concat(review_parts, ignore_index=True).drop_duplicates(["id_add", "id_drop", "decision_variant"], keep="first")
    keep = [
        "decision_variant", "decision", "swap_rank", "query",
        "id_add", "title_add", "category_add", "brand_add",
        "id_drop", "title_drop", "category_drop", "brand_drop",
        "sem_gain", "guard_score", "smart_guard_score", "symbolic_gain",
        "targeted_veto", "targeted_notes", "targeted_bonus", "targeted_penalty",
        "semantic_trap_add", "semantic_trap_repaired", "drop_seems_good",
        "lexical_support_add", "category_support_add", "lexical_support_drop", "category_support_drop",
        "guard_reason",
    ]
    review[[c for c in keep if c in review.columns]].to_csv(OUT_REVIEW, index=False)

    print("\nSUMMARY")
    cols = [
        "variant", "accepted_swaps", "weighted_macro", "used_min_macro", "mean_precision", "mean_recall",
        "diff_vs_anchor", "diff_vs_raw_v35", "smart_guard_mean", "smart_guard_min",
        "semantic_trap_rate", "drop_seems_good_rate", "candidate_file"
    ]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))

    print("\nTargeted veto counts:")
    print(swaps["targeted_notes"].replace("", np.nan).value_counts(dropna=True).head(30).to_string())

    print("\noutputs:")
    print(OUT_SWAPS)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
