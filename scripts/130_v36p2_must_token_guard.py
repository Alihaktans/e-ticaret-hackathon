from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
SWAPS = ROOT / "reports/manual_review/v36p1_smart_guard_scored_swaps.csv"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

RAW_V35_PATHS = [
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

OUT_SWAPS = OUT_DIR / "v36p2_must_token_guard_scored_swaps.csv"
OUT_SUMMARY = OUT_DIR / "v36p2_must_token_guard_summary.csv"
OUT_EVAL = OUT_DIR / "v36p2_must_token_guard_eval.csv"
OUT_REVIEW = OUT_DIR / "review_v36p2_must_token_guard_sample.csv"


TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "için", "bir", "adet", "set", "takim", "takım",
    "model", "uyumlu", "orjinal", "orijinal", "yeni", "renk", "boy",
    "numara", "beden", "cm", "mm", "lt", "kg", "gr", "ml", "x", "no",
    "var", "misin", "mısın", "mi", "mı", "mu", "mü", "de", "da",
    "olan", "icin", "için", "veya", "plus", "pro", "max",
}

PRODUCT_GENERIC = {
    # broad products
    "ayakkabi", "bot", "cizme", "sneaker", "terlik", "sandalet", "babet", "loafer",
    "pantolon", "gomlek", "elbise", "etek", "kazak", "mont", "ceket", "tshirt", "tisort",
    "jean", "tayt", "sort", "sweatshirt", "bluz", "hirka",
    "canta", "valiz", "cuzdan", "sirt", "beslenme",
    "telefon", "cep", "tablet", "laptop", "bilgisayar", "kulaklik", "kamera",
    "kilif", "kapak", "ekran", "koruyucu", "sarj", "adaptör", "adapter", "kablo",
    "lastik", "jant", "oto", "arac", "araba", "motosiklet", "paspas", "silecek", "suyu",
    "krem", "fondoten", "ruj", "sampuan", "parfum", "sac", "cilt", "serum", "wax",
    "masa", "sandalye", "koltuk", "hali", "perde", "dolap", "sehpa", "tablo",
    "kitap", "defter", "kalem", "oyuncak", "cikolata", "seker", "boncuk", "tesbih",
    "ram", "ssd", "ddr", "ddr4", "ddr5", "playstation", "ps2", "ps3", "ps4", "ps5",
    "makinesi", "makina", "cihazi", "cihaz", "aksesuar", "askisi", "tokasi", "toka",
}

COLOR = {
    "siyah", "beyaz", "kirmizi", "mavi", "lacivert", "yesil", "sari", "pembe", "mor",
    "turuncu", "gri", "antrasit", "kahverengi", "bej", "krem", "gold", "gumus", "silver", "bordo"
}

GENDER_AGE = {"erkek", "kadin", "bayan", "kiz", "cocuk", "bebek", "unisex", "genc", "yetiskin"}

GENERIC_QUERY_TOKENS = {
    "var", "misin", "3lu", "3lü", "2li", "4lu", "5li", "buyuk", "kucuk", "orta",
    "sade", "renkli", "set", "adet", "model", "urun", "ürün"
}

PHONE_WORDS = {"telefon", "iphone", "samsung", "xiaomi", "redmi", "oppo", "realme", "huawei", "cep"}
PHONE_ACCESSORY = {"kilif", "kılıf", "kapak", "case", "ekran", "koruyucu", "cam", "sarj", "şarj", "kablo", "adaptör", "adapter"}


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


def q_tokens_raw(x):
    return {t for t in norm(x).split() if len(t) >= 2}


def has_any(text, words):
    ts = q_tokens_raw(text)
    ns = {norm(w) for w in words}
    return len(ts & ns) > 0


def exact_phrase(query, text):
    q = norm(query)
    t = norm(text)
    if not q or not t:
        return False
    return len(q.split()) >= 2 and q in t


def important_query_tokens(query):
    qt = toks(query)
    generic = {norm(x) for x in PRODUCT_GENERIC}
    color = {norm(x) for x in COLOR}
    gender_age = {norm(x) for x in GENDER_AGE}
    out = set()
    for t in qt:
        if t in color or t in gender_age:
            continue
        # Keep model/brand-looking numbers and alphanumerics, but remove pure very common product tokens.
        if t in generic:
            continue
        out.add(t)
    return out


def support_ratio(tokens, text):
    if not tokens:
        return 1.0
    tt = q_tokens_raw(text)
    return len(tokens & tt) / max(1, len(tokens))


def support_any(tokens, text):
    if not tokens:
        return False
    return len(tokens & q_tokens_raw(text)) > 0


def generic_query_veto(query, title_add, category_add, lex_add, cat_add):
    raw = q_tokens_raw(query)
    clean = toks(query)
    generic_norm = {norm(x) for x in GENERIC_QUERY_TOKENS}

    # If all useful tokens disappeared or only generic tokens remain.
    if len(clean) == 0 or clean.issubset(generic_norm):
        if exact_phrase(query, title_add):
            return False
        if float(lex_add) >= 0.55 or float(cat_add) >= 0.65:
            return False
        return True
    return False


def main_item_accessory_veto(query, title_add, category_add):
    # Query asks actual phone, add is accessory/case.
    q = norm(query)
    t = norm(title_add)
    c = norm(category_add)
    if has_any(query, PHONE_WORDS):
        query_says_accessory = has_any(query, PHONE_ACCESSORY)
        add_is_accessory = has_any(title_add + " " + category_add, PHONE_ACCESSORY) or ("telefon aksesuarlari" in c)
        add_is_actual_phone = ("akilli cep telefonu" in c) or (("telefon" in c) and not add_is_accessory)
        if (not query_says_accessory) and add_is_accessory and not add_is_actual_phone:
            return True, "PHONE_MAIN_QUERY_ACCESSORY_ADD"

    # Query asks PlayStation/PS game/device; add title must contain PS/playstation tokens, not only category.
    if has_any(query, {"playstation", "ps2", "ps3", "ps4", "ps5"}):
        if not has_any(title_add, {"playstation", "ps2", "ps3", "ps4", "ps5"}):
            return True, "PLAYSTATION_QUERY_TITLE_MISSING"

    # Query asks RAM / DDR component; laptop with RAM can be a trap unless category is component/RAM.
    if has_any(query, {"ram", "ddr4", "ddr5"}):
        c_norm = norm(category_add)
        if "bellek" not in c_norm and "ram" not in c_norm:
            if "dizustu" in c_norm or "bilgisayarlar" in c_norm:
                return True, "RAM_QUERY_LAPTOP_ADD"

    # Query asks accessory part such as toka; add should be accessory or title strongly contains it.
    if has_any(query, {"tokasi", "toka"}):
        if not has_any(title_add + " " + category_add, {"toka", "aksesuar"}):
            return True, "TOKA_QUERY_ADD_NOT_TOKA"

    return False, ""


def must_token_features(row):
    query = row.get("query", "")
    title_add = row.get("title_add", "")
    category_add = row.get("category_add", "")
    brand_add = row.get("brand_add", "")
    title_drop = row.get("title_drop", "")
    category_drop = row.get("category_drop", "")
    brand_drop = row.get("brand_drop", "")

    important = important_query_tokens(query)
    add_text = f"{title_add} {category_add} {brand_add}"
    drop_text = f"{title_drop} {category_drop} {brand_drop}"

    add_ratio = support_ratio(important, add_text)
    drop_ratio = support_ratio(important, drop_text)
    add_any = support_any(important, add_text)
    drop_any = support_any(important, drop_text)

    # Modifier/brand/model missing: reject if add fails important tokens and drop is at least as good.
    must_missing_veto = False
    if important:
        if add_ratio < 0.50 and drop_ratio >= add_ratio:
            must_missing_veto = True
        # For one strong uncommon token, no match is suspicious.
        if len(important) == 1 and not add_any:
            must_missing_veto = True

    gen_veto = generic_query_veto(
        query,
        title_add,
        category_add,
        row.get("lexical_support_add", 0),
        row.get("category_support_add", 0),
    )

    acc_veto, acc_note = main_item_accessory_veto(query, title_add, category_add)

    notes = []
    if must_missing_veto:
        notes.append("IMPORTANT_QUERY_TOKEN_MISSING_ADD")
    if gen_veto:
        notes.append("GENERIC_QUERY_LOW_SUPPORT")
    if acc_veto:
        notes.append(acc_note)

    return pd.Series({
        "important_tokens": " ".join(sorted(important)),
        "must_add_ratio": float(add_ratio),
        "must_drop_ratio": float(drop_ratio),
        "must_gain": float(add_ratio - drop_ratio),
        "must_missing_veto": int(must_missing_veto),
        "generic_query_veto": int(gen_veto),
        "main_accessory_veto": int(acc_veto),
        "v36p2_veto": int(len(notes) > 0),
        "v36p2_notes": ";".join(notes),
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
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
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
        raise FileNotFoundError("anchor not found")
    if raw_path is None:
        raise FileNotFoundError("raw v35 not found")

    anchor = load_pred(anchor_path, sample)
    raw = load_pred(raw_path, sample)

    swaps = pd.read_csv(SWAPS)
    swaps["id_add"] = swaps["id_add"].astype(str)
    swaps["id_drop"] = swaps["id_drop"].astype(str)

    print("loaded swaps:", len(swaps))
    extra = swaps.apply(must_token_features, axis=1)
    swaps = pd.concat([swaps, extra], axis=1)

    # Keep existing smart guard, then add independent query-token intent signal.
    swaps["v36p2_score"] = (
        swaps["smart_guard_score"].astype(float)
        + 0.16 * swaps["must_gain"].astype(float)
        + 0.10 * swaps["must_add_ratio"].astype(float)
        - 0.45 * swaps["must_missing_veto"].astype(float)
        - 0.60 * swaps["generic_query_veto"].astype(float)
        - 0.65 * swaps["main_accessory_veto"].astype(float)
    )

    # Accept masks
    masks = {
        "v36p2_ultra": (
            (swaps["v36p2_veto"] == 0)
            & (swaps["targeted_veto"] == 0)
            & ((swaps["semantic_trap_add"] == 0) | (swaps["semantic_trap_repaired"] == 1))
            & (swaps["drop_seems_good"] == 0)
            & (swaps["v36p2_score"] >= 0.30)
        ),
        "v36p2_strict": (
            (swaps["v36p2_veto"] == 0)
            & (swaps["targeted_veto"] == 0)
            & ((swaps["semantic_trap_add"] == 0) | (swaps["semantic_trap_repaired"] == 1))
            & (swaps["v36p2_score"] >= 0.20)
            & ~((swaps["drop_seems_good"] == 1) & (swaps["v36p2_score"] < 0.34))
        ),
        "v36p2_balanced": (
            (swaps["v36p2_veto"] == 0)
            & (swaps["targeted_veto"] == 0)
            & ((swaps["semantic_trap_add"] == 0) | (swaps["semantic_trap_repaired"] == 1))
            & (swaps["v36p2_score"] >= 0.12)
        ),
        "v36p2_impact": (
            (swaps["v36p2_veto"] == 0)
            & (swaps["targeted_veto"] == 0)
            & ~((swaps["semantic_trap_add"] == 1) & (swaps["semantic_trap_repaired"] == 0))
            & (swaps["v36p2_score"] >= 0.06)
        ),
    }

    caps = {
        "v36p2_ultra": 1200,
        "v36p2_strict": 1800,
        "v36p2_balanced": 2600,
        "v36p2_impact": 3400,
    }

    variants = {
        "anchor_v33_qprob2000": anchor.copy(),
        "raw_v35_b5000": raw.copy(),
    }

    summary_rows = []

    for name, mask in masks.items():
        take = swaps[mask].copy()
        take = take.sort_values(["v36p2_score", "sem_gain"], ascending=False).head(caps[name]).copy()

        pred = anchor.copy()
        add_idx = take["id_add"].map(id_to_idx)
        drop_idx = take["id_drop"].map(id_to_idx)
        if add_idx.isna().any() or drop_idx.isna().any():
            raise RuntimeError(f"id map failed: {name}")

        pred[add_idx.astype(int).to_numpy()] = 1
        pred[drop_idx.astype(int).to_numpy()] = 0
        variants[name] = pred

        out = SUB_DIR / f"FINAL_CANDIDATE_{name}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

        if name in ["v36p2_strict", "v36p2_balanced"]:
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
            "v36p2_score_mean": float(take["v36p2_score"].mean()) if len(take) else np.nan,
            "v36p2_score_min": float(take["v36p2_score"].min()) if len(take) else np.nan,
            "smart_guard_mean": float(take["smart_guard_score"].mean()) if len(take) else np.nan,
            "must_add_ratio_mean": float(take["must_add_ratio"].mean()) if len(take) else np.nan,
            "must_gain_mean": float(take["must_gain"].mean()) if len(take) else np.nan,
            "semantic_trap_rate": float(take["semantic_trap_add"].mean()) if len(take) else np.nan,
            "drop_seems_good_rate": float(take["drop_seems_good"].mean()) if len(take) else np.nan,
        })

        print("saved:", out, "accepted:", len(take), "diff_vs_anchor:", int((pred != anchor).sum()))

    for name, pred in [("anchor_v33_qprob2000", anchor), ("raw_v35_b5000", raw)]:
        summary_rows.append({
            "variant": name,
            "accepted_swaps": -1,
            "candidate_file": "",
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != raw).sum()),
            "v36p2_score_mean": np.nan,
            "v36p2_score_min": np.nan,
            "smart_guard_mean": np.nan,
            "must_add_ratio_mean": np.nan,
            "must_gain_mean": np.nan,
            "semantic_trap_rate": np.nan,
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

    # Review sample
    parts = []
    for name, mask in masks.items():
        acc = swaps[mask].copy().sort_values(["v36p2_score", "sem_gain"], ascending=False).head(caps[name])
        acc["decision_variant"] = name
        acc["decision"] = "accepted"
        parts.append(acc.head(100))
        parts.append(acc.tail(100))

    vetoed = swaps[(swaps["v36p2_veto"] == 1) | (swaps["targeted_veto"] == 1)].copy()
    vetoed["decision_variant"] = "vetoed"
    vetoed["decision"] = "rejected"
    parts.append(vetoed.sort_values(["v36p2_veto", "targeted_veto", "v36p2_score"], ascending=[False, False, True]).head(500))

    borderline = swaps[swaps["v36p2_score"].between(0.05, 0.22)].copy()
    borderline["decision_variant"] = "borderline"
    borderline["decision"] = "borderline"
    if len(borderline):
        parts.append(borderline.sample(min(500, len(borderline)), random_state=2026))

    review = pd.concat(parts, ignore_index=True).drop_duplicates(["id_add", "id_drop", "decision_variant"], keep="first")

    keep = [
        "decision_variant", "decision", "swap_rank", "query", "important_tokens",
        "id_add", "title_add", "category_add", "brand_add",
        "id_drop", "title_drop", "category_drop", "brand_drop",
        "sem_gain", "smart_guard_score", "v36p2_score", "symbolic_gain",
        "must_add_ratio", "must_drop_ratio", "must_gain",
        "v36p2_veto", "v36p2_notes", "targeted_veto", "targeted_notes",
        "semantic_trap_add", "semantic_trap_repaired", "drop_seems_good",
        "lexical_support_add", "category_support_add", "lexical_support_drop", "category_support_drop",
    ]
    review[[c for c in keep if c in review.columns]].to_csv(OUT_REVIEW, index=False)

    print("\nSUMMARY")
    cols = [
        "variant", "accepted_swaps", "weighted_macro", "used_min_macro", "mean_precision", "mean_recall",
        "diff_vs_anchor", "diff_vs_raw_v35", "v36p2_score_mean", "v36p2_score_min",
        "must_add_ratio_mean", "must_gain_mean", "semantic_trap_rate", "drop_seems_good_rate",
        "candidate_file",
    ]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))

    print("\nV36P2 veto counts:")
    print(swaps["v36p2_notes"].replace("", np.nan).value_counts(dropna=True).head(30).to_string())

    print("\noutputs:")
    print(OUT_SWAPS)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
