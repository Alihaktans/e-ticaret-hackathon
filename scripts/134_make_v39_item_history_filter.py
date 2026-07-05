from pathlib import Path
import re
import json
import time
import math
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"

V38 = ROOT / "data/processed/v38_lexical_pair_scores.parquet"
V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

BASELINE_PATHS = {
    "v24": [
        ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v24_v24_swap_rE_tE_b6500.csv",
    ],
    "raw_v35_b5000": [
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv",
    ],
    "v36p2_strict": [
        ROOT / "submissions/FINAL_MAIN_v36p2_strict.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v36p2_strict.csv",
    ],
    "v36p2_balanced": [
        ROOT / "submissions/FINAL_MAIN_v36p2_balanced.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v36p2_balanced.csv",
    ],
    # from V38 saved candidates if present
    "v38_precision_loose_2800": [
        ROOT / "submissions/FINAL_CANDIDATE_v38_v38_raw_v35_b5000_v38_precision_score_loose_cap2800.csv",
        ROOT / "submissions/sleep_candidates/A_v38_precision_loose_cap2800.csv",
    ],
    "v38_veto_bad_only": [
        ROOT / "submissions/FINAL_CANDIDATE_v38_v38_raw_v35_b5000_veto_bad_only.csv",
        ROOT / "submissions/sleep_candidates/B_v38_veto_bad_only.csv",
    ],
}

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
    "v26_sparse": ROOT / "reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv",
}

OUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_SCORE = OUT_DIR / "v39_item_history_pair_scores.parquet"
OUT_SCORE_SUMMARY = OUT_DIR / "v39_item_history_score_summary.json"

OUT_SUMMARY = REPORT_DIR / "v39_item_history_candidate_summary.csv"
OUT_EVAL = REPORT_DIR / "v39_item_history_candidate_eval.csv"
OUT_SAVED = REPORT_DIR / "v39_item_history_saved_candidates.csv"
OUT_SWAPS = REPORT_DIR / "v39_item_history_swap_diagnostics.csv"
OUT_REVIEW = REPORT_DIR / "review_v39_item_history_candidates_sample.csv"


TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

STOP = {
    "ve", "ile", "icin", "bir", "adet", "set", "takim", "model", "uyumlu",
    "orjinal", "orijinal", "yeni", "renk", "boy", "numara", "beden",
    "cm", "mm", "lt", "kg", "gr", "ml", "x", "no", "olan", "veya",
    "plus", "pro", "max", "mini", "buyuk", "kucuk", "orta", "sade",
    "renkli", "urun", "da", "de", "mi", "mı", "mu", "mü",
}

COLOR_WORDS = {
    "siyah", "beyaz", "kirmizi", "mavi", "lacivert", "yesil", "sari",
    "pembe", "mor", "turuncu", "gri", "antrasit", "kahverengi", "bej",
    "krem", "gold", "gumus", "silver", "bordo", "lila", "ekru",
}

GENDER_AGE_WORDS = {
    "erkek", "kadin", "bayan", "kiz", "cocuk", "bebek", "unisex",
    "genc", "yetiskin",
}

PRODUCT_TYPE_WORDS = {
    "ayakkabi", "bot", "cizme", "sneaker", "terlik", "sandalet", "babet",
    "loafer", "hali", "saha", "pantolon", "gomlek", "elbise", "etek",
    "kazak", "mont", "ceket", "tshirt", "tisort", "jean", "tayt", "sort",
    "sweatshirt", "bluz", "hirka", "canta", "valiz", "bavul", "cuzdan",
    "telefon", "cep", "tablet", "laptop", "bilgisayar", "kulaklik",
    "kamera", "fotograf", "kilif", "kapak", "ekran", "koruyucu", "sarj",
    "kablo", "adaptor", "adapter", "lastik", "jant", "oto", "arac",
    "araba", "motosiklet", "paspas", "silecek", "cam", "suyu", "antifriz",
    "krem", "fondoten", "ruj", "sampuan", "parfum", "sac", "cilt",
    "serum", "wax", "masa", "sandalye", "koltuk", "perde", "dolap",
    "sehpa", "tablo", "kitap", "defter", "kalem", "oyuncak", "cikolata",
    "seker", "boncuk", "tesbih", "ram", "ssd", "ddr", "ddr4", "ddr5",
    "playstation", "ps2", "ps3", "ps4", "ps5", "makinesi", "makina",
    "cihazi", "cihaz", "aksesuar", "askisi", "tokasi", "toka",
}


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def toks(s, keep_stop=False):
    s = norm_text(s)
    if keep_stop:
        return {t for t in s.split() if len(t) >= 2}
    return {t for t in s.split() if len(t) >= 2 and t not in STOP}


def chargrams(s, max_grams=90):
    s = norm_text(s)
    if not s:
        return tuple()
    s = " " + s + " "
    out = []
    seen = set()
    for n in (3, 4, 5):
        if len(s) >= n:
            for i in range(len(s) - n + 1):
                g = s[i:i+n]
                if g not in seen:
                    seen.add(g)
                    out.append(g)
                    if len(out) >= max_grams:
                        return tuple(out)
    return tuple(out)


def number_tokens(s):
    return set(re.findall(r"\d+", norm_text(s)))


def model_tokens(tokens):
    out = set()
    for t in tokens:
        if re.search(r"[a-z]", t) and re.search(r"\d", t):
            out.add(t)
        if re.search(r"\d", t) and len(t) >= 2:
            out.add(t)
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def clean_name(x):
    x = (
        str(x)
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("=", "")
        .replace("+", "plus")
        .replace("-", "m")
    )
    return x[:170]


def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def align_parquet(path, sample, cols):
    d = pd.read_parquet(path, columns=["id"] + cols)
    d["id"] = d["id"].astype(str)
    if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        return d[["id"] + cols].copy()
    return sample[["id"]].merge(d[["id"] + cols], on="id", how="left", validate="one_to_one")


def coverage(a, b):
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def weighted_coverage(a, b, idf):
    if not a:
        return 0.0
    denom = 0.0
    hit = 0.0
    for t in a:
        w = float(idf.get(t, 1.0))
        denom += w
        if t in b:
            hit += w
    return hit / max(1e-9, denom)


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

        if label_set in ["v20_active", "v21_active", "v26_sparse"] and "review_bucket" in lab.columns:
            for b, g in lab.groupby("review_bucket"):
                if len(g) >= 20 and g["assistant_label"].nunique() >= 2:
                    subsets[f"bucket_{b}"] = g.copy()

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
    main_keys = {
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "manual_v1_high_medium_clean",
        "review_v15_vs_v13_clean",
        "review_v15_vs_v13_high_medium_clean",
    }

    rows = []
    if len(eval_df) == 0:
        return pd.DataFrame()

    for name, g in eval_df.groupby("variant"):
        val = 0.0
        wsum = 0.0
        used = []
        main = []
        for _, r in g.iterrows():
            k = r["eval_key"]
            w = weights.get(k, 0.0)
            if w:
                val += w * float(r["macro_f1"])
                wsum += w
                used.append(float(r["macro_f1"]))
            if k in main_keys:
                main.append(float(r["macro_f1"]))
        if wsum == 0:
            continue

        rows.append({
            "variant": name,
            "weighted_macro": float(val / wsum),
            "used_min_macro": float(np.min(used)) if used else np.nan,
            "main_min_macro": float(np.min(main)) if main else np.nan,
            "main_mean_macro": float(np.mean(main)) if main else np.nan,
            "eval_count": int(len(g)),
            "mean_precision": float(g["precision"].mean()),
            "mean_recall": float(g["recall"].mean()),
        })

    return pd.DataFrame(rows)


def build_term_info(terms):
    info = {}
    for r in terms[["term_id", "query"]].itertuples(index=False):
        q_norm = norm_text(r.query)
        qt = toks(q_norm)
        q_all = toks(q_norm, keep_stop=True)
        must = set(qt) - STOP - COLOR_WORDS - GENDER_AGE_WORDS
        special = set(must) - PRODUCT_TYPE_WORDS
        nums = number_tokens(q_norm)
        models = model_tokens(q_all)

        info[str(r.term_id)] = {
            "query": str(r.query),
            "q_norm": q_norm,
            "tokens": qt,
            "must": must,
            "special": special,
            "nums": nums,
            "models": models,
            "grams": set(chargrams(q_norm)),
        }
    return info


def build_item_history(train_pairs, term_info):
    # item -> positive training query signature
    item_tokens = defaultdict(Counter)
    item_queries = defaultdict(set)
    item_grams = defaultdict(set)
    item_nums = defaultdict(set)
    item_models = defaultdict(set)
    item_count = Counter()

    print("building item history from train positives...")

    used = 0
    for r in train_pairs[["term_id", "item_id"]].itertuples(index=False):
        tid = str(r.term_id)
        iid = str(r.item_id)
        qi = term_info.get(tid)
        if qi is None:
            continue

        used += 1
        item_count[iid] += 1
        item_queries[iid].add(qi["q_norm"])
        item_nums[iid].update(qi["nums"])
        item_models[iid].update(qi["models"])

        # Query tokens are behavioral signal; keep frequencies.
        for t in qi["tokens"]:
            item_tokens[iid][t] += 1
        for t in qi["must"]:
            item_tokens[iid][t] += 1
        for g in list(qi["grams"])[:80]:
            item_grams[iid].add(g)

    # Token idf over item histories
    df = Counter()
    for iid, cnt in item_tokens.items():
        for t in cnt.keys():
            df[t] += 1

    n_items = max(1, len(item_tokens))
    idf = {t: math.log(1.0 + (1.0 + n_items) / (1.0 + d)) + 1.0 for t, d in df.items()}

    hist = {}
    for iid, cnt in item_tokens.items():
        tok_set = set(cnt.keys())
        hist[iid] = {
            "tokens": tok_set,
            "queries": item_queries.get(iid, set()),
            "grams": item_grams.get(iid, set()),
            "nums": item_nums.get(iid, set()),
            "models": item_models.get(iid, set()),
            "count": int(item_count.get(iid, 0)),
        }

    print("train rows used:", used)
    print("items with history:", len(hist))
    print("idf tokens:", len(idf))

    return hist, idf


def score_history_pairs(sample, pairs, term_info, item_hist, idf):
    t0 = time.time()
    n = len(pairs)

    arr = {
        "v39_hist_score": np.zeros(n, dtype=np.float32),
        "v39_hist_has_item": np.zeros(n, dtype=np.int8),
        "v39_hist_count": np.zeros(n, dtype=np.int16),
        "v39_hist_token_cov": np.zeros(n, dtype=np.float32),
        "v39_hist_weighted_cov": np.zeros(n, dtype=np.float32),
        "v39_hist_must_cov": np.zeros(n, dtype=np.float32),
        "v39_hist_special_cov": np.zeros(n, dtype=np.float32),
        "v39_hist_exact_query": np.zeros(n, dtype=np.float32),
        "v39_hist_char_cov": np.zeros(n, dtype=np.float32),
        "v39_hist_number_match": np.zeros(n, dtype=np.float32),
        "v39_hist_model_match": np.zeros(n, dtype=np.float32),
    }

    term_ids = pairs["term_id"].astype(str).to_numpy()
    item_ids = pairs["item_id"].astype(str).to_numpy()

    print("scoring item-history pairs:", n)

    for i, (tid, iid) in enumerate(zip(term_ids, item_ids)):
        qi = term_info.get(str(tid))
        hi = item_hist.get(str(iid))
        if qi is None or hi is None:
            continue

        htokens = hi["tokens"]

        token_cov = coverage(qi["tokens"], htokens)
        weighted_cov = weighted_coverage(qi["tokens"], htokens, idf)
        must_cov = coverage(qi["must"], htokens)
        special_cov = coverage(qi["special"], htokens)

        exact = 1.0 if qi["q_norm"] in hi["queries"] else 0.0

        if qi["grams"]:
            char_cov = len(qi["grams"] & hi["grams"]) / max(1, len(qi["grams"]))
        else:
            char_cov = 0.0

        num_match = coverage(qi["nums"], hi["nums"])
        model_match = coverage(qi["models"], hi["models"])

        count_boost = min(1.0, math.log1p(hi["count"]) / math.log1p(8.0))

        score = (
            0.36 * weighted_cov +
            0.12 * token_cov +
            0.13 * must_cov +
            0.10 * special_cov +
            0.13 * exact +
            0.08 * char_cov +
            0.04 * num_match +
            0.03 * model_match +
            0.01 * count_boost
        )
        score = float(np.clip(score, 0.0, 1.15))

        arr["v39_hist_score"][i] = score
        arr["v39_hist_has_item"][i] = 1
        arr["v39_hist_count"][i] = min(32767, hi["count"])
        arr["v39_hist_token_cov"][i] = token_cov
        arr["v39_hist_weighted_cov"][i] = weighted_cov
        arr["v39_hist_must_cov"][i] = must_cov
        arr["v39_hist_special_cov"][i] = special_cov
        arr["v39_hist_exact_query"][i] = exact
        arr["v39_hist_char_cov"][i] = char_cov
        arr["v39_hist_number_match"][i] = num_match
        arr["v39_hist_model_match"][i] = model_match

        if (i + 1) % 250_000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"scored {i+1:,}/{n:,} elapsed_min={elapsed:.2f}")

    out = pd.DataFrame({
        "id": pairs["id"].astype(str).to_numpy(),
        "term_id": pairs["term_id"].astype(str).to_numpy(),
        "item_id": pairs["item_id"].astype(str).to_numpy(),
    })

    for c, v in arr.items():
        out[c] = v

    print("ranking v39 history within term...")
    out["v39_hist_rank"] = (
        out.groupby("term_id")["v39_hist_score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )

    cnt = out.groupby("term_id")["id"].transform("count").astype(np.float32)
    out["v39_hist_pct_rank"] = ((out["v39_hist_rank"].astype(np.float32) - 1.0) / np.maximum(1.0, cnt - 1.0)).astype(np.float32)

    term_mean = out.groupby("term_id")["v39_hist_score"].transform("mean").astype(np.float32)
    term_std = out.groupby("term_id")["v39_hist_score"].transform("std").fillna(0).astype(np.float32)
    out["v39_hist_term_z"] = ((out["v39_hist_score"].astype(np.float32) - term_mean) / np.maximum(1e-6, term_std)).astype(np.float32)

    if not out["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("v39 score output order mismatch sample")

    out.to_parquet(OUT_SCORE, index=False)

    summary = {
        "output": str(OUT_SCORE),
        "rows": int(len(out)),
        "unique_terms": int(out["term_id"].nunique()),
        "unique_items": int(out["item_id"].nunique()),
        "has_history_rate": float(out["v39_hist_has_item"].mean()),
        "score_min": float(out["v39_hist_score"].min()),
        "score_mean": float(out["v39_hist_score"].mean()),
        "score_std": float(out["v39_hist_score"].std()),
        "score_max": float(out["v39_hist_score"].max()),
        "exact_history_rate": float(out["v39_hist_exact_query"].mean()),
        "elapsed_min": float((time.time() - t0) / 60),
    }

    OUT_SCORE_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return out


def build_same_quota(df, score_col, anchor):
    tmp = pd.DataFrame({
        "term_id": df["term_id"].to_numpy(),
        "score": df[score_col].to_numpy(np.float32),
        "anchor": anchor,
    })

    quota = tmp.groupby("term_id")["anchor"].sum()
    tmp["quota"] = tmp["term_id"].map(quota).fillna(0).astype(np.int32)
    tmp["rank"] = tmp.groupby("term_id")["score"].rank(method="first", ascending=False).astype(np.int32)

    return (tmp["rank"].to_numpy() <= tmp["quota"].to_numpy()).astype(np.int8)


def make_score_cols(df, anchor):
    df["v39_rs"] = (1.0 - df["v39_hist_pct_rank"].astype(np.float32)).clip(0, 1)
    df["v39_zsig"] = sigmoid(df["v39_hist_term_z"].fillna(0).to_numpy(np.float32) / 1.7).astype(np.float32)

    df["v38_rs"] = (1.0 - df["v38_lex_pct_rank"].astype(np.float32)).clip(0, 1)
    df["v38_zsig"] = sigmoid(df["v38_lex_term_z"].fillna(0).to_numpy(np.float32) / 1.7).astype(np.float32)

    df["v33_rs"] = (1.0 - df["v33_ft_pct_rank"].astype(np.float32)).clip(0, 1)
    df["v34_rs"] = (1.0 - df["v34_ce_pct_rank"].astype(np.float32)).clip(0, 1)

    df["v33_zsig"] = sigmoid(df["v33_ft_term_z"].fillna(0).to_numpy(np.float32) / 1.8).astype(np.float32)
    df["v34_zsig"] = sigmoid(df["v34_ce_term_z"].fillna(0).to_numpy(np.float32) / 1.8).astype(np.float32)

    hist = df["v39_hist_score"].astype(np.float32)
    lex = df["v38_lex_score"].astype(np.float32)

    anchor_f = anchor.astype(np.float32)

    # Pure behavior/history ranking.
    df["v39_history_pure_score"] = (
        0.62 * df["v39_rs"] +
        0.15 * df["v39_zsig"] +
        0.13 * hist +
        0.04 * df["v39_hist_exact_query"].astype(np.float32) +
        0.03 * df["v39_hist_must_cov"].astype(np.float32) +
        0.03 * anchor_f
    ).astype(np.float32)

    # Product text + item history.
    df["v39_history_lex_score"] = (
        0.34 * df["v39_rs"] +
        0.32 * df["v38_rs"] +
        0.10 * df["v39_zsig"] +
        0.08 * df["v38_zsig"] +
        0.07 * hist +
        0.04 * df["v38_must_token_coverage_full"].astype(np.float32) +
        0.03 * df["v39_hist_exact_query"].astype(np.float32) +
        0.02 * anchor_f
    ).astype(np.float32)

    # History as a gate on semantic confidence.
    df["v39_history_sem_score"] = (
        0.24 * df["v39_rs"] +
        0.24 * df["v38_rs"] +
        0.22 * df["v34_rs"] +
        0.15 * df["v33_rs"] +
        0.05 * df["v39_zsig"] +
        0.04 * df["v38_zsig"] +
        0.03 * df["v39_hist_score"].astype(np.float32) +
        0.03 * anchor_f
    ).astype(np.float32)

    # Conservative: if history exists and is bad, it hurts; if no history, semantics still work.
    hist_gate = (
        0.70 * df["v39_hist_score"].astype(np.float32)
        + 0.30 * df["v39_hist_has_item"].astype(np.float32)
    )

    df["v39_history_gate_score"] = (
        0.30 * df["v34_rs"] +
        0.22 * df["v33_rs"] +
        0.22 * df["v38_rs"] +
        0.14 * df["v39_rs"] +
        0.06 * hist_gate +
        0.06 * anchor_f
    ).astype(np.float32)

    return df


def pair_swaps(df, anchor, cand, score_col):
    add_mask = (anchor == 0) & (cand == 1)
    drop_mask = (anchor == 1) & (cand == 0)

    cols = [
        "id", "term_id", "item_id", score_col,
        "v39_hist_score", "v39_hist_has_item", "v39_hist_count",
        "v39_hist_weighted_cov", "v39_hist_must_cov", "v39_hist_special_cov",
        "v39_hist_exact_query", "v39_hist_char_cov", "v39_hist_number_match", "v39_hist_model_match",
        "v38_lex_score", "v38_word_overlap_title", "v38_must_token_coverage_full",
        "v38_query_word_only_category_ratio",
        "v33_rs", "v34_rs", "v38_rs", "v39_rs",
    ]

    add = df.loc[add_mask, cols].copy()
    drop = df.loc[drop_mask, cols].copy()

    add = add.sort_values(["term_id", score_col], ascending=[True, False])
    drop = drop.sort_values(["term_id", score_col], ascending=[True, True])

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    sw = add.merge(drop, on=["term_id", "pair_rank"], suffixes=("_add", "_drop"), how="inner")
    sw["pair_score_gain"] = sw[f"{score_col}_add"] - sw[f"{score_col}_drop"]
    sw["hist_gain"] = sw["v39_hist_score_add"] - sw["v39_hist_score_drop"]
    sw["lex_gain"] = sw["v38_lex_score_add"] - sw["v38_lex_score_drop"]

    return sw.sort_values(["pair_score_gain", "hist_gain"], ascending=False).reset_index(drop=True)


def history_accept_mask(sw, mode):
    add_has = sw["v39_hist_has_item_add"].astype(int) == 1
    drop_has = sw["v39_hist_has_item_drop"].astype(int) == 1

    hist_support = (
        (sw["v39_hist_score_add"] >= 0.18)
        | (sw["v39_hist_weighted_cov_add"] >= 0.45)
        | (sw["v39_hist_exact_query_add"] == 1)
        | (sw["v39_hist_must_cov_add"] >= 0.70)
        | ((sw["v39_hist_number_match_add"] >= 1) & (sw["v39_hist_score_add"] >= 0.08))
        | ((sw["v39_hist_model_match_add"] >= 1) & (sw["v39_hist_score_add"] >= 0.08))
    )

    lex_sem_support = (
        (sw["v38_lex_score_add"] >= 0.25)
        | (sw["v38_word_overlap_title_add"] >= 0.45)
        | ((sw["v34_rs_add"] >= 0.82) & (sw["v33_rs_add"] >= 0.55))
        | ((sw["v39_rs_add"] >= 0.85) & add_has)
    )

    hist_contradiction = (
        add_has & drop_has
        & (sw["v39_hist_score_drop"] >= sw["v39_hist_score_add"] + 0.12)
        & (sw["v39_hist_weighted_cov_drop"] >= sw["v39_hist_weighted_cov_add"] + 0.18)
    )

    add_history_bad = (
        add_has
        & (sw["v39_hist_score_add"] <= 0.025)
        & (sw["v39_hist_weighted_cov_add"] <= 0.05)
        & (sw["v39_hist_must_cov_add"] <= 0.10)
        & (sw["v38_lex_score_add"] < 0.20)
    )

    category_only_lex_bad = (
        (sw["v38_query_word_only_category_ratio_add"] >= 0.50)
        & (sw["v38_word_overlap_title_add"] <= 0.10)
        & (~hist_support)
    )

    not_bad = ~(hist_contradiction | add_history_bad | category_only_lex_bad)

    if mode == "history_ultra":
        return hist_support & not_bad & (sw["hist_gain"] >= 0.05)
    if mode == "history_strict":
        return (hist_support | (lex_sem_support & (sw["hist_gain"] >= 0.02))) & not_bad
    if mode == "history_balanced":
        return ((hist_support | lex_sem_support) & not_bad) | ((~add_has) & lex_sem_support & (sw["pair_score_gain"] >= 0.10))
    if mode == "history_loose":
        return (~(hist_contradiction | category_only_lex_bad)) & ((hist_support | lex_sem_support) | (sw["pair_score_gain"] >= 0.16))
    raise ValueError(mode)


def apply_swaps(sample, base, sw, max_swaps=None):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])

    take = sw.sort_values(["pair_score_gain", "hist_gain", "lex_gain"], ascending=False).copy()
    if max_swaps is not None:
        take = take.head(max_swaps).copy()

    pred = base.copy()
    add_idx = take["id_add"].astype(str).map(id_to_idx)
    drop_idx = take["id_drop"].astype(str).map(id_to_idx)

    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("id mapping failed")

    pred[add_idx.astype(int).to_numpy()] = 1
    pred[drop_idx.astype(int).to_numpy()] = 0

    return pred, take


def build_review(chosen):
    parts = []
    for name, sw in chosen.items():
        if len(sw) == 0:
            continue
        x = sw.copy()
        x["variant"] = name
        parts.append(x.head(120))
        parts.append(x.tail(80))

    if not parts:
        return

    review = pd.concat(parts, ignore_index=True).drop_duplicates(["variant", "id_add", "id_drop"], keep="first")
    keep = [
        "variant", "term_id", "pair_rank",
        "id_add", "item_id_add", "id_drop", "item_id_drop",
        "pair_score_gain", "hist_gain", "lex_gain",
        "v39_hist_score_add", "v39_hist_score_drop",
        "v39_hist_weighted_cov_add", "v39_hist_weighted_cov_drop",
        "v39_hist_must_cov_add", "v39_hist_must_cov_drop",
        "v39_hist_special_cov_add", "v39_hist_special_cov_drop",
        "v39_hist_exact_query_add", "v39_hist_exact_query_drop",
        "v39_hist_has_item_add", "v39_hist_has_item_drop",
        "v39_hist_count_add", "v39_hist_count_drop",
        "v38_lex_score_add", "v38_lex_score_drop",
        "v38_word_overlap_title_add", "v38_word_overlap_title_drop",
        "v33_rs_add", "v33_rs_drop", "v34_rs_add", "v34_rs_drop",
    ]
    review[[c for c in keep if c in review.columns]].to_csv(OUT_REVIEW, index=False)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print("loading sample/pairs...")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("submission_pairs order mismatch sample_submission")

    # Score item-history signal if missing.
    if OUT_SCORE.exists():
        print("using existing v39 score:", OUT_SCORE)
        hist_scores = pd.read_parquet(OUT_SCORE)
        hist_scores["id"] = hist_scores["id"].astype(str)
        if not hist_scores["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            raise RuntimeError("existing v39 score order mismatch")
    else:
        print("loading terms/train positives...")
        terms = pd.read_csv(TERMS)
        terms["term_id"] = terms["term_id"].astype(str)

        train = pd.read_csv(TRAIN_PAIRS)
        if "term_id" not in train.columns or "item_id" not in train.columns:
            raise RuntimeError("training_pairs.csv must contain term_id,item_id")
        train["term_id"] = train["term_id"].astype(str)
        train["item_id"] = train["item_id"].astype(str)

        term_info = build_term_info(terms)
        item_hist, idf = build_item_history(train, term_info)
        hist_scores = score_history_pairs(sample, pairs, term_info, item_hist, idf)

    # Baselines
    anchor_path = first_existing(ANCHOR_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor v33 qprob2000 missing")

    anchor = load_pred(anchor_path, sample)
    print("anchor:", anchor_path, "ones:", int(anchor.sum()))

    baselines = {"anchor_v33_qprob2000": anchor.copy()}

    for name, paths in BASELINE_PATHS.items():
        p = first_existing(paths)
        if p is not None:
            baselines[name] = load_pred(p, sample)
            print("baseline:", name, p, "diff_vs_anchor:", int((baselines[name] != anchor).sum()))

    # Load external pair scores
    print("loading v38/v33/v34...")
    v38_cols = [
        "v38_lex_score", "v38_lex_pct_rank", "v38_lex_term_z",
        "v38_word_overlap_title", "v38_must_token_coverage_full",
        "v38_query_word_only_category_ratio",
    ]
    v38 = align_parquet(V38, sample, v38_cols)
    v33 = align_parquet(V33, sample, ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"])
    v34 = align_parquet(V34, sample, ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"])

    df = pairs.copy()

    hist_cols = [
        "v39_hist_score", "v39_hist_has_item", "v39_hist_count",
        "v39_hist_token_cov", "v39_hist_weighted_cov", "v39_hist_must_cov",
        "v39_hist_special_cov", "v39_hist_exact_query", "v39_hist_char_cov",
        "v39_hist_number_match", "v39_hist_model_match",
        "v39_hist_rank", "v39_hist_pct_rank", "v39_hist_term_z",
    ]

    for c in hist_cols:
        df[c] = pd.to_numeric(hist_scores[c], errors="coerce").fillna(0).astype(np.float32)

    for c in v38_cols:
        df[c] = pd.to_numeric(v38[c], errors="coerce").fillna(0).astype(np.float32)

    for c in ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"]:
        df[c] = pd.to_numeric(v33[c], errors="coerce").fillna(0).astype(np.float32)

    for c in ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"]:
        df[c] = pd.to_numeric(v34[c], errors="coerce").fillna(0).astype(np.float32)

    df = make_score_cols(df, anchor)

    variants = dict(baselines)
    meta_rows = []
    chosen_swaps = {}
    swap_diags = []

    # Same-quota reranks
    score_cols = [
        "v39_history_pure_score",
        "v39_history_lex_score",
        "v39_history_sem_score",
        "v39_history_gate_score",
    ]

    for sc in score_cols:
        pred = build_same_quota(df, sc, anchor)
        name = f"{sc}_same_quota"
        variants[name] = pred
        meta_rows.append({
            "variant": name,
            "source": "same_quota",
            "base": "anchor",
            "score_col": sc,
            "mode": "",
            "cap": "",
        })
        print("same quota:", name, "diff_vs_anchor:", int((pred != anchor).sum()))

    # Filter/veto swap candidates
    swap_bases = [x for x in [
        "raw_v35_b5000",
        "v38_precision_loose_2800",
        "v38_veto_bad_only",
        "v36p2_balanced",
        "v36p2_strict",
    ] if x in baselines]

    caps = {
        "history_ultra": [500, 900, 1300],
        "history_strict": [900, 1500, 2200],
        "history_balanced": [1400, 2400, 3400],
        "history_loose": [1800, 3000, 4500],
    }

    for base_name in swap_bases:
        base_pred = baselines[base_name]

        for sc in ["v39_history_gate_score", "v39_history_sem_score", "v39_history_lex_score", "v39_history_pure_score"]:
            sw = pair_swaps(df, anchor, base_pred, sc)
            sw["base_name"] = base_name
            sw["score_col"] = sc

            diag = sw.head(1500).copy()
            diag["candidate_source"] = f"{base_name}_{sc}"
            swap_diags.append(diag)

            for mode, cap_list in caps.items():
                m = history_accept_mask(sw, mode)
                accepted = sw[m].copy().sort_values(["pair_score_gain", "hist_gain", "lex_gain"], ascending=False)

                for cap in cap_list:
                    if len(accepted) == 0:
                        continue

                    pred, take = apply_swaps(sample, anchor, accepted, max_swaps=cap)
                    name = f"v39_{base_name}_{sc}_{mode}_cap{cap}"
                    variants[name] = pred
                    chosen_swaps[name] = take

                    meta_rows.append({
                        "variant": name,
                        "source": "history_filtered_swaps",
                        "base": base_name,
                        "score_col": sc,
                        "mode": mode,
                        "cap": cap,
                    })
                    print("filtered:", name, "accepted:", len(take), "diff_vs_anchor:", int((pred != anchor).sum()))

            # Veto-only: keep all base swaps except history contradiction.
            add_has = sw["v39_hist_has_item_add"].astype(int) == 1
            drop_has = sw["v39_hist_has_item_drop"].astype(int) == 1
            contradiction = (
                add_has & drop_has
                & (sw["v39_hist_score_drop"] >= sw["v39_hist_score_add"] + 0.12)
                & (sw["v39_hist_weighted_cov_drop"] >= sw["v39_hist_weighted_cov_add"] + 0.18)
            )
            add_bad = (
                add_has
                & (sw["v39_hist_score_add"] <= 0.02)
                & (sw["v38_lex_score_add"] < 0.18)
                & (sw["v34_rs_add"] < 0.70)
            )

            for vmode, keep_mask in {
                "history_veto_contradiction": ~contradiction,
                "history_veto_contradiction_and_bad": ~(contradiction | add_bad),
            }.items():
                kept = sw[keep_mask].copy().sort_values(["pair_score_gain", "hist_gain"], ascending=False)
                pred, take = apply_swaps(sample, anchor, kept, max_swaps=None)

                name = f"v39_{base_name}_{sc}_{vmode}"
                variants[name] = pred
                chosen_swaps[name] = take

                meta_rows.append({
                    "variant": name,
                    "source": "history_veto",
                    "base": base_name,
                    "score_col": sc,
                    "mode": vmode,
                    "cap": "all",
                })
                print("veto:", name, "kept:", len(take), "diff_vs_anchor:", int((pred != anchor).sum()))

    print("evaluating...")
    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)

    w = weighted_score(eval_df)

    summary = pd.DataFrame(meta_rows)
    for name, pred in baselines.items():
        summary = pd.concat([summary, pd.DataFrame([{
            "variant": name,
            "source": "baseline",
            "base": "",
            "score_col": "",
            "mode": "",
            "cap": "",
        }])], ignore_index=True)

    aux = []
    for name, pred in variants.items():
        aux.append({
            "variant": name,
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_v24": int((pred != baselines["v24"]).sum()) if "v24" in baselines else -1,
            "diff_vs_raw_v35": int((pred != baselines["raw_v35_b5000"]).sum()) if "raw_v35_b5000" in baselines else -1,
            "diff_vs_v38_precision_loose_2800": int((pred != baselines["v38_precision_loose_2800"]).sum()) if "v38_precision_loose_2800" in baselines else -1,
            "diff_vs_v36p2_strict": int((pred != baselines["v36p2_strict"]).sum()) if "v36p2_strict" in baselines else -1,
            "diff_vs_v36p2_balanced": int((pred != baselines["v36p2_balanced"]).sum()) if "v36p2_balanced" in baselines else -1,
        })

    summary = summary.merge(pd.DataFrame(aux), on="variant", how="right")
    if len(w):
        summary = summary.merge(w, on="variant", how="left")

    diff = summary["diff_vs_anchor"].fillna(0).astype(float)
    movement_bonus = np.minimum(0.020, np.log1p(diff) / np.log1p(25000) * 0.020)
    too_big_penalty = np.maximum(0, diff - 55000) / 950000.0

    # History is a filter, not a promise. Local quality still dominates.
    summary["v39_decision_score"] = summary["weighted_macro"].fillna(0) + movement_bonus - too_big_penalty

    summary = summary.sort_values(["v39_decision_score", "weighted_macro"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if swap_diags:
        pd.concat(swap_diags, ignore_index=True).to_csv(OUT_SWAPS, index=False)

    build_review(chosen_swaps)

    # Save top candidates.
    save = []
    top = summary[
        (summary["source"].isin(["same_quota", "history_filtered_swaps", "history_veto"]))
        & (summary["diff_vs_anchor"] >= 1500)
        & (summary["diff_vs_anchor"] <= 60000)
    ].head(35)

    for nm in top["variant"].tolist():
        if nm not in save:
            save.append(nm)

    for nm in ["anchor_v33_qprob2000", "raw_v35_b5000", "v38_precision_loose_2800", "v38_veto_bad_only", "v36p2_strict", "v36p2_balanced"]:
        if nm in variants and nm not in save:
            save.append(nm)

    saved_rows = []
    for nm in save[:45]:
        pred = variants[nm]
        out = SUB_DIR / f"FINAL_CANDIDATE_v39_{clean_name(nm)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

        row = summary[summary["variant"] == nm].iloc[0].to_dict()
        row["file"] = str(out)
        saved_rows.append(row)
        print("saved:", out)

    pd.DataFrame(saved_rows).to_csv(OUT_SAVED, index=False)

    print("\nTOP V39")
    cols = [
        "variant", "v39_decision_score", "weighted_macro", "source", "base", "score_col", "mode", "cap",
        "ones", "pos_ratio", "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v38_precision_loose_2800",
        "main_min_macro", "main_mean_macro", "mean_precision", "mean_recall",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))

    print("\nSAVED")
    saved_df = pd.DataFrame(saved_rows)
    if len(saved_df):
        print(saved_df[["variant", "file", "v39_decision_score", "weighted_macro", "diff_vs_anchor"]].to_string(index=False))

    print("\noutputs:")
    print(OUT_SCORE)
    print(OUT_SCORE_SUMMARY)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_SAVED)
    print(OUT_SWAPS)
    print(OUT_REVIEW)
    print("elapsed_min:", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
