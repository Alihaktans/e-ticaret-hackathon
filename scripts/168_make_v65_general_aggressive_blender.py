from pathlib import Path
import re
import hashlib
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
V64_REVIEW = ROOT / "reports/manual_review/review_v64_qwen25_v60_swaps.csv"
V60_REVIEW = ROOT / "reports/manual_review/review_v60_public080_bge_swaps.csv"
V60_CAP250 = ROOT / "submissions/final_candidates_v60/v60_public080_ultra_cap250.csv"

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v65"

OUT_POOL = OUT_DIR / "review_v65_general_aggressive_pool.csv"
OUT_PICKED = OUT_DIR / "review_v65_general_aggressive_picked.csv"
OUT_SUMMARY = OUT_DIR / "v65_general_aggressive_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v65_general_aggressive_saved_candidates.csv"

TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

ACCESSORY_WORDS = {
    "kilif", "kılıf", "kordon", "kayis", "kayış", "stand", "tutucu", "tutacagi", "tutacağı",
    "aksesuar", "aksesuari", "yedek", "torba", "torbasi", "torbası", "kapak", "minder",
    "kulpu", "ayagi", "ayağı", "susu", "süsü", "askisi", "askısı", "parca", "parça",
}

QUERY_ACCESSORY_WORDS = ACCESSORY_WORDS | {
    "mousepad", "anahtarlik", "anahtarlık", "halkasi", "halkası", "bicaklari", "bıçakları",
    "bıçak", "bicak", "paspas", "canta", "çanta", "cantasi", "çantası",
}

SOFT_STOP = {
    "ve", "ile", "icin", "için", "bir", "adet", "set", "takim", "takım", "model",
    "uyumlu", "orjinal", "orijinal", "yeni", "renk", "boy", "numara", "beden",
    "cm", "mm", "lt", "kg", "gr", "ml", "x", "no", "olan", "veya", "plus",
    "pro", "max", "mini", "urun", "ürün", "da", "de", "mi", "mu", "mü",
    "siyah", "beyaz", "kirmizi", "kırmızı", "mavi", "yesil", "yeşil", "bej",
    "gri", "lacivert", "kahverengi", "sari", "sarı", "pembe",
    "erkek", "kadin", "kadın", "bayan", "cocuk", "çocuk", "bebek", "unisex",
}

MODEL_HINT_COLS_HIGHER = [
    "bge_gain", "robust_gain", "pair_prob", "v55_prob", "root_gain",
    "signal_win_count", "rank_win_count",
    "diff_v38_lex_score", "pct_diff_v38_lex_score",
    "diff_v33_ft_score", "pct_diff_v33_ft_score",
    "diff_model_support", "pct_diff_model_support",
    "diff_v47_mutual_score", "pct_diff_v47_mutual_score",
    "diff_v39_hist_score", "pct_diff_v39_hist_score",
    "diff_v38_word_overlap_full", "diff_v38_must_token_coverage_full",
    "diff_v38_special_token_coverage_full", "diff_v38_char_ngram_sim",
    "diff_v47_reverse_base", "diff_v47_forward_base",
    "diff_v38_exact_phrase_title", "diff_v38_word_overlap_title",
    "diff_v38_number_match", "diff_v38_model_match", "diff_v38_brand_match",
]

MODEL_HINT_COLS_LOWER_BETTER = [
    "diff_v38_must_missing_ratio",
    "diff_v38_title_missing_ratio",
]

ADD_QUALITY_COLS = [
    "bge_add", "robust_row_score_add", "v38_lex_score_pct_add", "v33_ft_score_pct_add",
    "model_support_pct_add", "v47_mutual_score_pct_add", "v39_hist_score_pct_add",
    "v38_word_overlap_full_add", "v38_must_token_coverage_full_add",
    "v38_special_token_coverage_full_add", "v38_char_ngram_sim_add",
    "v38_exact_phrase_title_add", "v38_word_overlap_title_add",
    "v38_number_match_add", "v38_model_match_add", "v38_brand_match_add",
    "root_prob_add",
]

DROP_QUALITY_COLS = [
    "bge_drop", "robust_row_score_drop", "v38_lex_score_pct_drop", "v33_ft_score_pct_drop",
    "model_support_pct_drop", "v47_mutual_score_pct_drop", "v39_hist_score_pct_drop",
    "v38_word_overlap_full_drop", "v38_must_token_coverage_full_drop",
    "v38_special_token_coverage_full_drop", "v38_char_ngram_sim_drop",
    "v38_exact_phrase_title_drop", "v38_word_overlap_title_drop",
    "v38_number_match_drop", "v38_model_match_drop", "v38_brand_match_drop",
    "root_prob_drop",
]


def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def tokens(x):
    return [t for t in norm(x).split() if len(t) >= 2]


def content_tokens(x):
    return [t for t in tokens(x) if t not in {norm(s) for s in SOFT_STOP} and not t.isdigit() and len(t) >= 3]


def find_col(df, names, contains=None):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    if contains:
        for pat in contains:
            for c in df.columns:
                if pat.lower() in c.lower():
                    return c
    return None


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


def to_bool(s):
    if isinstance(s, bool):
        return s
    if pd.isna(s):
        return False
    return str(s).strip().lower() in {"true", "1", "yes", "evet"}


def safe_num(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype("float64")


def pct01(x):
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index, dtype="float64")
    s = s.fillna(s.median())
    if float(s.std()) < 1e-12:
        return pd.Series(0.5, index=s.index, dtype="float64")
    return s.rank(method="average", pct=True).astype("float64")


def robust_mean(cols):
    if not cols:
        return None
    m = pd.concat(cols, axis=1)
    return m.mean(axis=1).astype("float64")


def get_text(row, col):
    if col and col in row.index and not pd.isna(row[col]):
        return str(row[col])
    return ""


def lexical_general_penalties(df, cols):
    penalties = []
    reasons = []

    for _, row in df.iterrows():
        q = get_text(row, cols["query"])
        add_title = get_text(row, cols["title_add"])
        add_cat = get_text(row, cols["category_add"])
        add_text = norm(add_title + " " + add_cat)
        q_norm = norm(q)
        q_content = content_tokens(q)
        reason = []
        penalty = 0.0

        if q_content:
            hits = sum(1 for t in q_content if t in add_text)
            coverage = hits / max(1, len(q_content))
            if coverage < 0.34:
                penalty += 0.30
                reason.append("low_query_token_coverage")
            # Last meaningful noun/token is often the product type. Penalize if missing.
            tail = q_content[-1]
            if tail not in add_text:
                penalty += 0.18
                reason.append("tail_product_token_missing:" + tail)

        add_has_accessory = any(norm(w) in add_text for w in ACCESSORY_WORDS)
        query_has_accessory = any(norm(w) in q_norm for w in QUERY_ACCESSORY_WORDS)
        if add_has_accessory and not query_has_accessory:
            # Generic accessory-vs-main guard, not product-specific.
            penalty += 0.45
            reason.append("add_accessory_query_not_accessory")

        # If query has a compact number token, the add should preserve at least one important number.
        q_nums = set(re.findall(r"\b\d+[a-z]*\b", q_norm))
        a_nums = set(re.findall(r"\b\d+[a-z]*\b", add_text))
        important_num_ctx = any(w in q_norm for w in ["gb", "tb", "jant", "cm", "mm", "sinif", "sınıf", "w30", "mah", "hz"])
        if q_nums and important_num_ctx and not (q_nums & a_nums):
            penalty += 0.35
            reason.append("important_number_not_preserved")

        penalties.append(penalty)
        reasons.append("|".join(reason))

    return pd.Series(penalties, index=df.index), pd.Series(reasons, index=df.index)


def load_inputs():
    if not V64_REVIEW.exists():
        raise FileNotFoundError(V64_REVIEW)
    if not V60_CAP250.exists():
        raise FileNotFoundError(V60_CAP250)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    v60 = pd.read_csv(V60_CAP250)
    v60["id"] = v60["id"].astype(str)
    if not v60["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("V60 cap250 id order mismatch")

    df = pd.read_csv(V64_REVIEW)
    return sample, v60, df


def prepare_pool(df):
    cols = {
        "query": find_col(df, ["query_x", "query", "dst_query_real"], contains=["query"]),
        "id_add": find_col(df, ["id_add", "id_add_str", "add_id"]),
        "id_drop": find_col(df, ["id_drop", "id_drop_str", "drop_id"]),
        "title_add": find_col(df, ["title_add_x", "title_add", "add_title", "title_add_y"], contains=["title_add"]),
        "title_drop": find_col(df, ["title_drop_x", "title_drop", "drop_title", "title_drop_y"], contains=["title_drop"]),
        "category_add": find_col(df, ["category_add", "category_add_x", "add_category"], contains=["category_add"]),
        "category_drop": find_col(df, ["category_drop", "category_drop_x", "drop_category"], contains=["category_drop"]),
    }
    for k in ["query", "id_add", "id_drop"]:
        if cols[k] is None:
            raise RuntimeError(f"missing {k}; columns={list(df.columns)}")

    out = df.copy()
    out["id_add_str"] = out[cols["id_add"]].astype(str)
    out["id_drop_str"] = out[cols["id_drop"]].astype(str)
    out["v65_order"] = np.arange(len(out), dtype=np.int32)

    # Qwen score, kept as a teacher signal but not the only model.
    conf = safe_num(out, "confidence", 0.0).clip(0, 1)
    winner = out.get("winner", pd.Series("", index=out.index)).astype(str).str.upper()
    a_rel = out.get("a_relevant", pd.Series(False, index=out.index)).map(to_bool)
    b_rel = out.get("b_relevant", pd.Series(False, index=out.index)).map(to_bool)
    hard_guard = out.get("hard_guard_reasons", pd.Series("", index=out.index)).fillna("").astype(str)

    qwen_score = pd.Series(0.0, index=out.index, dtype="float64")
    qwen_score += ((winner == "A") & a_rel & (~b_rel)).astype(float) * (0.75 + 0.55 * conf)
    qwen_score += ((winner == "A") & a_rel & b_rel).astype(float) * (0.30 + 0.25 * conf)
    qwen_score += ((winner == "UNCERTAIN") & a_rel & (~b_rel)).astype(float) * (0.15 + 0.25 * conf)
    qwen_score += ((winner == "TIE") & a_rel).astype(float) * (0.05 + 0.10 * conf)
    qwen_score -= ((winner == "B") | b_rel).astype(float) * (0.65 + 0.35 * conf)
    qwen_score -= (hard_guard != "").astype(float) * 0.75
    out["v65_qwen_teacher_score"] = qwen_score

    # General feature consensus from all existing numeric models.
    gain_pcts = []
    for c in MODEL_HINT_COLS_HIGHER:
        if c in out.columns:
            gain_pcts.append(pct01(safe_num(out, c, 0.0)))
    for c in MODEL_HINT_COLS_LOWER_BETTER:
        if c in out.columns:
            gain_pcts.append(pct01(-safe_num(out, c, 0.0)))
    out["v65_model_gain_score"] = robust_mean(gain_pcts) if gain_pcts else 0.5

    add_pcts = [pct01(safe_num(out, c, 0.0)) for c in ADD_QUALITY_COLS if c in out.columns]
    drop_pcts = [pct01(safe_num(out, c, 0.0)) for c in DROP_QUALITY_COLS if c in out.columns]
    out["v65_add_quality"] = robust_mean(add_pcts) if add_pcts else 0.5
    out["v65_drop_quality"] = robust_mean(drop_pcts) if drop_pcts else 0.5
    out["v65_drop_badness"] = 1.0 - out["v65_drop_quality"]

    lex_pen, lex_reason = lexical_general_penalties(out, cols)
    out["v65_general_penalty"] = lex_pen
    out["v65_general_penalty_reasons"] = lex_reason

    # Final utility. This is intentionally not a hard product-rule model.
    out["v65_swap_utility"] = (
        0.42 * out["v65_qwen_teacher_score"]
        + 0.28 * out["v65_model_gain_score"]
        + 0.16 * out["v65_add_quality"]
        + 0.10 * out["v65_drop_badness"]
        + 0.04 * pct01(safe_num(out, "signal_win_count", 0.0))
        - 0.55 * out["v65_general_penalty"]
    )

    out["v65_plus_utility"] = (
        0.46 * out["v65_qwen_teacher_score"]
        + 0.34 * out["v65_add_quality"]
        + 0.16 * out["v65_model_gain_score"]
        - 0.65 * out["v65_general_penalty"]
    )

    out["v65_minus_utility"] = (
        0.52 * ((~b_rel).astype(float) * (0.5 + conf))
        + 0.28 * out["v65_drop_badness"]
        + 0.16 * out["v65_model_gain_score"]
        - 0.15 * out["v65_general_penalty"]
    )

    out["v65_qwen_clear_a"] = ((winner == "A") & a_rel & (~b_rel) & (conf >= 0.60) & (hard_guard == ""))
    out["v65_qwen_soft_a"] = ((winner == "A") & a_rel & (conf >= 0.55) & (hard_guard == ""))
    out["v65_model_strong"] = (
        (out["v65_model_gain_score"] >= out["v65_model_gain_score"].quantile(0.62))
        & (out["v65_add_quality"] >= out["v65_add_quality"].quantile(0.50))
        & (out["v65_general_penalty"] <= 0.25)
    )

    # Modes: increasing aggressiveness without becoming rule-only.
    out["v65_mode_ultra"] = (
        out["v65_qwen_clear_a"]
        & (conf >= 0.80)
        & (out["v65_general_penalty"] <= 0.05)
        & (out["v65_swap_utility"] >= out["v65_swap_utility"].quantile(0.70))
    )
    out["v65_mode_safe"] = (
        out["v65_qwen_clear_a"]
        & (out["v65_general_penalty"] <= 0.18)
        & (out["v65_swap_utility"] >= out["v65_swap_utility"].quantile(0.58))
    )
    out["v65_mode_aggressive"] = (
        ((out["v65_qwen_clear_a"] & (out["v65_swap_utility"] >= out["v65_swap_utility"].quantile(0.42)))
         | (out["v65_qwen_soft_a"] & out["v65_model_strong"] & (out["v65_swap_utility"] >= out["v65_swap_utility"].quantile(0.50))))
        & (out["v65_general_penalty"] <= 0.35)
    )
    out["v65_mode_model_boost"] = (
        out["v65_model_strong"]
        & (out["v65_qwen_teacher_score"] > 0.15)
        & (out["v65_general_penalty"] <= 0.25)
        & (out["v65_swap_utility"] >= out["v65_swap_utility"].quantile(0.55))
    )

    return out, cols


def reconstruct_base(sample, v60, pool):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    pred_v60 = v60["prediction"].astype(np.int8).to_numpy()
    base = pred_v60.copy()

    applied = []
    for i, r in pool.iterrows():
        aid = str(r["id_add_str"])
        did = str(r["id_drop_str"])
        if aid not in id_to_idx.index or did not in id_to_idx.index:
            continue
        ai = int(id_to_idx.loc[aid])
        di = int(id_to_idx.loc[did])
        # Only undo swaps actually present in V60 cap250.
        if pred_v60[ai] == 1 and pred_v60[di] == 0:
            base[ai] = 0
            base[di] = 1
            applied.append(i)

    print("V60 applied swaps detected:", len(applied))
    print("base ones:", int(base.sum()), "v60 ones:", int(pred_v60.sum()), "diff base-v60:", int((base != pred_v60).sum()))
    return base, pred_v60, id_to_idx


def unique_sorted(pool, flag_col, score_col, base, id_to_idx, require_add0=True, require_drop1=True):
    cand = pool[pool[flag_col].eq(True)].copy()
    cand = cand.sort_values([score_col, "v65_order"], ascending=[False, True])

    rows = []
    used_add = set()
    used_drop = set()
    for _, r in cand.iterrows():
        aid = str(r["id_add_str"])
        did = str(r["id_drop_str"])
        if aid not in id_to_idx.index or did not in id_to_idx.index:
            continue
        ai = int(id_to_idx.loc[aid])
        di = int(id_to_idx.loc[did])
        if require_add0 and base[ai] != 0:
            continue
        if require_drop1 and base[di] != 1:
            continue
        if aid in used_add or did in used_drop:
            continue
        used_add.add(aid)
        used_drop.add(did)
        rows.append(r)
    if not rows:
        return cand.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def apply_plan(sample, base, id_to_idx, swaps=None, plus=None, minus=None):
    pred = base.copy()
    used_add = set()
    used_drop = set()

    if swaps is not None and len(swaps):
        for _, r in swaps.iterrows():
            aid = str(r["id_add_str"])
            did = str(r["id_drop_str"])
            if aid in used_add or did in used_drop:
                continue
            pred[int(id_to_idx.loc[aid])] = 1
            pred[int(id_to_idx.loc[did])] = 0
            used_add.add(aid)
            used_drop.add(did)

    if plus is not None and len(plus):
        for _, r in plus.iterrows():
            aid = str(r["id_add_str"])
            if aid in used_add:
                continue
            pred[int(id_to_idx.loc[aid])] = 1
            used_add.add(aid)

    if minus is not None and len(minus):
        for _, r in minus.iterrows():
            did = str(r["id_drop_str"])
            if did in used_drop:
                continue
            pred[int(id_to_idx.loc[did])] = 0
            used_drop.add(did)

    return pred


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sample, v60, raw = load_inputs()
    pool, cols = prepare_pool(raw)
    base, pred_v60, id_to_idx = reconstruct_base(sample, v60, pool)

    # Eligibility after base reconstruction.
    pool["base_add_is0"] = pool["id_add_str"].map(lambda x: int(base[int(id_to_idx.loc[x])]) == 0 if x in id_to_idx.index else False)
    pool["base_drop_is1"] = pool["id_drop_str"].map(lambda x: int(base[int(id_to_idx.loc[x])]) == 1 if x in id_to_idx.index else False)
    pool["v65_eligible_swap"] = pool["base_add_is0"] & pool["base_drop_is1"]
    pool["v65_eligible_plus"] = pool["base_add_is0"]
    pool["v65_eligible_minus"] = pool["base_drop_is1"]

    for mode in ["ultra", "safe", "aggressive", "model_boost"]:
        pool[f"v65_mode_{mode}"] = pool[f"v65_mode_{mode}"] & pool["v65_eligible_swap"]

    pool = pool.sort_values(["v65_swap_utility", "v65_order"], ascending=[False, True])
    pool.to_csv(OUT_POOL, index=False)

    print("\nPOOLS")
    for mode in ["ultra", "safe", "aggressive", "model_boost"]:
        print(mode, int(pool[f"v65_mode_{mode}"].sum()))
    print("plus eligible clear:", int((pool["v65_eligible_plus"] & pool["v65_qwen_clear_a"] & (pool["v65_general_penalty"] <= 0.18)).sum()))
    print("minus eligible:", int((pool["v65_eligible_minus"] & (pool["b_relevant"].map(to_bool) == False)).sum()) if "b_relevant" in pool.columns else int(pool["v65_eligible_minus"].sum()))

    # Build sorted pools.
    swap_pools = {
        mode: unique_sorted(pool, f"v65_mode_{mode}", "v65_swap_utility", base, id_to_idx, True, True)
        for mode in ["ultra", "safe", "aggressive", "model_boost"]
    }

    plus_pool = pool[
        pool["v65_eligible_plus"]
        & pool["v65_qwen_clear_a"]
        & (pool["v65_general_penalty"] <= 0.20)
        & (pool["v65_plus_utility"] >= pool["v65_plus_utility"].quantile(0.55))
    ].sort_values(["v65_plus_utility", "v65_order"], ascending=[False, True]).drop_duplicates("id_add_str")

    minus_pool = pool[
        pool["v65_eligible_minus"]
        & (pool.get("b_relevant", pd.Series(False, index=pool.index)).map(to_bool) == False)
        & (pd.to_numeric(pool.get("confidence", pd.Series(0, index=pool.index)), errors="coerce").fillna(0) >= 0.65)
        & (pool["v65_minus_utility"] >= pool["v65_minus_utility"].quantile(0.55))
    ].sort_values(["v65_minus_utility", "v65_order"], ascending=[False, True]).drop_duplicates("id_drop_str")

    picked_frames = []
    saved = []

    def save_variant(name, pred, meta, swaps=None, plus=None, minus=None):
        file = SUB_DIR / f"FINAL_CANDIDATE_{name}_{short_hash(name)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(file, index=False)
        row = {
            "variant": name,
            "file": str(file),
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_reconstructed_base": int((pred != base).sum()),
            "diff_vs_v60_cap250": int((pred != pred_v60).sum()),
            **meta,
        }
        saved.append(row)

        if swaps is not None and len(swaps):
            x = swaps.copy()
            x["variant"] = name
            x["action"] = "swap"
            picked_frames.append(x)
        if plus is not None and len(plus):
            x = plus.copy()
            x["variant"] = name
            x["action"] = "plus"
            picked_frames.append(x)
        if minus is not None and len(minus):
            x = minus.copy()
            x["variant"] = name
            x["action"] = "minus"
            picked_frames.append(x)

        print("saved", file, row)

    # Same-quota swaps. More aggressive than V64 because it uses model utility + Qwen, not only Qwen.
    for mode, sp in swap_pools.items():
        for cap in [25, 50, 64, 75, 100, 150, 200, 250]:
            if len(sp) == 0:
                continue
            take = sp.head(min(cap, len(sp)))
            pred = apply_plan(sample, base, id_to_idx, swaps=take)
            save_variant(
                f"v65_general_{mode}_swap_cap{cap}",
                pred,
                {
                    "family": "swap",
                    "mode": mode,
                    "swap_cap": cap,
                    "used_swaps": int(len(take)),
                    "used_plus": 0,
                    "used_minus": 0,
                    "pool_size": int(len(sp)),
                    "mean_swap_utility": float(take["v65_swap_utility"].mean()),
                    "mean_qwen_score": float(take["v65_qwen_teacher_score"].mean()),
                    "mean_model_gain": float(take["v65_model_gain_score"].mean()),
                    "mean_penalty": float(take["v65_general_penalty"].mean()),
                },
                swaps=take,
            )

    # Quota shift: plus/minus/hybrid. Keep small but meaningful.
    for plus_cap in [25, 50, 75, 100]:
        take_plus = plus_pool.head(min(plus_cap, len(plus_pool)))
        if len(take_plus):
            pred = apply_plan(sample, base, id_to_idx, plus=take_plus)
            save_variant(
                f"v65_general_plus_cap{plus_cap}",
                pred,
                {
                    "family": "plus",
                    "mode": "plus",
                    "swap_cap": 0,
                    "used_swaps": 0,
                    "used_plus": int(len(take_plus)),
                    "used_minus": 0,
                    "pool_size": int(len(plus_pool)),
                    "mean_swap_utility": float(take_plus["v65_swap_utility"].mean()),
                    "mean_qwen_score": float(take_plus["v65_qwen_teacher_score"].mean()),
                    "mean_model_gain": float(take_plus["v65_model_gain_score"].mean()),
                    "mean_penalty": float(take_plus["v65_general_penalty"].mean()),
                },
                plus=take_plus,
            )

    for minus_cap in [25, 50, 75, 100]:
        take_minus = minus_pool.head(min(minus_cap, len(minus_pool)))
        if len(take_minus):
            pred = apply_plan(sample, base, id_to_idx, minus=take_minus)
            save_variant(
                f"v65_general_minus_cap{minus_cap}",
                pred,
                {
                    "family": "minus",
                    "mode": "minus",
                    "swap_cap": 0,
                    "used_swaps": 0,
                    "used_plus": 0,
                    "used_minus": int(len(take_minus)),
                    "pool_size": int(len(minus_pool)),
                    "mean_swap_utility": float(take_minus["v65_swap_utility"].mean()),
                    "mean_qwen_score": float(take_minus["v65_qwen_teacher_score"].mean()),
                    "mean_model_gain": float(take_minus["v65_model_gain_score"].mean()),
                    "mean_penalty": float(take_minus["v65_general_penalty"].mean()),
                },
                minus=take_minus,
            )

    # Hybrid candidates: controlled quota shifts on top of strong swaps.
    for mode in ["ultra", "safe", "aggressive"]:
        sp = swap_pools[mode]
        if len(sp) == 0:
            continue
        for swap_cap, plus_cap, minus_cap in [
            (25, 25, 0),
            (50, 25, 0),
            (50, 50, 0),
            (64, 25, 0),
            (64, 50, 0),
            (50, 0, 25),
            (64, 0, 25),
            (50, 25, 25),
            (64, 25, 25),
        ]:
            take_sw = sp.head(min(swap_cap, len(sp)))
            used_adds = set(take_sw["id_add_str"])
            used_drops = set(take_sw["id_drop_str"])
            take_plus = plus_pool[~plus_pool["id_add_str"].isin(used_adds)].head(min(plus_cap, len(plus_pool)))
            take_minus = minus_pool[~minus_pool["id_drop_str"].isin(used_drops)].head(min(minus_cap, len(minus_pool)))

            if len(take_sw) + len(take_plus) + len(take_minus) == 0:
                continue

            pred = apply_plan(sample, base, id_to_idx, swaps=take_sw, plus=take_plus, minus=take_minus)
            save_variant(
                f"v65_general_{mode}_swap{swap_cap}_plus{plus_cap}_minus{minus_cap}",
                pred,
                {
                    "family": "hybrid",
                    "mode": mode,
                    "swap_cap": swap_cap,
                    "used_swaps": int(len(take_sw)),
                    "used_plus": int(len(take_plus)),
                    "used_minus": int(len(take_minus)),
                    "pool_size": int(len(sp)),
                    "mean_swap_utility": float(pd.concat([take_sw, take_plus, take_minus])["v65_swap_utility"].mean()),
                    "mean_qwen_score": float(pd.concat([take_sw, take_plus, take_minus])["v65_qwen_teacher_score"].mean()),
                    "mean_model_gain": float(pd.concat([take_sw, take_plus, take_minus])["v65_model_gain_score"].mean()),
                    "mean_penalty": float(pd.concat([take_sw, take_plus, take_minus])["v65_general_penalty"].mean()),
                },
                swaps=take_sw,
                plus=take_plus,
                minus=take_minus,
            )

    summary = pd.DataFrame(saved)
    if len(summary):
        # Diagnostic, not true validation score.
        summary["v65_diagnostic_score"] = (
            0.040 * np.minimum(1, np.log1p(summary["diff_vs_reconstructed_base"]) / np.log1p(700))
            + 0.025 * summary["mean_qwen_score"].fillna(0)
            + 0.018 * summary["mean_model_gain"].fillna(0)
            - 0.030 * summary["mean_penalty"].fillna(0)
            - np.maximum(0, summary["diff_vs_reconstructed_base"] - 700) / 20000
        )
        summary = summary.sort_values(["v65_diagnostic_score", "mean_penalty", "diff_vs_reconstructed_base"], ascending=[False, True, True])
    summary.to_csv(OUT_SUMMARY, index=False)
    summary.to_csv(OUT_SAVED, index=False)

    if picked_frames:
        picked = pd.concat(picked_frames, ignore_index=True)
        picked.to_csv(OUT_PICKED, index=False)

    print("\nTOP SUMMARY")
    cols = [
        "variant", "file", "family", "mode", "used_swaps", "used_plus", "used_minus",
        "ones", "pos_ratio", "diff_vs_reconstructed_base", "diff_vs_v60_cap250",
        "mean_qwen_score", "mean_model_gain", "mean_penalty", "v65_diagnostic_score",
    ]
    if len(summary):
        print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))

    print("\noutputs:")
    print(OUT_POOL)
    print(OUT_PICKED)
    print(OUT_SUMMARY)
    print(OUT_SAVED)
    print(SUB_DIR)


if __name__ == "__main__":
    main()
