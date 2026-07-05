from pathlib import Path
import re
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"

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

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_SUMMARY = OUT_DIR / "v38_lexical_candidate_summary.csv"
OUT_EVAL = OUT_DIR / "v38_lexical_candidate_eval.csv"
OUT_SWAPS = OUT_DIR / "v38_lexical_swap_diagnostics.csv"
OUT_SAVED = OUT_DIR / "v38_lexical_saved_candidates.csv"
OUT_REVIEW = OUT_DIR / "review_v38_lexical_candidates_sample.csv"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def clean_name(x):
    return (
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


def build_same_quota(df, score_col, anchor_pred):
    tmp = pd.DataFrame({
        "term_id": df["term_id"].to_numpy(),
        "score": df[score_col].to_numpy(np.float32),
        "anchor": anchor_pred,
    })
    quota = tmp.groupby("term_id")["anchor"].sum()
    tmp["quota"] = tmp["term_id"].map(quota).fillna(0).astype(np.int32)
    tmp["rank"] = tmp.groupby("term_id")["score"].rank(method="first", ascending=False).astype(np.int32)
    return (tmp["rank"].to_numpy() <= tmp["quota"].to_numpy()).astype(np.int8)


def pair_swaps_from_preds(df, anchor, cand, pair_score_col):
    add_mask = (anchor == 0) & (cand == 1)
    drop_mask = (anchor == 1) & (cand == 0)

    add = df.loc[add_mask, ["id", "term_id", "item_id", pair_score_col,
                            "v38_lex_score", "v38_lex_pct_rank", "v38_lex_term_z",
                            "v38_exact_phrase_title", "v38_word_overlap_title",
                            "v38_must_token_coverage_full", "v38_special_token_coverage_full",
                            "v38_number_match", "v38_model_match", "v38_query_word_only_category_ratio",
                            "v33_rs", "v34_rs"]].copy()

    drop = df.loc[drop_mask, ["id", "term_id", "item_id", pair_score_col,
                              "v38_lex_score", "v38_lex_pct_rank", "v38_lex_term_z",
                              "v38_exact_phrase_title", "v38_word_overlap_title",
                              "v38_must_token_coverage_full", "v38_special_token_coverage_full",
                              "v38_number_match", "v38_model_match", "v38_query_word_only_category_ratio",
                              "v33_rs", "v34_rs"]].copy()

    add = add.sort_values(["term_id", pair_score_col], ascending=[True, False])
    drop = drop.sort_values(["term_id", pair_score_col], ascending=[True, True])

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    sw = add.merge(drop, on=["term_id", "pair_rank"], suffixes=("_add", "_drop"), how="inner")
    sw["pair_score_gain"] = sw[f"{pair_score_col}_add"] - sw[f"{pair_score_col}_drop"]
    sw["lex_gain"] = sw["v38_lex_score_add"] - sw["v38_lex_score_drop"]
    sw["v38_rank_gain"] = (1.0 - sw["v38_lex_pct_rank_add"]) - (1.0 - sw["v38_lex_pct_rank_drop"])

    return sw.sort_values(["pair_score_gain", "lex_gain"], ascending=False).reset_index(drop=True)


def lexical_accept_mask(sw, mode):
    add_strong = (
        (sw["v38_lex_score_add"] >= 0.28)
        | (sw["v38_exact_phrase_title_add"] == 1)
        | (sw["v38_word_overlap_title_add"] >= 0.50)
        | (sw["v38_must_token_coverage_full_add"] >= 0.75)
        | ((sw["v38_number_match_add"] >= 1) & (sw["v38_word_overlap_title_add"] >= 0.25))
        | ((sw["v38_model_match_add"] >= 1) & (sw["v38_word_overlap_title_add"] >= 0.20))
    )

    add_bad = (
        (sw["v38_lex_score_add"] < -0.02)
        | (
            (sw["v38_query_word_only_category_ratio_add"] >= 0.50)
            & (sw["v38_word_overlap_title_add"] <= 0.10)
        )
        | (
            (sw["v38_must_token_coverage_full_add"] <= 0.20)
            & (sw["v38_special_token_coverage_full_add"] <= 0.20)
            & (sw["v38_word_overlap_title_add"] <= 0.15)
        )
    )

    drop_good = (
        (sw["v38_lex_score_drop"] >= 0.24)
        & (sw["v38_word_overlap_title_drop"] >= sw["v38_word_overlap_title_add"])
        & (sw["v38_must_token_coverage_full_drop"] >= sw["v38_must_token_coverage_full_add"])
    )

    lex_gain_ok = sw["lex_gain"] >= 0.03
    rank_gain_ok = sw["v38_rank_gain"] >= 0.06

    if mode == "ultra":
        return add_strong & (~add_bad) & (~drop_good) & (sw["lex_gain"] >= 0.08)
    if mode == "strict":
        return add_strong & (~add_bad) & (~drop_good) & (lex_gain_ok | rank_gain_ok)
    if mode == "balanced":
        return add_strong & (~add_bad) & ((lex_gain_ok | rank_gain_ok) | (sw["pair_score_gain"] >= 0.12))
    if mode == "loose":
        return (~add_bad) & ((add_strong & (lex_gain_ok | rank_gain_ok)) | (sw["pair_score_gain"] >= 0.18))
    raise ValueError(mode)


def apply_swaps(sample, anchor, sw, max_swaps=None):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    take = sw.copy()
    take = take.sort_values(["pair_score_gain", "lex_gain"], ascending=False)
    if max_swaps is not None:
        take = take.head(max_swaps).copy()

    pred = anchor.copy()
    add_idx = take["id_add"].astype(str).map(id_to_idx)
    drop_idx = take["id_drop"].astype(str).map(id_to_idx)

    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("id map failed in apply_swaps")

    pred[add_idx.astype(int).to_numpy()] = 1
    pred[drop_idx.astype(int).to_numpy()] = 0
    return pred, take


def make_score_columns(df, anchor):
    # Rank-like transforms: higher is better.
    df["v38_rs"] = (1.0 - df["v38_lex_pct_rank"].astype(np.float32)).clip(0, 1)
    df["v38_zsig"] = sigmoid(df["v38_lex_term_z"].fillna(0).to_numpy(np.float32) / 1.7).astype(np.float32)

    df["v33_rs"] = (1.0 - df["v33_ft_pct_rank"].astype(np.float32)).clip(0, 1)
    df["v34_rs"] = (1.0 - df["v34_ce_pct_rank"].astype(np.float32)).clip(0, 1)

    df["v33_zsig"] = sigmoid(df["v33_ft_term_z"].fillna(0).to_numpy(np.float32) / 1.8).astype(np.float32)
    df["v34_zsig"] = sigmoid(df["v34_ce_term_z"].fillna(0).to_numpy(np.float32) / 1.8).astype(np.float32)

    # Convert raw lexical into rough [0,1] support.
    df["v38_raw01"] = ((df["v38_lex_score"].astype(np.float32) + 0.16) / 1.29).clip(0, 1)

    anchor_f = anchor.astype(np.float32)

    # Pure lexical/search engine.
    df["v38_pure_lex_score"] = (
        0.54 * df["v38_rs"]
        + 0.16 * df["v38_zsig"]
        + 0.12 * df["v38_raw01"]
        + 0.08 * df["v38_exact_phrase_title"].astype(np.float32)
        + 0.05 * df["v38_must_token_coverage_full"].astype(np.float32)
        + 0.03 * df["v38_number_match"].astype(np.float32)
        + 0.02 * df["v38_model_match"].astype(np.float32)
    ).astype(np.float32)

    # Lexical + V33: mostly independent from V34.
    df["v38_lex_v33_score"] = (
        0.42 * df["v38_rs"]
        + 0.26 * df["v33_rs"]
        + 0.10 * df["v38_zsig"]
        + 0.08 * df["v33_zsig"]
        + 0.06 * df["v38_raw01"]
        + 0.05 * df["v38_must_token_coverage_full"].astype(np.float32)
        + 0.03 * anchor_f
    ).astype(np.float32)

    # Lexical + V33 + V34 balanced.
    df["v38_lex_sem_balanced_score"] = (
        0.32 * df["v38_rs"]
        + 0.25 * df["v34_rs"]
        + 0.20 * df["v33_rs"]
        + 0.08 * df["v38_zsig"]
        + 0.06 * df["v34_zsig"]
        + 0.04 * df["v33_zsig"]
        + 0.03 * df["v38_raw01"]
        + 0.02 * anchor_f
    ).astype(np.float32)

    # Precision blend: add needs both semantic and lexical support.
    df["v38_precision_score"] = (
        0.30 * df["v38_rs"]
        + 0.30 * df["v34_rs"]
        + 0.20 * df["v33_rs"]
        + 0.10 * np.minimum(df["v38_rs"], np.maximum(df["v33_rs"], df["v34_rs"]))
        + 0.06 * df["v38_must_token_coverage_full"].astype(np.float32)
        + 0.04 * anchor_f
    ).astype(np.float32)

    return df


def build_review_file(sample, df, chosen_swaps):
    pieces = []
    for name, sw in chosen_swaps.items():
        if len(sw) == 0:
            continue
        x = sw.copy()
        x["variant"] = name
        pieces.append(x.head(120))
        pieces.append(x.tail(80))

    if not pieces:
        return

    review = pd.concat(pieces, ignore_index=True).drop_duplicates(["variant", "id_add", "id_drop"], keep="first")

    # Attach a few useful item/query fields if raw item text exists in previous review files? Not guaranteed.
    keep = [
        "variant", "term_id", "pair_rank",
        "id_add", "item_id_add", "id_drop", "item_id_drop",
        "pair_score_gain", "lex_gain", "v38_rank_gain",
        "v38_lex_score_add", "v38_lex_score_drop",
        "v38_exact_phrase_title_add", "v38_exact_phrase_title_drop",
        "v38_word_overlap_title_add", "v38_word_overlap_title_drop",
        "v38_must_token_coverage_full_add", "v38_must_token_coverage_full_drop",
        "v38_special_token_coverage_full_add", "v38_special_token_coverage_full_drop",
        "v38_number_match_add", "v38_number_match_drop",
        "v38_model_match_add", "v38_model_match_drop",
        "v38_query_word_only_category_ratio_add", "v38_query_word_only_category_ratio_drop",
        "v33_rs_add", "v33_rs_drop", "v34_rs_add", "v34_rs_drop",
    ]
    review[[c for c in keep if c in review.columns]].to_csv(OUT_REVIEW, index=False)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print("loading sample/pairs...")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("submission_pairs order mismatch sample")

    anchor_path = first_existing(ANCHOR_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("v33 qprob2000 anchor missing")

    anchor = load_pred(anchor_path, sample)
    print("anchor:", anchor_path, "ones:", int(anchor.sum()))

    baselines = {"anchor_v33_qprob2000": anchor.copy()}

    for name, paths in BASELINE_PATHS.items():
        p = first_existing(paths)
        if p is not None:
            baselines[name] = load_pred(p, sample)
            print("baseline:", name, p, "diff_vs_anchor:", int((baselines[name] != anchor).sum()))

    print("loading v38/v33/v34 scores...")
    v38_cols = [
        "v38_lex_score", "v38_lex_rank", "v38_lex_pct_rank", "v38_lex_term_z",
        "v38_exact_phrase_title", "v38_exact_phrase_full",
        "v38_word_overlap_title", "v38_word_overlap_category", "v38_word_overlap_full",
        "v38_word_jaccard_title", "v38_word_jaccard_full",
        "v38_char_ngram_sim",
        "v38_must_token_coverage_title", "v38_must_token_coverage_full",
        "v38_special_token_coverage_title", "v38_special_token_coverage_full",
        "v38_brand_match", "v38_number_match", "v38_model_match",
        "v38_category_title_support",
        "v38_query_word_in_title_any",
        "v38_query_word_only_category_ratio",
        "v38_query_word_only_category_any",
    ]

    v38 = align_parquet(V38, sample, v38_cols)
    v33 = align_parquet(V33, sample, ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"])
    v34 = align_parquet(V34, sample, ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"])

    df = pairs.copy()

    for c in v38_cols:
        df[c] = pd.to_numeric(v38[c], errors="coerce").fillna(0).astype(np.float32)
    for c in ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"]:
        df[c] = pd.to_numeric(v33[c], errors="coerce").fillna(0).astype(np.float32)
    for c in ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"]:
        df[c] = pd.to_numeric(v34[c], errors="coerce").fillna(0).astype(np.float32)

    df = make_score_columns(df, anchor)

    variants = dict(baselines)
    meta_rows = []
    chosen_swaps = {}
    swap_tables = []

    # 1) Pure lexical / blend same-quota reranks.
    score_cols = [
        "v38_pure_lex_score",
        "v38_lex_v33_score",
        "v38_lex_sem_balanced_score",
        "v38_precision_score",
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

    # 2) Lexical-filtered swap candidates from raw_v35 and v36p2.
    swap_bases = []
    for base_name in ["raw_v35_b5000", "v36p2_balanced", "v36p2_strict"]:
        if base_name in baselines:
            swap_bases.append(base_name)

    caps = {
        "ultra": [500, 900, 1300],
        "strict": [1000, 1600, 2200],
        "balanced": [1400, 2200, 3200],
        "loose": [1800, 2800, 4200],
    }

    for base_name in swap_bases:
        base_pred = baselines[base_name]

        for sc in ["v38_precision_score", "v38_lex_sem_balanced_score", "v38_lex_v33_score", "v38_pure_lex_score"]:
            sw = pair_swaps_from_preds(df, anchor, base_pred, sc)
            sw["base_name"] = base_name
            sw["score_col"] = sc

            for mode in ["ultra", "strict", "balanced", "loose"]:
                m = lexical_accept_mask(sw, mode)
                accepted = sw[m].copy()
                accepted = accepted.sort_values(["pair_score_gain", "lex_gain"], ascending=False)

                for cap in caps[mode]:
                    if len(accepted) == 0:
                        continue

                    pred, take = apply_swaps(sample, anchor, accepted, max_swaps=cap)
                    name = f"v38_{base_name}_{sc}_{mode}_cap{cap}"
                    variants[name] = pred
                    chosen_swaps[name] = take

                    meta_rows.append({
                        "variant": name,
                        "source": "lexical_filtered_swaps",
                        "base": base_name,
                        "score_col": sc,
                        "mode": mode,
                        "cap": cap,
                    })

                    print("swap variant:", name, "accepted:", len(take), "diff_vs_anchor:", int((pred != anchor).sum()))

            # Save a diagnostic per base/score, but not full giant tables.
            diag = sw.head(2000).copy()
            diag["candidate_source"] = f"{base_name}_{sc}"
            swap_tables.append(diag)

    # 3) Lexical rescue over V36P2: take raw V35 swaps rejected by V36P2 but lexically strong.
    if "raw_v35_b5000" in baselines and "v36p2_strict" in baselines:
        raw = baselines["raw_v35_b5000"]
        strict = baselines["v36p2_strict"]

        raw_sw = pair_swaps_from_preds(df, anchor, raw, "v38_precision_score")

        # A swap exists in strict if add/drop both present in strict delta approximately.
        strict_add_ids = set(df.loc[(anchor == 0) & (strict == 1), "id"].astype(str))
        strict_drop_ids = set(df.loc[(anchor == 1) & (strict == 0), "id"].astype(str))

        raw_sw["in_strict"] = raw_sw["id_add"].astype(str).isin(strict_add_ids) & raw_sw["id_drop"].astype(str).isin(strict_drop_ids)

        rescue_mask = (
            (~raw_sw["in_strict"])
            & (
                (raw_sw["v38_exact_phrase_title_add"] == 1)
                | (raw_sw["v38_lex_score_add"] >= 0.55)
                | (
                    (raw_sw["v38_must_token_coverage_full_add"] >= 0.90)
                    & (raw_sw["v38_word_overlap_title_add"] >= 0.45)
                )
                | (
                    (raw_sw["v38_number_match_add"] >= 1)
                    & (raw_sw["v38_model_match_add"] >= raw_sw["v38_model_match_drop"])
                    & (raw_sw["v38_word_overlap_title_add"] >= 0.30)
                )
            )
            & (raw_sw["lex_gain"] >= 0.03)
            & (raw_sw["v38_query_word_only_category_ratio_add"] <= 0.35)
        )

        rescue = raw_sw[rescue_mask].copy().sort_values(["v38_lex_score_add", "lex_gain"], ascending=False)

        for cap in [200, 400, 700, 1000]:
            if len(rescue) == 0:
                continue

            pred, take = apply_swaps(sample, strict, rescue, max_swaps=cap)
            name = f"v38_rescue_raw35_over_v36p2_strict_cap{cap}"
            variants[name] = pred
            chosen_swaps[name] = take

            meta_rows.append({
                "variant": name,
                "source": "lexical_rescue",
                "base": "v36p2_strict",
                "score_col": "v38_precision_score",
                "mode": "rescue",
                "cap": cap,
            })
            print("rescue:", name, "rescued:", len(take), "diff_vs_anchor:", int((pred != anchor).sum()))

    # 4) Lexical veto over raw V35 and V36P2 balanced: keep base swaps unless add is lexically bad.
    for base_name in ["raw_v35_b5000", "v36p2_balanced"]:
        if base_name not in baselines:
            continue

        base_pred = baselines[base_name]
        sw = pair_swaps_from_preds(df, anchor, base_pred, "v38_precision_score")

        add_bad = (
            (sw["v38_lex_score_add"] < -0.02)
            | (
                (sw["v38_query_word_only_category_ratio_add"] >= 0.50)
                & (sw["v38_word_overlap_title_add"] <= 0.10)
            )
            | (
                (sw["v38_must_token_coverage_full_add"] <= 0.15)
                & (sw["v38_special_token_coverage_full_add"] <= 0.15)
                & (sw["v38_word_overlap_title_add"] <= 0.10)
            )
        )

        drop_good = (
            (sw["v38_lex_score_drop"] >= 0.28)
            & (sw["v38_word_overlap_title_drop"] >= sw["v38_word_overlap_title_add"])
            & (sw["v38_must_token_coverage_full_drop"] >= sw["v38_must_token_coverage_full_add"])
        )

        for strength, mask in {
            "veto_bad_only": ~add_bad,
            "veto_bad_and_drop_good": ~(add_bad | drop_good),
        }.items():
            kept = sw[mask].copy().sort_values(["pair_score_gain", "lex_gain"], ascending=False)
            pred, take = apply_swaps(sample, anchor, kept, max_swaps=None)

            name = f"v38_{base_name}_{strength}"
            variants[name] = pred
            chosen_swaps[name] = take

            meta_rows.append({
                "variant": name,
                "source": "lexical_veto",
                "base": base_name,
                "score_col": "v38_precision_score",
                "mode": strength,
                "cap": "all",
            })
            print("veto:", name, "kept:", len(take), "diff_vs_anchor:", int((pred != anchor).sum()))

    # Eval
    print("evaluating...")
    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)

    w = weighted_score(eval_df)

    # Summary
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

    aux_rows = []
    for name, pred in variants.items():
        aux_rows.append({
            "variant": name,
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_v24": int((pred != baselines["v24"]).sum()) if "v24" in baselines else -1,
            "diff_vs_raw_v35": int((pred != baselines["raw_v35_b5000"]).sum()) if "raw_v35_b5000" in baselines else -1,
            "diff_vs_v36p2_strict": int((pred != baselines["v36p2_strict"]).sum()) if "v36p2_strict" in baselines else -1,
            "diff_vs_v36p2_balanced": int((pred != baselines["v36p2_balanced"]).sum()) if "v36p2_balanced" in baselines else -1,
        })

    summary = summary.merge(pd.DataFrame(aux_rows), on="variant", how="right")

    if len(w):
        summary = summary.merge(w, on="variant", how="left")

    # Score: local quality + enough movement but avoid completely wild candidates.
    diff = summary["diff_vs_anchor"].fillna(0).astype(float)
    movement_bonus = np.minimum(0.018, np.log1p(diff) / np.log1p(25000) * 0.018)
    too_big_penalty = np.maximum(0, diff - 50000) / 1000000.0

    summary["v38_decision_score"] = (
        summary["weighted_macro"].fillna(0)
        + movement_bonus
        - too_big_penalty
    )

    summary = summary.sort_values(["v38_decision_score", "weighted_macro"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if swap_tables:
        pd.concat(swap_tables, ignore_index=True).to_csv(OUT_SWAPS, index=False)

    build_review_file(sample, df, chosen_swaps)

    # Save top candidates
    save_names = []

    save_filter = summary[
        (summary["source"].isin(["same_quota", "lexical_filtered_swaps", "lexical_veto", "lexical_rescue"]))
        & (summary["diff_vs_anchor"] >= 1500)
        & (summary["diff_vs_anchor"] <= 50000)
    ].head(30)

    for nm in save_filter["variant"].tolist():
        if nm not in save_names:
            save_names.append(nm)

    # Always include best baselines for comparison.
    for nm in ["anchor_v33_qprob2000", "raw_v35_b5000", "v36p2_strict", "v36p2_balanced"]:
        if nm in variants and nm not in save_names:
            save_names.append(nm)

    saved = []
    for nm in save_names[:35]:
        pred = variants[nm]
        out = SUB_DIR / f"FINAL_CANDIDATE_v38_{clean_name(nm)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

        row = summary[summary["variant"] == nm].iloc[0].to_dict()
        row["file"] = str(out)
        saved.append(row)
        print("saved:", out)

    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)

    print("\nTOP V38")
    cols = [
        "variant", "v38_decision_score", "weighted_macro", "source", "base", "score_col", "mode", "cap",
        "ones", "pos_ratio", "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v36p2_strict",
        "main_min_macro", "main_mean_macro", "mean_precision", "mean_recall",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))

    print("\nSAVED")
    saved_df = pd.DataFrame(saved)
    if len(saved_df):
        print(saved_df[["variant", "file", "v38_decision_score", "weighted_macro", "diff_vs_anchor"]].to_string(index=False))

    elapsed = (time.time() - t0) / 60
    print("\noutputs:")
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_SWAPS)
    print(OUT_SAVED)
    print(OUT_REVIEW)
    print("elapsed_min:", elapsed)


if __name__ == "__main__":
    main()
