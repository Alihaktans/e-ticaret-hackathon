from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings("ignore")

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"

V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"
V38 = ROOT / "data/processed/v38_lexical_pair_scores.parquet"
V39 = ROOT / "data/processed/v39_item_history_pair_scores.parquet"

ANCHOR_PATHS = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

DIRECT_BASES = {
    "raw_v35_b5000": [
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv",
    ],
    "v36p2_balanced": [
        ROOT / "submissions/FINAL_MAIN_v36p2_balanced.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v36p2_balanced.csv",
    ],
    "v36p2_strict": [
        ROOT / "submissions/FINAL_MAIN_v36p2_strict.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v36p2_strict.csv",
    ],
}

SAVED_CSVS = [
    ROOT / "reports/manual_review/v38_lexical_saved_candidates.csv",
    ROOT / "reports/manual_review/v39_item_history_saved_candidates.csv",
    ROOT / "reports/manual_review/v40_boundary_stability_saved_candidates.csv",
]

SAVED_VARIANTS = {
    "v38_precision_loose_2800": "v38_raw_v35_b5000_v38_precision_score_loose_cap2800",
    "v38_veto_bad_only": "v38_raw_v35_b5000_veto_bad_only",
    "v39_raw35_history_lex_veto_bad": "v39_raw_v35_b5000_v39_history_lex_score_history_veto_contradiction_and_bad",
    "v39_raw35_history_gate_veto": "v39_raw_v35_b5000_v39_history_gate_score_history_veto_contradiction",
    "v40_history_impact2200": "v40_raw_v35_b5000_v40_history_score_impact_cap2200",
    "v40_text_impact5200": "v40_raw_v35_b5000_v40_text_score_impact_cap5200",
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

REPORT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"
PROC_DIR = ROOT / "data/processed"

OUT_PROB = PROC_DIR / "v41_learned_meta_prob.parquet"
OUT_MODEL_INFO = REPORT_DIR / "v41_learned_meta_model_info.csv"
OUT_LABEL_OOF = REPORT_DIR / "v41_learned_meta_label_oof.csv"
OUT_SUMMARY = REPORT_DIR / "v41_learned_meta_candidate_summary.csv"
OUT_EVAL = REPORT_DIR / "v41_learned_meta_candidate_eval.csv"
OUT_SAVED = REPORT_DIR / "v41_learned_meta_saved_candidates.csv"
OUT_REVIEW = REPORT_DIR / "review_v41_learned_meta_swaps.csv"


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def find_saved_variant(variant_name):
    for csv_path in SAVED_CSVS:
        if not csv_path.exists():
            continue
        d = pd.read_csv(csv_path)
        if "variant" not in d.columns or "file" not in d.columns:
            continue
        hit = d[d["variant"].astype(str).eq(variant_name)]
        if len(hit):
            p = Path(str(hit.iloc[0]["file"]))
            if p.exists():
                return p
    return None


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


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


def load_labels():
    parts = []
    for name, path in LABEL_FILES.items():
        if not path.exists():
            print("missing label:", name, path)
            continue

        d = pd.read_csv(path)
        if "id" not in d.columns or "assistant_label" not in d.columns:
            continue

        d["id"] = d["id"].astype(str)
        d = d[d["assistant_label"].isin([0, 1, "0", "1"])].copy()
        if len(d) == 0:
            continue

        d["label"] = d["assistant_label"].astype(int)
        d["label_source"] = name

        w = np.ones(len(d), dtype=np.float32)

        if "needs_recheck" in d.columns:
            nr = pd.to_numeric(d["needs_recheck"], errors="coerce").fillna(0).astype(int).to_numpy()
            w *= np.where(nr == 0, 1.0, 0.35)

        if "assistant_confidence" in d.columns:
            conf = d["assistant_confidence"].astype(str).str.lower().to_numpy()
            w *= np.where(conf == "high", 1.25, np.where(conf == "medium", 0.85, 0.55))

        # Random/manual labels are more important than active targeted labels.
        if name.startswith("random"):
            w *= 1.30
        if name.startswith("manual"):
            w *= 1.20
        if name in ["v20_active", "v21_active", "v26_sparse"]:
            w *= 0.90

        d["sample_weight"] = w
        parts.append(d[["id", "label", "sample_weight", "label_source"]])

    if not parts:
        raise RuntimeError("no label files found")

    lab = pd.concat(parts, ignore_index=True)

    # Deduplicate by weighted vote.
    agg = []
    for id_, g in lab.groupby("id"):
        pos_w = float(g.loc[g["label"] == 1, "sample_weight"].sum())
        neg_w = float(g.loc[g["label"] == 0, "sample_weight"].sum())
        label = int(pos_w >= neg_w)
        total = pos_w + neg_w
        source = "|".join(sorted(g["label_source"].astype(str).unique()))
        agg.append({
            "id": id_,
            "label": label,
            "sample_weight": max(0.25, total),
            "label_margin": abs(pos_w - neg_w) / max(1e-9, total),
            "label_sources": source,
            "label_dup_count": int(len(g)),
        })

    out = pd.DataFrame(agg)
    print("labels raw:", len(lab), "dedup:", len(out), "pos_rate:", out["label"].mean())
    return out


def build_features(sample, pairs, baselines):
    print("loading pair feature files...")
    df = pairs.copy()

    v33 = align_parquet(V33, sample, ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"])
    v34 = align_parquet(V34, sample, ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"])

    v38_cols = [
        "v38_lex_score", "v38_lex_pct_rank", "v38_lex_term_z",
        "v38_exact_phrase_title", "v38_exact_phrase_full",
        "v38_word_overlap_title", "v38_word_overlap_category", "v38_word_overlap_full",
        "v38_word_jaccard_title", "v38_char_ngram_sim",
        "v38_must_token_coverage_full", "v38_special_token_coverage_full",
        "v38_brand_match", "v38_number_match", "v38_model_match",
        "v38_category_title_support",
        "v38_query_word_only_category_ratio", "v38_query_word_only_category_any",
    ]
    v38 = align_parquet(V38, sample, v38_cols)

    v39_cols = [
        "v39_hist_score", "v39_hist_has_item", "v39_hist_count",
        "v39_hist_token_cov", "v39_hist_weighted_cov", "v39_hist_must_cov",
        "v39_hist_special_cov", "v39_hist_exact_query", "v39_hist_char_cov",
        "v39_hist_number_match", "v39_hist_model_match",
        "v39_hist_pct_rank", "v39_hist_term_z",
    ]
    if V39.exists():
        v39 = align_parquet(V39, sample, v39_cols)
    else:
        v39 = pd.DataFrame({"id": sample["id"]})
        for c in v39_cols:
            v39[c] = 0

    for src in [v33, v34, v38, v39]:
        for c in src.columns:
            if c != "id":
                df[c] = pd.to_numeric(src[c], errors="coerce").fillna(0).astype(np.float32)

    # Query group context.
    df["term_count"] = df.groupby("term_id")["id"].transform("count").astype(np.float32)

    # Rank scores
    df["v33_rs"] = (1.0 - df["v33_ft_pct_rank"]).clip(0, 1)
    df["v34_rs"] = (1.0 - df["v34_ce_pct_rank"]).clip(0, 1)
    df["v38_rs"] = (1.0 - df["v38_lex_pct_rank"]).clip(0, 1)
    df["v39_rs"] = (1.0 - df["v39_hist_pct_rank"]).clip(0, 1)

    df["v33_zsig"] = sigmoid(df["v33_ft_term_z"].to_numpy(np.float32) / 1.8).astype(np.float32)
    df["v34_zsig"] = sigmoid(df["v34_ce_term_z"].to_numpy(np.float32) / 1.8).astype(np.float32)
    df["v38_zsig"] = sigmoid(df["v38_lex_term_z"].to_numpy(np.float32) / 1.7).astype(np.float32)
    df["v39_zsig"] = sigmoid(df["v39_hist_term_z"].to_numpy(np.float32) / 1.7).astype(np.float32)

    rank_cols = ["v33_rs", "v34_rs", "v38_rs", "v39_rs"]
    df["rank_mean"] = df[rank_cols].mean(axis=1).astype(np.float32)
    df["rank_min"] = df[rank_cols].min(axis=1).astype(np.float32)
    df["rank_max"] = df[rank_cols].max(axis=1).astype(np.float32)
    df["rank_std"] = df[rank_cols].std(axis=1).fillna(0).astype(np.float32)
    df["sem_mean"] = df[["v33_rs", "v34_rs"]].mean(axis=1).astype(np.float32)
    df["text_hist_mean"] = df[["v38_rs", "v39_rs"]].mean(axis=1).astype(np.float32)
    df["sem_text_gap"] = (df["sem_mean"] - df["text_hist_mean"]).astype(np.float32)

    # Baseline votes/preds as features.
    for name, pred in baselines.items():
        df[f"pred_{name}"] = pred.astype(np.int8)

    pred_cols = [f"pred_{k}" for k in baselines.keys()]
    df["pred_vote_sum"] = df[pred_cols].sum(axis=1).astype(np.float32)
    df["pred_vote_mean"] = df[pred_cols].mean(axis=1).astype(np.float32)
    df["pred_vote_std"] = df[pred_cols].std(axis=1).fillna(0).astype(np.float32)

    # Quota from anchor
    if "anchor_v33_qprob2000" in baselines:
        quota = pd.DataFrame({
            "term_id": df["term_id"],
            "anchor": baselines["anchor_v33_qprob2000"],
        }).groupby("term_id")["anchor"].sum()
        df["anchor_quota"] = df["term_id"].map(quota).fillna(0).astype(np.float32)
        df["anchor_quota_ratio"] = (df["anchor_quota"] / np.maximum(1, df["term_count"])).astype(np.float32)
    else:
        df["anchor_quota"] = 0.0
        df["anchor_quota_ratio"] = 0.0

    # Per-term rank positions for meta features.
    denom = np.maximum(1.0, df["term_count"].to_numpy(np.float32) - 1.0)
    df["v33_rankpos"] = 1.0 + df["v33_ft_pct_rank"].to_numpy(np.float32) * denom
    df["v34_rankpos"] = 1.0 + df["v34_ce_pct_rank"].to_numpy(np.float32) * denom
    df["v38_rankpos"] = 1.0 + df["v38_lex_pct_rank"].to_numpy(np.float32) * denom
    df["v39_rankpos"] = 1.0 + df["v39_hist_pct_rank"].to_numpy(np.float32) * denom

    return df


def get_feature_columns(df):
    drop = {"id", "term_id", "item_id"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def make_model(random_seed=42):
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            loss_function="Logloss",
            iterations=1600,
            depth=5,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            random_seed=random_seed,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
        ), "catboost"
    except Exception as e:
        print("CatBoost unavailable, fallback sklearn:", repr(e))
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=450,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            random_state=random_seed,
        ), "hist_gradient_boosting"


def predict_proba_model(model, model_kind, X):
    if model_kind == "catboost":
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


def fit_model(model, model_kind, X, y, w):
    if model_kind == "catboost":
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def train_oof_and_score_all(df, labels, feature_cols, sample):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])

    lab = labels.copy()
    lab["row_idx"] = lab["id"].map(id_to_idx)
    lab = lab[lab["row_idx"].notna()].copy()
    lab["row_idx"] = lab["row_idx"].astype(int)

    X_lab = df.loc[lab["row_idx"].to_numpy(), feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = lab["label"].astype(int).to_numpy()
    w = lab["sample_weight"].astype(float).to_numpy()
    groups = df.loc[lab["row_idx"].to_numpy(), "term_id"].astype(str).to_numpy()

    oof = np.zeros(len(lab), dtype=np.float32)

    n_unique_groups = len(np.unique(groups))
    n_splits = 5 if len(y) >= 250 and n_unique_groups >= 5 else 3

    if n_unique_groups >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X_lab, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = splitter.split(X_lab, y)

    fold_rows = []

    for fold, (tr, va) in enumerate(splits, 1):
        model, kind = make_model(100 + fold)
        model = fit_model(model, kind, X_lab.iloc[tr], y[tr], w[tr])
        pred = predict_proba_model(model, kind, X_lab.iloc[va])
        oof[va] = pred.astype(np.float32)

        auc = roc_auc_score(y[va], pred) if len(np.unique(y[va])) == 2 else np.nan
        ap = average_precision_score(y[va], pred) if len(np.unique(y[va])) == 2 else np.nan
        fold_rows.append({
            "fold": fold,
            "model_kind": kind,
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "auc": float(auc) if not np.isnan(auc) else np.nan,
            "ap": float(ap) if not np.isnan(ap) else np.nan,
            "valid_pos_rate": float(y[va].mean()),
        })
        print("fold", fold, "auc", auc, "ap", ap, "n_valid", len(va))

    overall_auc = roc_auc_score(y, oof) if len(np.unique(y)) == 2 else np.nan
    overall_ap = average_precision_score(y, oof) if len(np.unique(y)) == 2 else np.nan

    # Threshold diagnostics
    best_f1 = -1
    best_t = 0.5
    for t in np.linspace(0.10, 0.90, 81):
        p = (oof >= t).astype(int)
        f1 = f1_score(y, p, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    print("OOF auc:", overall_auc, "ap:", overall_ap, "best_macro_f1:", best_f1, "best_t:", best_t)

    oof_df = lab[["id", "label", "sample_weight", "label_sources", "label_margin", "label_dup_count"]].copy()
    oof_df["v41_oof_prob"] = oof
    oof_df.to_csv(OUT_LABEL_OOF, index=False)

    info = pd.DataFrame(fold_rows + [{
        "fold": "OOF",
        "model_kind": fold_rows[0]["model_kind"] if fold_rows else "",
        "n_train": int(len(y)),
        "n_valid": int(len(y)),
        "auc": float(overall_auc) if not np.isnan(overall_auc) else np.nan,
        "ap": float(overall_ap) if not np.isnan(overall_ap) else np.nan,
        "valid_pos_rate": float(y.mean()),
        "best_macro_f1": float(best_f1),
        "best_threshold": float(best_t),
        "n_features": int(len(feature_cols)),
        "n_labels": int(len(y)),
    }])
    info.to_csv(OUT_MODEL_INFO, index=False)

    # Final model on all labels
    final_model, kind = make_model(999)
    final_model = fit_model(final_model, kind, X_lab, y, w)

    # Score all pairs in chunks to avoid memory spike.
    print("scoring full submission with v41 learned meta model...")
    probs = np.zeros(len(df), dtype=np.float32)
    chunk = 250_000
    for start in range(0, len(df), chunk):
        end = min(len(df), start + chunk)
        X = df.iloc[start:end][feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        probs[start:end] = predict_proba_model(final_model, kind, X).astype(np.float32)
        print("scored", end, "/", len(df))

    out_prob = pd.DataFrame({
        "id": sample["id"].to_numpy(),
        "v41_meta_prob": probs,
    })
    out_prob.to_parquet(OUT_PROB, index=False)

    return probs, {
        "oof_auc": float(overall_auc) if not np.isnan(overall_auc) else np.nan,
        "oof_ap": float(overall_ap) if not np.isnan(overall_ap) else np.nan,
        "oof_best_macro_f1": float(best_f1),
        "oof_best_threshold": float(best_t),
        "model_kind": kind,
        "n_labels": int(len(y)),
        "n_features": int(len(feature_cols)),
    }


def build_same_quota(df, prob, anchor):
    tmp = pd.DataFrame({"term_id": df["term_id"].to_numpy(), "prob": prob, "anchor": anchor})
    quota = tmp.groupby("term_id")["anchor"].sum()
    tmp["quota"] = tmp["term_id"].map(quota).fillna(0).astype(np.int32)
    tmp["rank"] = tmp.groupby("term_id")["prob"].rank(method="first", ascending=False).astype(np.int32)
    return (tmp["rank"].to_numpy() <= tmp["quota"].to_numpy()).astype(np.int8)


def pair_swaps(df, sample, anchor, base, prob):
    add_mask = (anchor == 0) & (base == 1)
    drop_mask = (anchor == 1) & (base == 0)

    cols = [
        "id", "term_id", "item_id",
        "v33_rs", "v34_rs", "v38_rs", "v39_rs",
        "v38_lex_score", "v38_word_overlap_title", "v38_must_token_coverage_full",
        "v38_query_word_only_category_ratio",
        "v39_hist_score", "v39_hist_has_item", "v39_hist_weighted_cov",
        "pred_vote_mean", "pred_vote_std", "rank_mean", "rank_std", "sem_text_gap",
    ]

    local = df[cols].copy()
    local["prob"] = prob

    add = local.loc[add_mask].copy()
    drop = local.loc[drop_mask].copy()

    add = add.sort_values(["term_id", "prob"], ascending=[True, False])
    drop = drop.sort_values(["term_id", "prob"], ascending=[True, True])

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    sw = add.merge(drop, on=["term_id", "pair_rank"], suffixes=("_add", "_drop"), how="inner")
    sw["v41_prob_gain"] = sw["prob_add"] - sw["prob_drop"]
    sw["sem_gain"] = (sw["v33_rs_add"] + sw["v34_rs_add"]) / 2 - (sw["v33_rs_drop"] + sw["v34_rs_drop"]) / 2
    sw["lex_gain"] = sw["v38_lex_score_add"] - sw["v38_lex_score_drop"]
    sw["hist_gain"] = sw["v39_hist_score_add"] - sw["v39_hist_score_drop"]

    sw["v41_support_score"] = (
        1.00 * sw["v41_prob_gain"]
        + 0.18 * sw["sem_gain"]
        + 0.08 * sw["lex_gain"]
        + 0.05 * sw["hist_gain"]
        - 0.10 * (
            (sw["v38_query_word_only_category_ratio_add"] >= 0.50)
            & (sw["v38_word_overlap_title_add"] <= 0.10)
        ).astype(float)
    ).astype(np.float32)

    return sw.sort_values(["v41_support_score", "v41_prob_gain"], ascending=False).reset_index(drop=True)


def v41_mask(sw, mode):
    add_bad_category = (
        (sw["v38_query_word_only_category_ratio_add"] >= 0.50)
        & (sw["v38_word_overlap_title_add"] <= 0.10)
        & (sw["prob_add"] < 0.80)
    )

    history_contra = (
        (sw["v39_hist_has_item_add"].astype(int) == 1)
        & (sw["v39_hist_has_item_drop"].astype(int) == 1)
        & (sw["v39_hist_score_drop"] >= sw["v39_hist_score_add"] + 0.12)
        & (sw["v39_hist_weighted_cov_drop"] >= sw["v39_hist_weighted_cov_add"] + 0.18)
    )

    if mode == "ultra":
        return (
            (sw["prob_add"] >= 0.82)
            & (sw["prob_drop"] <= 0.48)
            & (sw["v41_prob_gain"] >= 0.26)
            & (sw["v41_support_score"] >= 0.25)
            & (~add_bad_category)
            & (~history_contra)
        )

    if mode == "strict":
        return (
            (sw["prob_add"] >= 0.74)
            & (sw["prob_drop"] <= 0.53)
            & (sw["v41_prob_gain"] >= 0.18)
            & (sw["v41_support_score"] >= 0.17)
            & (~add_bad_category)
            & (~history_contra)
        )

    if mode == "balanced":
        return (
            (sw["prob_add"] >= 0.66)
            & (sw["prob_drop"] <= 0.58)
            & (sw["v41_prob_gain"] >= 0.11)
            & (sw["v41_support_score"] >= 0.10)
            & (~add_bad_category)
        )

    if mode == "impact":
        return (
            (sw["prob_add"] >= 0.58)
            & (sw["v41_prob_gain"] >= 0.06)
            & (sw["v41_support_score"] >= 0.06)
            & (~add_bad_category)
        )

    raise ValueError(mode)


def apply_swaps(sample, anchor, sw, cap):
    take = sw.sort_values(["v41_support_score", "v41_prob_gain"], ascending=False).head(cap).copy()
    pred = anchor.copy()
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])

    add_idx = take["id_add"].astype(str).map(id_to_idx)
    drop_idx = take["id_drop"].astype(str).map(id_to_idx)

    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("id mapping failed in apply_swaps")

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
        parts.append(x.head(100))
        parts.append(x.tail(70))

    if not parts:
        return

    r = pd.concat(parts, ignore_index=True).drop_duplicates(["variant", "id_add", "id_drop"], keep="first")
    keep = [
        "variant", "term_id", "pair_rank",
        "id_add", "item_id_add", "prob_add", "id_drop", "item_id_drop", "prob_drop",
        "v41_prob_gain", "v41_support_score", "sem_gain", "lex_gain", "hist_gain",
        "v33_rs_add", "v33_rs_drop", "v34_rs_add", "v34_rs_drop",
        "v38_lex_score_add", "v38_lex_score_drop",
        "v39_hist_score_add", "v39_hist_score_drop",
        "pred_vote_mean_add", "pred_vote_mean_drop",
    ]
    r[[c for c in keep if c in r.columns]].to_csv(OUT_REVIEW, index=False)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

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
        raise FileNotFoundError("anchor v33 qprob2000 missing")
    anchor = load_pred(anchor_path, sample)

    baselines = {"anchor_v33_qprob2000": anchor.copy()}

    for name, paths in DIRECT_BASES.items():
        p = first_existing(paths)
        if p is not None:
            baselines[name] = load_pred(p, sample)
            print("baseline:", name, p, "diff_vs_anchor", int((baselines[name] != anchor).sum()))

    for short_name, variant in SAVED_VARIANTS.items():
        p = find_saved_variant(variant)
        if p is not None:
            baselines[short_name] = load_pred(p, sample)
            print("saved baseline:", short_name, p, "diff_vs_anchor", int((baselines[short_name] != anchor).sum()))

    df = build_features(sample, pairs, baselines)
    feature_cols = get_feature_columns(df)
    labels = load_labels()

    probs, model_info = train_oof_and_score_all(df, labels, feature_cols, sample)
    df["v41_meta_prob"] = probs

    variants = dict(baselines)
    meta = []
    chosen = {}

    # Same quota is truly new model output; risky but include.
    same = build_same_quota(df, probs, anchor)
    variants["v41_meta_same_quota"] = same
    meta.append({
        "variant": "v41_meta_same_quota",
        "source": "learned_meta_same_quota",
        "base": "anchor",
        "mode": "",
        "cap": "",
        "used_swaps": int((same != anchor).sum() // 2),
        "accepted_pool": "",
        "prob_gain_mean": np.nan,
        "prob_add_min": np.nan,
        "prob_drop_max": np.nan,
    })

    bases = [
        "raw_v35_b5000",
        "v40_history_impact2200",
        "v40_text_impact5200",
        "v39_raw35_history_lex_veto_bad",
        "v38_precision_loose_2800",
        "v38_veto_bad_only",
        "v36p2_balanced",
        "v36p2_strict",
    ]
    bases = [b for b in bases if b in baselines]

    caps = {
        "ultra": [500, 900, 1300],
        "strict": [1000, 1600, 2400],
        "balanced": [1600, 2600, 4000],
        "impact": [2400, 4000, 6500],
    }

    for base in bases:
        base_pred = baselines[base]
        sw = pair_swaps(df, sample, anchor, base_pred, probs)
        print("swap pool", base, len(sw))

        for mode, cap_list in caps.items():
            m = v41_mask(sw, mode)
            acc = sw[m].copy()
            if len(acc) == 0:
                continue

            for cap in cap_list:
                pred, take = apply_swaps(sample, anchor, acc, min(cap, len(acc)))
                name = f"v41_{base}_{mode}_cap{cap}"
                variants[name] = pred
                chosen[name] = take

                meta.append({
                    "variant": name,
                    "source": "learned_meta_swap_filter",
                    "base": base,
                    "mode": mode,
                    "cap": cap,
                    "accepted_pool": int(len(acc)),
                    "used_swaps": int(len(take)),
                    "prob_gain_mean": float(take["v41_prob_gain"].mean()) if len(take) else np.nan,
                    "prob_add_min": float(take["prob_add"].min()) if len(take) else np.nan,
                    "prob_drop_max": float(take["prob_drop"].max()) if len(take) else np.nan,
                    "support_mean": float(take["v41_support_score"].mean()) if len(take) else np.nan,
                    "support_min": float(take["v41_support_score"].min()) if len(take) else np.nan,
                })
                print("candidate", name, "used", len(take), "diff", int((pred != anchor).sum()))

    print("evaluating candidates...")
    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)
    wdf = weighted_score(eval_df)

    summary = pd.DataFrame(meta)

    for name, pred in baselines.items():
        summary = pd.concat([summary, pd.DataFrame([{
            "variant": name,
            "source": "baseline",
            "base": "",
            "mode": "",
            "cap": "",
            "accepted_pool": "",
            "used_swaps": "",
            "prob_gain_mean": np.nan,
            "prob_add_min": np.nan,
            "prob_drop_max": np.nan,
            "support_mean": np.nan,
            "support_min": np.nan,
        }])], ignore_index=True)

    aux = []
    for name, pred in variants.items():
        aux.append({
            "variant": name,
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != baselines["raw_v35_b5000"]).sum()) if "raw_v35_b5000" in baselines else -1,
            "diff_vs_v40_history": int((pred != baselines["v40_history_impact2200"]).sum()) if "v40_history_impact2200" in baselines else -1,
            "diff_vs_v38_precision": int((pred != baselines["v38_precision_loose_2800"]).sum()) if "v38_precision_loose_2800" in baselines else -1,
            "diff_vs_v36p2_balanced": int((pred != baselines["v36p2_balanced"]).sum()) if "v36p2_balanced" in baselines else -1,
        })

    summary = summary.merge(pd.DataFrame(aux), on="variant", how="right")

    if len(wdf):
        summary = summary.merge(wdf, on="variant", how="left")

    for k, v in model_info.items():
        summary[k] = v

    diff = summary["diff_vs_anchor"].fillna(0).astype(float)
    move_bonus = np.minimum(0.018, np.log1p(diff) / np.log1p(30000) * 0.018)
    prob_bonus = np.minimum(0.004, summary["prob_gain_mean"].fillna(0).astype(float) * 0.010)
    overfit_penalty = max(0.0, 0.88 - float(model_info["oof_auc"])) * 0.020 if not pd.isna(model_info["oof_auc"]) else 0.010
    too_big_penalty = np.maximum(0, diff - 70000) / 750000.0

    summary["v41_decision_score"] = (
        summary["weighted_macro"].fillna(0)
        + move_bonus
        + prob_bonus
        - too_big_penalty
        - overfit_penalty
    )

    summary = summary.sort_values(["v41_decision_score", "weighted_macro"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    build_review(chosen)

    save_names = []
    top = summary[
        (summary["source"].isin(["learned_meta_swap_filter", "learned_meta_same_quota"]))
        & (summary["diff_vs_anchor"] >= 1500)
        & (summary["diff_vs_anchor"] <= 90000)
    ].head(40)

    for nm in top["variant"].tolist():
        if nm not in save_names:
            save_names.append(nm)

    for nm in [
        "v41_meta_same_quota",
        "raw_v35_b5000",
        "v40_history_impact2200",
        "v38_precision_loose_2800",
        "v36p2_balanced",
        "v36p2_strict",
        "anchor_v33_qprob2000",
    ]:
        if nm in variants and nm not in save_names:
            save_names.append(nm)

    saved = []
    for i, nm in enumerate(save_names[:50], 1):
        pred = variants[nm]
        out = SUB_DIR / f"FINAL_CANDIDATE_v41_meta_{i:03d}_{short_hash(nm)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

        row = summary[summary["variant"] == nm].iloc[0].to_dict()
        row["file"] = str(out)
        saved.append(row)
        print("saved", out, "<-", nm)

    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)

    print("\nMODEL INFO")
    print(model_info)

    print("\nTOP V41")
    cols = [
        "variant", "v41_decision_score", "weighted_macro", "source", "base", "mode", "cap",
        "used_swaps", "accepted_pool", "prob_gain_mean", "prob_add_min", "prob_drop_max",
        "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v40_history",
        "oof_auc", "oof_ap", "oof_best_macro_f1", "main_min_macro", "main_mean_macro",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))

    print("\nSAVED")
    sdf = pd.DataFrame(saved)
    if len(sdf):
        print(sdf[["variant", "file", "v41_decision_score", "weighted_macro", "diff_vs_anchor"]].to_string(index=False))

    print("\noutputs:")
    print(OUT_PROB)
    print(OUT_MODEL_INFO)
    print(OUT_LABEL_OOF)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_SAVED)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
