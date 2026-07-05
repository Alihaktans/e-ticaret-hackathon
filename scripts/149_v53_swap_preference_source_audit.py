from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
OUT = ROOT / "reports/manual_review/v53_swap_preference_source_audit.csv"
OUT_FEATURES = ROOT / "data/processed/v53_swap_preference_features.parquet"
OUT_INFO = ROOT / "reports/manual_review/v53_swap_preference_source_audit.json"

LABEL_SPECS = [
    ("v29", ROOT / "reports/manual_review/review_v29_qswap_targets_assistant_labeled.csv"),
    ("v33", ROOT / "reports/manual_review/review_v33_final_qswap_targets_assistant_labeled.csv"),
]

SCORE_SPECS = [
    (ROOT / "data/processed/v33_ft_pair_scores.parquet", ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"]),
    (ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet", ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"]),
    (ROOT / "data/processed/v38_lexical_pair_scores.parquet", [
        "v38_lex_score", "v38_lex_pct_rank", "v38_lex_term_z", "v38_exact_phrase_title",
        "v38_exact_phrase_full", "v38_word_overlap_title", "v38_word_overlap_category",
        "v38_word_overlap_full", "v38_word_jaccard_title", "v38_char_ngram_sim",
        "v38_must_token_coverage_full", "v38_special_token_coverage_full", "v38_brand_match",
        "v38_number_match", "v38_model_match", "v38_category_title_support",
        "v38_query_word_only_category_ratio", "v38_title_missing_ratio", "v38_must_missing_ratio",
    ]),
    (ROOT / "data/processed/v39_item_history_pair_scores.parquet", [
        "v39_hist_score", "v39_hist_has_item", "v39_hist_count", "v39_hist_token_cov",
        "v39_hist_weighted_cov", "v39_hist_must_cov", "v39_hist_special_cov",
        "v39_hist_exact_query", "v39_hist_char_cov", "v39_hist_number_match",
        "v39_hist_model_match", "v39_hist_pct_rank", "v39_hist_term_z",
    ]),
    (ROOT / "data/processed/v41_learned_meta_prob.parquet", ["v41_meta_prob"]),
    (ROOT / "data/processed/v42_query_graph_pair_scores.parquet", [
        "v42_graph_score", "v42_graph_norm_score", "v42_graph_max_sim", "v42_graph_hit_count",
        "v42_graph_char_score", "v42_graph_word_score", "v42_graph_pct_rank", "v42_graph_term_z",
    ]),
    (ROOT / "data/processed/v43_pop_brand_category_pair_scores.parquet", [
        "v43_item_pop", "v43_brand_pop", "v43_category_pop", "v43_qbrand_affinity",
        "v43_qcategory_affinity", "v43_brand_query_hit", "v43_category_query_hit",
        "v43_query_pop_support", "v43_popularity_trap", "v43_pop_prior_score", "v43_pop_pct_rank",
    ]),
    (ROOT / "data/processed/v43_pop_brand_category_meta_prob.parquet", ["v43_meta_prob"]),
    (ROOT / "data/processed/v47_reciprocal_shadow_scores.parquet", [
        "v47_forward_base", "v47_reverse_base", "v47_forward_pct", "v47_reverse_pct",
        "v47_item_query_count", "v47_shadow_conflict", "v47_mutual_score", "v47_mutual_strict_score",
    ]),
    (ROOT / "data/processed/v48_train_memory_bridge_scores.parquet", [
        "mem_exact_item_hit", "mem_fuzzy_item_hit", "mem_fuzzy_item_sim",
        "train_item_pos_count", "mem_hit_strength", "model_support", "v48_memory_score",
    ]),
]


def load_labels():
    parts = []
    for source, path in LABEL_SPECS:
        d = pd.read_csv(path)
        label = pd.to_numeric(d["assistant_swap_label"], errors="coerce")
        clean = pd.to_numeric(d.get("needs_recheck", 0), errors="coerce").fillna(1).eq(0)
        confidence = d.get("assistant_confidence", "").astype(str).str.lower().isin(["high", "medium"])
        keep = label.isin([0, 1]) & clean & confidence
        d = d.loc[keep].copy()
        d["label"] = label.loc[keep].astype(np.int8)
        d["source"] = source
        d["swap_key"] = source + "_" + d.index.astype(str)
        parts.append(d[["swap_key", "source", "term_id", "query", "id_add", "id_drop", "label", "assistant_confidence"]])
        print(source, "clean labels", len(d), "positive rate", float(d.label.mean()))
    return pd.concat(parts, ignore_index=True)


def build_features(labels):
    ids = set(labels.id_add.astype(str)) | set(labels.id_drop.astype(str))
    feature = pd.DataFrame({"id": sorted(ids)})
    for path, cols in SCORE_SPECS:
        if not path.exists():
            print("missing", path)
            continue
        schema_cols = pq.ParquetFile(path).schema.names
        use = [c for c in cols if c in schema_cols]
        if "id" not in schema_cols or not use:
            continue
        available = pd.read_parquet(path, columns=["id"] + use)
        available["id"] = available.id.astype(str)
        small = available[available.id.isin(ids)][["id"] + use].drop_duplicates("id")
        feature = feature.merge(small, on="id", how="left", validate="one_to_one")
        print("loaded", path.name, "features", len(use), "matched", len(small))
        del available, small

    add = feature.rename(columns={"id": "id_add", **{c: f"add_{c}" for c in feature.columns if c != "id"}})
    drop = feature.rename(columns={"id": "id_drop", **{c: f"drop_{c}" for c in feature.columns if c != "id"}})
    swaps = labels.merge(add, on="id_add", how="left").merge(drop, on="id_drop", how="left")
    base_cols = [c for c in feature.columns if c != "id"]
    for col in base_cols:
        swaps[f"diff_{col}"] = pd.to_numeric(swaps[f"add_{col}"], errors="coerce") - pd.to_numeric(
            swaps[f"drop_{col}"], errors="coerce"
        )
    swaps.to_parquet(OUT_FEATURES, index=False)
    return swaps, [f"diff_{c}" for c in base_cols]


def metrics(y, prob, threshold=.5):
    pred = (prob >= threshold).astype(np.int8)
    return {
        "auc": float(roc_auc_score(y, prob)),
        "ap": float(average_precision_score(y, prob)),
        "macro_f1_0p5": float(f1_score(y, pred, average="macro")),
        "pred_positive_rate": float(pred.mean()),
        "true_positive_rate": float(y.mean()),
    }


def main():
    labels = load_labels()
    frame, features = build_features(labels)
    rows = []
    for train_source, test_source in [("v29", "v33"), ("v33", "v29")]:
        train = frame[frame.source.eq(train_source)].copy()
        test = frame[frame.source.eq(test_source)].copy()
        # Remove any pair IDs appearing in the held-out source to prevent cross-source leakage.
        heldout_ids = set(test.id_add) | set(test.id_drop)
        train = train[~train.id_add.isin(heldout_ids) & ~train.id_drop.isin(heldout_ids)]
        xtr, ytr = train[features], train.label.to_numpy()
        xte, yte = test[features], test.label.to_numpy()
        models = {
            "logistic": make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=.25, max_iter=1000, class_weight="balanced", random_state=42),
            ),
            "hgb": make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(
                    max_iter=220, learning_rate=.04, max_leaf_nodes=15,
                    min_samples_leaf=18, l2_regularization=6, random_state=42,
                ),
            ),
        }
        for name, model in models.items():
            model.fit(xtr, ytr)
            prob = model.predict_proba(xte)[:, 1]
            row = {
                "train_source": train_source, "test_source": test_source, "model": name,
                "n_train": len(train), "n_test": len(test), "n_features": len(features),
            }
            row.update(metrics(yte, prob))
            rows.append(row)
            print(row)
        # Single-signal references.
        for col in [
            "diff_v33_ft_score", "diff_v34_ce_score", "diff_v38_lex_score",
            "diff_v41_meta_prob", "diff_v42_graph_score", "diff_v43_meta_prob",
            "diff_v47_mutual_score", "diff_model_support",
        ]:
            if col not in test:
                continue
            raw = pd.to_numeric(test[col], errors="coerce").fillna(0).to_numpy()
            # Convert arbitrary score to a monotonic 0-1 rank for common metrics.
            prob = pd.Series(raw).rank(pct=True).to_numpy()
            row = {
                "train_source": "none", "test_source": test_source, "model": col,
                "n_train": 0, "n_test": len(test), "n_features": 1,
            }
            row.update(metrics(yte, prob))
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    info = {
        "features": features,
        "n_features": len(features),
        "label_counts": frame.groupby("source").size().to_dict(),
        "results": rows,
        "warning": "Assistant swap labels may encode shared heuristic bias; source holdout reduces but does not remove it.",
    }
    OUT_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRESULTS")
    print(result.sort_values(["test_source", "auc"], ascending=[True, False]).to_string(index=False))
    print("saved", OUT, OUT_FEATURES, OUT_INFO)


if __name__ == "__main__":
    main()
