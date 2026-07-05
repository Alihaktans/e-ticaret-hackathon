from pathlib import Path
import re
import json
import math
import hashlib
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TRAIN = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"

V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"
V38 = ROOT / "data/processed/v38_lexical_pair_scores.parquet"
V39 = ROOT / "data/processed/v39_item_history_pair_scores.parquet"
V41 = ROOT / "data/processed/v41_learned_meta_prob.parquet"

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
    ROOT / "reports/manual_review/v41_learned_meta_saved_candidates.csv",
]

SAVED_VARIANTS = {
    "v38_precision_loose_2800": "v38_raw_v35_b5000_v38_precision_score_loose_cap2800",
    "v39_raw35_history_lex_veto_bad": "v39_raw_v35_b5000_v39_history_lex_score_history_veto_contradiction_and_bad",
    "v40_history_impact2200": "v40_raw_v35_b5000_v40_history_score_impact_cap2200",
    "v41_meta_same_quota": "v41_meta_same_quota",
    "v41_raw35_balanced": "v41_raw_v35_b5000_balanced_cap4000",
    "v41_raw35_strict": "v41_raw_v35_b5000_strict_cap1000",
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

PROC_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"

OUT_SCORE = PROC_DIR / "v42_query_graph_pair_scores.parquet"
OUT_INFO = PROC_DIR / "v42_query_graph_info.json"
OUT_SUMMARY = REPORT_DIR / "v42_query_graph_candidate_summary.csv"
OUT_EVAL = REPORT_DIR / "v42_query_graph_candidate_eval.csv"
OUT_SAVED = REPORT_DIR / "v42_query_graph_saved_candidates.csv"
OUT_REVIEW = REPORT_DIR / "review_v42_query_graph_swaps.csv"


TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


def norm_text(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


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


def build_query_graph_scores(sample, pairs):
    print("building V42 query-neighbor graph propagation...")

    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)

    if "query" not in terms.columns:
        # fallback: use the first non-id object column.
        candidates = [c for c in terms.columns if c != "term_id" and terms[c].dtype == "object"]
        if not candidates:
            raise RuntimeError("Could not find query text column in terms.csv")
        terms = terms.rename(columns={candidates[0]: "query"})

    train = pd.read_csv(TRAIN, usecols=["term_id", "item_id"])
    train["term_id"] = train["term_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    test_terms = pairs["term_id"].astype(str).drop_duplicates().tolist()
    train_terms = train["term_id"].drop_duplicates().tolist()

    qmap = dict(zip(terms["term_id"], terms["query"].astype(str)))
    train_terms = [t for t in train_terms if t in qmap]
    test_terms = [t for t in test_terms if t in qmap]

    print("train_terms:", len(train_terms), "test_terms:", len(test_terms))

    train_texts = [norm_text(qmap[t]) for t in train_terms]
    test_texts = [norm_text(qmap[t]) for t in test_terms]

    # Positive item lists per train query.
    pos_by_train_term = train.groupby("term_id")["item_id"].apply(list).to_dict()

    # Candidate row map per test query.
    print("building candidate row maps...")
    cand_maps = {}
    for tid, g in pairs.groupby("term_id", sort=False):
        cand_maps[str(tid)] = dict(zip(g["item_id"].astype(str), g.index.to_numpy(np.int32)))

    n = len(pairs)
    graph_score = np.zeros(n, dtype=np.float32)
    graph_max_sim = np.zeros(n, dtype=np.float32)
    graph_hit_count = np.zeros(n, dtype=np.int16)
    graph_neighbor_count = np.zeros(n, dtype=np.int16)
    graph_char_score = np.zeros(n, dtype=np.float32)
    graph_word_score = np.zeros(n, dtype=np.float32)

    def run_channel(name, analyzer, ngram_range, max_features, weight, min_sim, k):
        print(f"vectorizing {name} tfidf...")
        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)

        nn = NearestNeighbors(n_neighbors=min(k, X_train.shape[0]), metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(X_train)

        batch = 512
        for start in range(0, len(test_terms), batch):
            end = min(len(test_terms), start + batch)
            dist, ind = nn.kneighbors(X_test[start:end], return_distance=True)

            for local_i, tid in enumerate(test_terms[start:end]):
                cmap = cand_maps.get(str(tid))
                if not cmap:
                    continue

                # Aggregate neighboring train queries by item.
                for d, j in zip(dist[local_i], ind[local_i]):
                    sim = 1.0 - float(d)
                    if sim < min_sim:
                        continue

                    tr_tid = train_terms[int(j)]
                    positives = pos_by_train_term.get(tr_tid, [])
                    if not positives:
                        continue

                    val = weight * (sim ** 1.35)
                    for iid in positives:
                        row = cmap.get(str(iid))
                        if row is None:
                            continue

                        graph_score[row] += val
                        graph_max_sim[row] = max(graph_max_sim[row], sim)
                        graph_hit_count[row] = min(32767, graph_hit_count[row] + 1)
                        graph_neighbor_count[row] = min(32767, graph_neighbor_count[row] + 1)

                        if name == "char":
                            graph_char_score[row] += val
                        else:
                            graph_word_score[row] += val

            if end % 4096 == 0 or end == len(test_terms):
                print(name, "neighbors", end, "/", len(test_terms))

    # Character channel catches misspelling/model variants; word channel catches exact concepts.
    run_channel("char", analyzer="char_wb", ngram_range=(3, 5), max_features=220_000, weight=0.62, min_sim=0.42, k=90)
    run_channel("word", analyzer="word", ngram_range=(1, 2), max_features=120_000, weight=0.38, min_sim=0.25, k=70)

    # Normalize repeated hits.
    norm = np.sqrt(np.maximum(1, graph_hit_count.astype(np.float32)))
    graph_norm_score = graph_score / norm

    out = pd.DataFrame({
        "id": sample["id"].to_numpy(),
        "term_id": pairs["term_id"].astype(str).to_numpy(),
        "item_id": pairs["item_id"].astype(str).to_numpy(),
        "v42_graph_score": graph_score,
        "v42_graph_norm_score": graph_norm_score.astype(np.float32),
        "v42_graph_max_sim": graph_max_sim,
        "v42_graph_hit_count": graph_hit_count,
        "v42_graph_neighbor_count": graph_neighbor_count,
        "v42_graph_char_score": graph_char_score,
        "v42_graph_word_score": graph_word_score,
    })

    out["v42_graph_rank"] = (
        out.groupby("term_id")["v42_graph_norm_score"]
        .rank(method="first", ascending=False)
        .astype(np.int32)
    )

    cnt = out.groupby("term_id")["id"].transform("count").astype(np.float32)
    out["v42_graph_pct_rank"] = ((out["v42_graph_rank"].astype(np.float32) - 1.0) / np.maximum(1.0, cnt - 1.0)).astype(np.float32)

    term_mean = out.groupby("term_id")["v42_graph_norm_score"].transform("mean").astype(np.float32)
    term_std = out.groupby("term_id")["v42_graph_norm_score"].transform("std").fillna(0).astype(np.float32)
    out["v42_graph_term_z"] = ((out["v42_graph_norm_score"].astype(np.float32) - term_mean) / np.maximum(1e-6, term_std)).astype(np.float32)

    out.to_parquet(OUT_SCORE, index=False)

    info = {
        "rows": int(len(out)),
        "unique_terms": int(out["term_id"].nunique()),
        "unique_items": int(out["item_id"].nunique()),
        "nonzero_rate": float((out["v42_graph_score"] > 0).mean()),
        "score_mean": float(out["v42_graph_score"].mean()),
        "score_std": float(out["v42_graph_score"].std()),
        "score_max": float(out["v42_graph_score"].max()),
        "norm_score_mean": float(out["v42_graph_norm_score"].mean()),
        "norm_score_max": float(out["v42_graph_norm_score"].max()),
        "max_sim_mean": float(out["v42_graph_max_sim"].mean()),
        "max_sim_max": float(out["v42_graph_max_sim"].max()),
    }
    OUT_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    return out


def load_or_build_v42(sample, pairs):
    if OUT_SCORE.exists():
        print("using existing V42:", OUT_SCORE)
        out = pd.read_parquet(OUT_SCORE)
        out["id"] = out["id"].astype(str)
        if not out["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            raise RuntimeError("existing V42 id order mismatch")
        return out
    return build_query_graph_scores(sample, pairs)


def build_feature_frame(sample, pairs, v42, baselines):
    df = pairs[["id", "term_id", "item_id"]].copy()

    for c in [
        "v42_graph_score", "v42_graph_norm_score", "v42_graph_max_sim",
        "v42_graph_hit_count", "v42_graph_neighbor_count",
        "v42_graph_char_score", "v42_graph_word_score",
        "v42_graph_pct_rank", "v42_graph_term_z",
    ]:
        df[c] = pd.to_numeric(v42[c], errors="coerce").fillna(0).astype(np.float32)

    if V33.exists():
        x = align_parquet(V33, sample, ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"])
        for c in x.columns:
            if c != "id":
                df[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(np.float32)
    else:
        for c in ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"]:
            df[c] = 0.0

    if V34.exists():
        x = align_parquet(V34, sample, ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"])
        for c in x.columns:
            if c != "id":
                df[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(np.float32)
    else:
        for c in ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"]:
            df[c] = 0.0

    if V38.exists():
        x = align_parquet(V38, sample, [
            "v38_lex_score", "v38_lex_pct_rank", "v38_word_overlap_title",
            "v38_must_token_coverage_full", "v38_query_word_only_category_ratio",
        ])
        for c in x.columns:
            if c != "id":
                df[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(np.float32)
    else:
        for c in ["v38_lex_score", "v38_lex_pct_rank", "v38_word_overlap_title", "v38_must_token_coverage_full", "v38_query_word_only_category_ratio"]:
            df[c] = 0.0

    if V39.exists():
        x = align_parquet(V39, sample, ["v39_hist_score", "v39_hist_pct_rank", "v39_hist_has_item", "v39_hist_weighted_cov"])
        for c in x.columns:
            if c != "id":
                df[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(np.float32)
    else:
        for c in ["v39_hist_score", "v39_hist_pct_rank", "v39_hist_has_item", "v39_hist_weighted_cov"]:
            df[c] = 0.0

    if V41.exists():
        x = align_parquet(V41, sample, ["v41_meta_prob"])
        df["v41_meta_prob"] = pd.to_numeric(x["v41_meta_prob"], errors="coerce").fillna(0).astype(np.float32)
    else:
        df["v41_meta_prob"] = 0.0

    df["v33_rs"] = (1.0 - df["v33_ft_pct_rank"]).clip(0, 1)
    df["v34_rs"] = (1.0 - df["v34_ce_pct_rank"]).clip(0, 1)
    df["v38_rs"] = (1.0 - df["v38_lex_pct_rank"]).clip(0, 1)
    df["v39_rs"] = (1.0 - df["v39_hist_pct_rank"]).clip(0, 1)
    df["v42_rs"] = (1.0 - df["v42_graph_pct_rank"]).clip(0, 1)

    df["v42_present"] = (df["v42_graph_score"] > 0).astype(np.int8)
    df["v42_strong_present"] = ((df["v42_graph_score"] > 0) & (df["v42_graph_max_sim"] >= 0.55)).astype(np.int8)

    # New graph-aware scoring blends. These are intentionally not just semantic ranking.
    df["v42_graph_only_score"] = (
        0.58 * df["v42_rs"] +
        0.16 * np.tanh(df["v42_graph_norm_score"] / 1.5) +
        0.13 * df["v42_graph_max_sim"] +
        0.08 * np.log1p(df["v42_graph_hit_count"].astype(float)) +
        0.05 * df["v42_strong_present"]
    ).astype(np.float32)

    df["v42_graph_meta_score"] = (
        0.36 * df["v42_graph_only_score"] +
        0.28 * df["v41_meta_prob"] +
        0.16 * df["v34_rs"] +
        0.11 * df["v33_rs"] +
        0.06 * df["v38_rs"] +
        0.03 * df["v39_rs"]
    ).astype(np.float32)

    df["v42_graph_safe_score"] = (
        0.32 * df["v42_graph_only_score"] +
        0.22 * df["v41_meta_prob"] +
        0.18 * df["v38_rs"] +
        0.14 * df["v34_rs"] +
        0.10 * df["v33_rs"] +
        0.04 * df["v39_rs"]
    ).astype(np.float32)

    for name, pred in baselines.items():
        df[f"pred_{name}"] = pred.astype(np.int8)

    return df


def build_same_quota(df, score_col, anchor):
    tmp = pd.DataFrame({"term_id": df["term_id"].to_numpy(), "score": df[score_col].to_numpy(np.float32), "anchor": anchor})
    quota = tmp.groupby("term_id")["anchor"].sum()
    tmp["quota"] = tmp["term_id"].map(quota).fillna(0).astype(np.int32)
    tmp["rank"] = tmp.groupby("term_id")["score"].rank(method="first", ascending=False).astype(np.int32)
    return (tmp["rank"].to_numpy() <= tmp["quota"].to_numpy()).astype(np.int8)


def pair_swaps(df, anchor, base, score_col):
    add_mask = (anchor == 0) & (base == 1)
    drop_mask = (anchor == 1) & (base == 0)

    cols = [
        "id", "term_id", "item_id", score_col,
        "v42_graph_score", "v42_graph_norm_score", "v42_graph_max_sim",
        "v42_graph_hit_count", "v42_rs", "v42_present",
        "v41_meta_prob", "v33_rs", "v34_rs", "v38_rs", "v39_rs",
        "v38_lex_score", "v38_word_overlap_title", "v38_must_token_coverage_full",
        "v38_query_word_only_category_ratio",
        "v39_hist_score", "v39_hist_has_item", "v39_hist_weighted_cov",
    ]

    add = df.loc[add_mask, cols].copy()
    drop = df.loc[drop_mask, cols].copy()

    add = add.sort_values(["term_id", score_col], ascending=[True, False])
    drop = drop.sort_values(["term_id", score_col], ascending=[True, True])

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    sw = add.merge(drop, on=["term_id", "pair_rank"], suffixes=("_add", "_drop"), how="inner")

    sw["score_gain"] = sw[f"{score_col}_add"] - sw[f"{score_col}_drop"]
    sw["graph_gain"] = sw["v42_graph_only_score_add"] - sw["v42_graph_only_score_drop"] if f"v42_graph_only_score_add" in sw.columns else sw["v42_rs_add"] - sw["v42_rs_drop"]
    sw["meta_gain"] = sw["v41_meta_prob_add"] - sw["v41_meta_prob_drop"]
    sw["sem_gain"] = (sw["v33_rs_add"] + sw["v34_rs_add"]) / 2 - (sw["v33_rs_drop"] + sw["v34_rs_drop"]) / 2
    sw["lex_gain"] = sw["v38_lex_score_add"] - sw["v38_lex_score_drop"]
    sw["hist_gain"] = sw["v39_hist_score_add"] - sw["v39_hist_score_drop"]

    sw["category_only_risk"] = (
        (sw["v38_query_word_only_category_ratio_add"] >= 0.50)
        & (sw["v38_word_overlap_title_add"] <= 0.10)
        & (sw["v42_graph_score_add"] <= 0)
    )

    sw["history_contra"] = (
        (sw["v39_hist_has_item_add"].astype(int) == 1)
        & (sw["v39_hist_has_item_drop"].astype(int) == 1)
        & (sw["v39_hist_score_drop"] >= sw["v39_hist_score_add"] + 0.12)
        & (sw["v39_hist_weighted_cov_drop"] >= sw["v39_hist_weighted_cov_add"] + 0.18)
    )

    sw["graph_support_score"] = (
        0.58 * sw["score_gain"]
        + 0.20 * sw["meta_gain"]
        + 0.12 * sw["sem_gain"]
        + 0.08 * sw["lex_gain"]
        + 0.05 * sw["hist_gain"]
        + 0.10 * ((sw["v42_graph_score_add"] > 0) & (sw["v42_graph_score_drop"] <= 0)).astype(float)
        - 0.12 * sw["category_only_risk"].astype(float)
        - 0.12 * sw["history_contra"].astype(float)
    ).astype(np.float32)

    return sw.sort_values(["graph_support_score", "score_gain"], ascending=False).reset_index(drop=True)


def graph_mask(sw, mode):
    if mode == "ultra":
        return (
            (sw["v42_graph_score_add"] > 0)
            & (sw["v42_graph_max_sim_add"] >= 0.55)
            & (sw["meta_gain"] >= 0.10)
            & (sw["score_gain"] >= 0.08)
            & (~sw["category_only_risk"])
            & (~sw["history_contra"])
        )

    if mode == "strict":
        return (
            (sw["v42_graph_score_add"] > 0)
            & (sw["v42_graph_max_sim_add"] >= 0.45)
            & (sw["meta_gain"] >= 0.05)
            & (sw["score_gain"] >= 0.05)
            & (~sw["category_only_risk"])
        )

    if mode == "balanced":
        return (
            (
                ((sw["v42_graph_score_add"] > 0) & (sw["score_gain"] >= 0.03))
                | ((sw["meta_gain"] >= 0.15) & (sw["v42_graph_score_add"] >= sw["v42_graph_score_drop"]))
            )
            & (~sw["category_only_risk"])
        )

    if mode == "impact":
        return (
            (
                (sw["graph_support_score"] >= 0.04)
                | ((sw["v42_graph_score_add"] > 0) & (sw["meta_gain"] >= 0.02))
            )
            & (~sw["category_only_risk"])
        )

    raise ValueError(mode)


def apply_swaps(sample, anchor, sw, cap):
    take = sw.sort_values(["graph_support_score", "score_gain"], ascending=False).head(cap).copy()
    pred = anchor.copy()
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])

    add_idx = take["id_add"].astype(str).map(id_to_idx)
    drop_idx = take["id_drop"].astype(str).map(id_to_idx)

    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("id map fail")

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

    r = pd.concat(parts, ignore_index=True).drop_duplicates(["variant", "id_add", "id_drop"], keep="first")
    keep = [
        "variant", "term_id", "pair_rank",
        "id_add", "item_id_add", "id_drop", "item_id_drop",
        "graph_support_score", "score_gain", "graph_gain", "meta_gain", "sem_gain", "lex_gain", "hist_gain",
        "v42_graph_score_add", "v42_graph_score_drop",
        "v42_graph_max_sim_add", "v42_graph_max_sim_drop",
        "v42_graph_hit_count_add", "v42_graph_hit_count_drop",
        "v41_meta_prob_add", "v41_meta_prob_drop",
        "v33_rs_add", "v33_rs_drop", "v34_rs_add", "v34_rs_drop",
        "v38_lex_score_add", "v38_lex_score_drop",
        "v39_hist_score_add", "v39_hist_score_drop",
        "category_only_risk", "history_contra",
    ]
    r[[c for c in keep if c in r.columns]].to_csv(OUT_REVIEW, index=False)


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError("pairs id order mismatch sample")

    anchor_path = first_existing(ANCHOR_PATHS)
    if anchor_path is None:
        raise FileNotFoundError("anchor missing")
    anchor = load_pred(anchor_path, sample)

    baselines = {"anchor_v33_qprob2000": anchor.copy()}

    for name, paths in DIRECT_BASES.items():
        p = first_existing(paths)
        if p is not None:
            baselines[name] = load_pred(p, sample)
            print("baseline:", name, p, "diff", int((baselines[name] != anchor).sum()))

    for short_name, variant in SAVED_VARIANTS.items():
        p = find_saved_variant(variant)
        if p is not None:
            baselines[short_name] = load_pred(p, sample)
            print("saved baseline:", short_name, p, "diff", int((baselines[short_name] != anchor).sum()))

    v42 = load_or_build_v42(sample, pairs)
    df = build_feature_frame(sample, pairs, v42, baselines)

    variants = dict(baselines)
    meta_rows = []
    chosen = {}

    # Same-quota graph outputs: genuine new graph model variants.
    for sc in ["v42_graph_only_score", "v42_graph_meta_score", "v42_graph_safe_score"]:
        pred = build_same_quota(df, sc, anchor)
        name = f"{sc}_same_quota"
        variants[name] = pred
        meta_rows.append({
            "variant": name,
            "source": "query_graph_same_quota",
            "base": "anchor",
            "score_col": sc,
            "mode": "",
            "cap": "",
            "used_swaps": int((pred != anchor).sum() // 2),
            "accepted_pool": "",
            "graph_support_mean": np.nan,
            "graph_add_nonzero_rate": np.nan,
            "meta_gain_mean": np.nan,
        })
        print("same quota:", name, "diff", int((pred != anchor).sum()))

    bases = [
        "v41_meta_same_quota",
        "raw_v35_b5000",
        "v40_history_impact2200",
        "v41_raw35_balanced",
        "v41_raw35_strict",
        "v38_precision_loose_2800",
        "v39_raw35_history_lex_veto_bad",
        "v36p2_balanced",
        "v36p2_strict",
    ]
    bases = [b for b in bases if b in baselines]

    caps = {
        "ultra": [400, 800, 1200],
        "strict": [800, 1400, 2200],
        "balanced": [1400, 2400, 3600],
        "impact": [2400, 4200, 7000],
    }

    for base in bases:
        for sc in ["v42_graph_meta_score", "v42_graph_safe_score", "v42_graph_only_score"]:
            sw = pair_swaps(df, anchor, baselines[base], sc)
            print("swap pool:", base, sc, len(sw))

            for mode, cap_list in caps.items():
                mask = graph_mask(sw, mode)
                acc = sw[mask].copy()

                if len(acc) == 0:
                    continue

                for cap in cap_list:
                    pred, take = apply_swaps(sample, anchor, acc, min(cap, len(acc)))
                    name = f"v42_{base}_{sc}_{mode}_cap{cap}"
                    variants[name] = pred
                    chosen[name] = take

                    meta_rows.append({
                        "variant": name,
                        "source": "query_graph_swap_filter",
                        "base": base,
                        "score_col": sc,
                        "mode": mode,
                        "cap": cap,
                        "accepted_pool": int(len(acc)),
                        "used_swaps": int(len(take)),
                        "graph_support_mean": float(take["graph_support_score"].mean()) if len(take) else np.nan,
                        "graph_support_min": float(take["graph_support_score"].min()) if len(take) else np.nan,
                        "graph_add_nonzero_rate": float((take["v42_graph_score_add"] > 0).mean()) if len(take) else np.nan,
                        "graph_drop_nonzero_rate": float((take["v42_graph_score_drop"] > 0).mean()) if len(take) else np.nan,
                        "meta_gain_mean": float(take["meta_gain"].mean()) if len(take) else np.nan,
                        "max_sim_add_mean": float(take["v42_graph_max_sim_add"].mean()) if len(take) else np.nan,
                    })
                    print("candidate:", name, "used", len(take), "diff", int((pred != anchor).sum()))

    print("evaluating...")
    eval_df = evaluate_variants(variants, sample)
    eval_df.to_csv(OUT_EVAL, index=False)
    wdf = weighted_score(eval_df)

    summary = pd.DataFrame(meta_rows)

    for name, pred in baselines.items():
        summary = pd.concat([summary, pd.DataFrame([{
            "variant": name,
            "source": "baseline",
            "base": "",
            "score_col": "",
            "mode": "",
            "cap": "",
            "used_swaps": "",
            "accepted_pool": "",
            "graph_support_mean": np.nan,
            "graph_support_min": np.nan,
            "graph_add_nonzero_rate": np.nan,
            "graph_drop_nonzero_rate": np.nan,
            "meta_gain_mean": np.nan,
            "max_sim_add_mean": np.nan,
        }])], ignore_index=True)

    aux = []
    for name, pred in variants.items():
        aux.append({
            "variant": name,
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_anchor": int((pred != anchor).sum()),
            "diff_vs_raw_v35": int((pred != baselines["raw_v35_b5000"]).sum()) if "raw_v35_b5000" in baselines else -1,
            "diff_vs_v41_same": int((pred != baselines["v41_meta_same_quota"]).sum()) if "v41_meta_same_quota" in baselines else -1,
            "diff_vs_v40_history": int((pred != baselines["v40_history_impact2200"]).sum()) if "v40_history_impact2200" in baselines else -1,
            "diff_vs_v38_precision": int((pred != baselines["v38_precision_loose_2800"]).sum()) if "v38_precision_loose_2800" in baselines else -1,
        })

    summary = summary.merge(pd.DataFrame(aux), on="variant", how="right")
    if len(wdf):
        summary = summary.merge(wdf, on="variant", how="left")

    diff = summary["diff_vs_anchor"].fillna(0).astype(float)
    move_bonus = np.minimum(0.020, np.log1p(diff) / np.log1p(50000) * 0.020)
    graph_bonus = np.minimum(0.006, summary["graph_support_mean"].fillna(0).astype(float) * 0.010)
    too_big_penalty = np.maximum(0, diff - 95000) / 600000.0

    summary["v42_decision_score"] = summary["weighted_macro"].fillna(0) + move_bonus + graph_bonus - too_big_penalty
    summary = summary.sort_values(["v42_decision_score", "weighted_macro"], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    build_review(chosen)

    save_names = []
    top = summary[
        (summary["source"].isin(["query_graph_same_quota", "query_graph_swap_filter"]))
        & (summary["diff_vs_anchor"] >= 1500)
        & (summary["diff_vs_anchor"] <= 120000)
    ].head(45)

    for nm in top["variant"].tolist():
        if nm not in save_names:
            save_names.append(nm)

    for nm in [
        "v42_graph_meta_score_same_quota",
        "v42_graph_safe_score_same_quota",
        "v42_graph_only_score_same_quota",
        "v41_meta_same_quota",
        "raw_v35_b5000",
        "v40_history_impact2200",
        "v38_precision_loose_2800",
        "anchor_v33_qprob2000",
    ]:
        if nm in variants and nm not in save_names:
            save_names.append(nm)

    saved = []
    for i, nm in enumerate(save_names[:55], 1):
        pred = variants[nm]
        out = SUB_DIR / f"FINAL_CANDIDATE_v42_qgraph_{i:03d}_{short_hash(nm)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(out, index=False)

        row = summary[summary["variant"] == nm].iloc[0].to_dict()
        row["file"] = str(out)
        saved.append(row)
        print("saved:", out, "<-", nm)

    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)

    print("\nTOP V42")
    cols = [
        "variant", "v42_decision_score", "weighted_macro", "source", "base", "score_col", "mode", "cap",
        "used_swaps", "accepted_pool", "graph_support_mean", "graph_add_nonzero_rate",
        "meta_gain_mean", "max_sim_add_mean",
        "diff_vs_anchor", "diff_vs_raw_v35", "diff_vs_v41_same",
        "mean_precision", "mean_recall",
    ]
    print(summary[[c for c in cols if c in summary.columns]].head(90).to_string(index=False))

    print("\nSAVED")
    sdf = pd.DataFrame(saved)
    if len(sdf):
        print(sdf[["variant", "file", "v42_decision_score", "weighted_macro", "diff_vs_anchor"]].to_string(index=False))

    print("\noutputs:")
    print(OUT_SCORE)
    print(OUT_INFO)
    print(OUT_SUMMARY)
    print(OUT_EVAL)
    print(OUT_SAVED)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
