from pathlib import Path
import os
import re
import gc
import time
import traceback
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

from catboost import CatBoostClassifier, Pool


ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

TRAIN_CAND = ROOT / "data/processed/v21_train_candidates.parquet"
TRAIN_FEAT = ROOT / "data/processed/v21_train_features.parquet"
TEST_FEAT = ROOT / "data/processed/v21_test_features.parquet"
FEATURE_COLS_FILE = ROOT / "data/processed/v21_feature_columns.txt"

V5_SCORE = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
V18_SCORE = ROOT / "data/processed/v18_minilm_candidate_scores.parquet"
V21_SCORE = ROOT / "data/processed/v21_catboost_test_scores.parquet"

SPARSE_TEST_EXISTING = ROOT / "data/processed/v26_sparse_text_scores.parquet"
SPARSE_TRAIN_OUT = ROOT / "data/processed/v27_train_sparse_scores.parquet"
SPARSE_TEST_OUT = ROOT / "data/processed/v27_test_sparse_scores.parquet"

MODEL_VALID = ROOT / "models/v27_sparse_aug_catboost_valid_fold4.cbm"
MODEL_FINAL = ROOT / "models/v27_sparse_aug_catboost_final.cbm"

OUT_SCORE = ROOT / "data/processed/v27_sparse_aug_catboost_test_scores.parquet"
OUT_VALID_REPORT = ROOT / "reports/manual_review/v27_sparse_aug_valid_report.csv"
OUT_TEST_REPORT = ROOT / "reports/manual_review/v27_sparse_aug_test_score_report.csv"

OUT_EVAL = ROOT / "reports/manual_review/v27_sparse_aug_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v27_sparse_aug_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v27_sparse_aug_full_summary.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
    "v20_active": ROOT / "reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv",
    "v21_active": ROOT / "reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv",
}

PATHS = {
    "v13": [
        ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v13_banded_ce_aggressive_w025_t0275_GSB.csv",
    ],
    "v16": [
        ROOT / "submissions/FINAL_MAIN_v16_remove_all_add_blend_ge_0p3.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v16_v16_remove_all_add_blend_ge_0p3.csv",
    ],
    "v19p1": [
        ROOT / "submissions/FINAL_MAIN_v19p1_honest_addonly_ba012_v18th0p9.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v19p1_v19p1_honest_addonly_b04_ba012_r1_v18th0p9.csv",
    ],
    "v20": [
        ROOT / "submissions/FINAL_MAIN_v20_safe_rem_t0p01_v5lt0p8.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v20_v20_rem_t0p01_v5lt0p8.csv",
    ],
    "add5": [
        ROOT / "submissions/FINAL_MAIN_v21_v20_add5000.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v21_v21_v20_add5000.csv",
    ],
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
    "v26_sparse_supported2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_supported_budget2000.csv",
    ],
    "v26_sparse_strict2000": [
        ROOT / "submissions/FINAL_CANDIDATE_v26_sparse_addonly_strictpct3_budget2000.csv",
    ],
}


def now():
    return time.strftime("%H:%M:%S")


def log(*args):
    print(f"[{now()}]", *args, flush=True)


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


def load_sub(name, sample, required=True):
    p = first_existing(PATHS[name])
    if p is None:
        if required:
            raise FileNotFoundError(f"missing submission: {name}")
        log("missing optional submission:", name)
        return None

    log("loading submission", name, p)
    d = pd.read_csv(p)
    d["id"] = d["id"].astype(str)

    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {name} {p}")

    return d["prediction"].astype(np.int8).to_numpy()


def load_csv_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def compute_pair_dot(term_mat, item_mat, term_idx, item_idx, chunk=350_000):
    out = np.zeros(len(term_idx), dtype=np.float32)
    for s in range(0, len(term_idx), chunk):
        e = min(s + chunk, len(term_idx))
        A = term_mat[term_idx[s:e]]
        B = item_mat[item_idx[s:e]]
        out[s:e] = np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)
        log("dot", e, "/", len(term_idx))
    return out


def build_sparse_for_pairs(pair_df, out_path, label):
    log("building sparse features for", label)

    pair_df = pair_df.copy()
    pair_df["term_id"] = pair_df["term_id"].astype(str)
    pair_df["item_id"] = pair_df["item_id"].astype(str)

    log("loading terms/items for sparse")
    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)
    terms["query_clean"] = terms["query"].map(clean_text)

    items = pd.read_csv(ITEMS, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)

    for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if c not in items.columns:
            items[c] = ""

    items["title_clean"] = items["title"].map(clean_text)
    items["category_clean"] = items["category"].map(clean_text)
    items["brand_clean"] = items["brand"].map(clean_text)
    items["gender_clean"] = items["gender"].map(clean_text)
    items["age_clean"] = items["age_group"].map(clean_text)
    items["attr_clean"] = items["attributes"].map(clean_text)

    items["item_text"] = (
        items["title_clean"] + " " + items["title_clean"] + " "
        + items["brand_clean"] + " "
        + items["category_clean"] + " "
        + items["gender_clean"] + " "
        + items["age_clean"] + " "
        + items["attr_clean"].str.slice(0, 350)
    ).str.strip()

    term_pos = pd.Series(np.arange(len(terms), dtype=np.int32), index=terms["term_id"])
    item_pos = pd.Series(np.arange(len(items), dtype=np.int32), index=items["item_id"])

    term_idx = pair_df["term_id"].map(term_pos).astype(np.int32).to_numpy()
    item_idx = pair_df["item_id"].map(item_pos).astype(np.int32).to_numpy()

    log("word vectorizing")
    word_vec = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        norm="l2",
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
    )

    Tw = word_vec.transform(terms["query_clean"].fillna("").tolist())
    Iw = word_vec.transform(items["item_text"].fillna("").tolist())
    word_score = compute_pair_dot(Tw, Iw, term_idx, item_idx)

    del Tw, Iw, word_vec
    gc.collect()

    log("char vectorizing")
    char_vec = HashingVectorizer(
        n_features=2**21,
        alternate_sign=False,
        norm="l2",
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
    )

    Tc = char_vec.transform(terms["query_clean"].fillna("").tolist())
    Ic = char_vec.transform(items["item_text"].fillna("").tolist())
    char_score = compute_pair_dot(Tc, Ic, term_idx, item_idx)

    del Tc, Ic, char_vec, terms, items
    gc.collect()

    out = pd.DataFrame({
        "term_id": pair_df["term_id"].to_numpy(),
        "item_id": pair_df["item_id"].to_numpy(),
        "sparse_word": word_score,
        "sparse_char": char_score,
    })

    if "id" in pair_df.columns:
        out.insert(0, "id", pair_df["id"].astype(str).to_numpy())

    out["sparse_blend"] = (
        0.55 * out["sparse_word"].astype("float32")
        + 0.45 * out["sparse_char"].astype("float32")
    ).astype("float32")

    out["sparse_candidate_count"] = out.groupby("term_id")["item_id"].transform("count").astype(np.int32)
    out["sparse_rank"] = out.groupby("term_id")["sparse_blend"].rank(method="first", ascending=False).astype(np.int32)
    out["sparse_pct_rank"] = (out["sparse_rank"] / out["sparse_candidate_count"]).astype("float32")
    out["sparse_term_max"] = out.groupby("term_id")["sparse_blend"].transform("max").astype("float32")
    out["sparse_delta_from_max"] = (out["sparse_term_max"] - out["sparse_blend"]).astype("float32")

    out.to_parquet(out_path, index=False)
    log("saved sparse:", out_path, out.shape)
    return out


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


def add_variant(variants, name, pred):
    variants[name] = pred.astype(np.int8)


def main():
    ROOT.joinpath("models").mkdir(exist_ok=True)
    ROOT.joinpath("reports/manual_review").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("submissions").mkdir(exist_ok=True)

    log("loading sample")
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    ids = sample["id"].to_numpy()
    n_test = len(sample)
    id_to_idx = pd.Series(np.arange(n_test), index=sample["id"])

    # ------------------------------------------------------------
    # Sparse train/test features
    # ------------------------------------------------------------
    if SPARSE_TRAIN_OUT.exists() and os.getenv("V27_FORCE_SPARSE", "0") != "1":
        log("loading cached train sparse", SPARSE_TRAIN_OUT)
        train_sparse = pd.read_parquet(SPARSE_TRAIN_OUT)
    else:
        log("loading train candidates")
        train_pairs = pd.read_parquet(TRAIN_CAND, columns=["term_id", "item_id"])
        train_sparse = build_sparse_for_pairs(train_pairs, SPARSE_TRAIN_OUT, "train")
        del train_pairs
        gc.collect()

    if SPARSE_TEST_OUT.exists() and os.getenv("V27_FORCE_SPARSE", "0") != "1":
        log("loading cached test sparse", SPARSE_TEST_OUT)
        test_sparse = pd.read_parquet(SPARSE_TEST_OUT)
    elif SPARSE_TEST_EXISTING.exists():
        log("using existing v26 test sparse", SPARSE_TEST_EXISTING)
        test_sparse = pd.read_parquet(SPARSE_TEST_EXISTING)
        test_sparse.to_parquet(SPARSE_TEST_OUT, index=False)
    else:
        log("building test sparse from submission pairs")
        test_pairs = pd.read_csv(PAIRS)
        test_pairs["id"] = test_pairs["id"].astype(str)
        if not test_pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            raise RuntimeError("submission_pairs id order mismatch sample")
        test_sparse = build_sparse_for_pairs(test_pairs[["id", "term_id", "item_id"]], SPARSE_TEST_OUT, "test")

    sparse_cols = [
        "sparse_word",
        "sparse_char",
        "sparse_blend",
        "sparse_rank",
        "sparse_pct_rank",
        "sparse_delta_from_max",
        "sparse_term_max",
    ]

    for c in sparse_cols:
        if c not in train_sparse.columns:
            raise RuntimeError(f"missing train sparse col {c}")
        if c not in test_sparse.columns:
            raise RuntimeError(f"missing test sparse col {c}")

    # ------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------
    log("loading v21 feature tables")
    train = pd.read_parquet(TRAIN_FEAT)
    test = pd.read_parquet(TEST_FEAT)

    log("train shape", train.shape, "test shape", test.shape)

    if len(train) != len(train_sparse):
        raise RuntimeError(f"train sparse length mismatch {len(train)} vs {len(train_sparse)}")
    if len(test) != len(test_sparse):
        raise RuntimeError(f"test sparse length mismatch {len(test)} vs {len(test_sparse)}")

    if "id" in test.columns and "id" in test_sparse.columns:
        t1 = test["id"].astype(str).reset_index(drop=True)
        t2 = test_sparse["id"].astype(str).reset_index(drop=True)
        if not t1.equals(t2):
            raise RuntimeError("test sparse id order mismatch")

    with open(FEATURE_COLS_FILE, "r", encoding="utf-8") as f:
        feature_cols = [x.strip() for x in f if x.strip()]

    for c in sparse_cols:
        train[c] = train_sparse[c].astype("float32").to_numpy()
        test[c] = test_sparse[c].astype("float32").to_numpy()
        if c not in feature_cols:
            feature_cols.append(c)

    del train_sparse
    gc.collect()

    # Extra sparse interactions
    train["sparse_blend_x_title_cov"] = (
        train["sparse_blend"].astype("float32")
        * pd.to_numeric(train.get("title_cov", 0), errors="coerce").fillna(0).astype("float32")
    )
    test["sparse_blend_x_title_cov"] = (
        test["sparse_blend"].astype("float32")
        * pd.to_numeric(test.get("title_cov", 0), errors="coerce").fillna(0).astype("float32")
    )
    feature_cols.append("sparse_blend_x_title_cov")

    if "candidate_count" in train.columns:
        train["sparse_pct_x_candidate_count_log"] = (
            train["sparse_pct_rank"].astype("float32")
            * np.log1p(pd.to_numeric(train["candidate_count"], errors="coerce").fillna(0).astype("float32"))
        ).astype("float32")
        test["sparse_pct_x_candidate_count_log"] = (
            test["sparse_pct_rank"].astype("float32")
            * np.log1p(pd.to_numeric(test["candidate_count"], errors="coerce").fillna(0).astype("float32"))
        ).astype("float32")
        feature_cols.append("sparse_pct_x_candidate_count_log")

    feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]
    log("feature count", len(feature_cols))

    # Cast numeric compactly
    log("casting features")
    for c in feature_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(-1).astype("float32")
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(-1).astype("float32")

    # Metadata
    if "label" in train.columns:
        y = train["label"].astype(np.int8).to_numpy()
    else:
        cand = pd.read_parquet(TRAIN_CAND, columns=["label"])
        y = cand["label"].astype(np.int8).to_numpy()

    if "fold" in train.columns:
        fold = train["fold"].astype(np.int8).to_numpy()
    else:
        cand = pd.read_parquet(TRAIN_CAND, columns=["fold"])
        fold = cand["fold"].astype(np.int8).to_numpy()

    if "source" in train.columns:
        source = train["source"].astype(str).to_numpy()
    else:
        try:
            cand = pd.read_parquet(TRAIN_CAND, columns=["source"])
            source = cand["source"].astype(str).to_numpy()
        except Exception:
            source = np.array(["unknown"] * len(train), dtype=object)

    weights = np.ones(len(train), dtype=np.float32)
    weights[y == 1] *= float(os.getenv("V27_POS_WEIGHT", "4.2"))
    weights[source == "lex_top"] *= 1.15
    weights[source == "same_root"] *= 1.00
    weights[source == "random_easy"] *= 0.35

    valid_mask = fold == 4
    train_mask = ~valid_mask

    log("train rows", int(train_mask.sum()), "valid rows", int(valid_mask.sum()), "pos", int(y.sum()))

    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(os.getenv("V27_ITERATIONS", "1800")),
        learning_rate=float(os.getenv("V27_LR", "0.035")),
        depth=int(os.getenv("V27_DEPTH", "8")),
        l2_leaf_reg=float(os.getenv("V27_L2", "10.0")),
        random_strength=float(os.getenv("V27_RANDOM_STRENGTH", "0.8")),
        bootstrap_type="Bernoulli",
        subsample=float(os.getenv("V27_SUBSAMPLE", "0.82")),
        random_seed=2027,
        od_type="Iter",
        od_wait=int(os.getenv("V27_OD_WAIT", "140")),
        verbose=100,
        allow_writing_files=True,
    )

    if os.getenv("V27_USE_GPU", "1") == "1":
        params.update(task_type="GPU", devices="0", gpu_ram_part=float(os.getenv("V27_GPU_RAM_PART", "0.86")))
    else:
        params.update(task_type="CPU", thread_count=max(1, os.cpu_count() or 8))

    # Optional downsample
    max_train_rows = int(os.getenv("V27_MAX_TRAIN_ROWS", "0"))
    train_idx = np.where(train_mask)[0]
    valid_idx = np.where(valid_mask)[0]

    if max_train_rows > 0 and len(train_idx) > max_train_rows:
        rng = np.random.default_rng(2027)
        pos_idx = train_idx[y[train_idx] == 1]
        neg_idx = train_idx[y[train_idx] == 0]
        keep_pos = min(len(pos_idx), max_train_rows // 4)
        keep_neg = max_train_rows - keep_pos
        pos_take = rng.choice(pos_idx, size=keep_pos, replace=False)
        neg_take = rng.choice(neg_idx, size=keep_neg, replace=False)
        train_idx = np.concatenate([pos_take, neg_take])
        rng.shuffle(train_idx)
        log("downsampled train rows", len(train_idx))

    # ------------------------------------------------------------
    # Train validation model
    # ------------------------------------------------------------
    log("creating pools")
    train_pool = Pool(train.iloc[train_idx][feature_cols], label=y[train_idx], weight=weights[train_idx])
    valid_pool = Pool(train.iloc[valid_idx][feature_cols], label=y[valid_idx], weight=weights[valid_idx])

    log("training validation CatBoost")
    valid_model = CatBoostClassifier(**params)
    valid_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    valid_model.save_model(str(MODEL_VALID))

    valid_pred = valid_model.predict_proba(valid_pool)[:, 1].astype("float32")
    valid_auc = roc_auc_score(y[valid_idx], valid_pred)

    thresholds = np.linspace(0.02, 0.98, 49)
    rows = []
    for th in thresholds:
        p = (valid_pred >= th).astype(np.int8)
        m = metrics(y[valid_idx], p)
        m["threshold"] = float(th)
        m["valid_auc"] = float(valid_auc)
        rows.append(m)

    vr = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    vr.to_csv(OUT_VALID_REPORT, index=False)
    log("valid best")
    log(vr.head(10).to_string(index=False))

    best_iter = valid_model.get_best_iteration()
    if best_iter is None or best_iter <= 0:
        best_iter = int(params["iterations"])

    final_iters = int(os.getenv("V27_FINAL_ITERATIONS", str(min(max(best_iter + 120, 700), int(params["iterations"])))))
    log("best_iter", best_iter, "final_iters", final_iters)

    del train_pool, valid_pool, valid_model
    gc.collect()

    # ------------------------------------------------------------
    # Train final model
    # ------------------------------------------------------------
    final_params = dict(params)
    final_params["iterations"] = final_iters
    final_params.pop("od_type", None)
    final_params.pop("od_wait", None)

    log("training final model on all rows")
    final_pool = Pool(train[feature_cols], label=y, weight=weights)
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(final_pool)
    final_model.save_model(str(MODEL_FINAL))

    del final_pool, train
    gc.collect()

    # ------------------------------------------------------------
    # Predict test
    # ------------------------------------------------------------
    log("predicting test")
    test_scores = np.zeros(len(test), dtype=np.float32)
    chunk = int(os.getenv("V27_PRED_CHUNK", "350000"))

    for s in range(0, len(test), chunk):
        e = min(s + chunk, len(test))
        pool = Pool(test.iloc[s:e][feature_cols])
        test_scores[s:e] = final_model.predict_proba(pool)[:, 1].astype("float32")
        log("pred", e, "/", len(test))

    score_df = pd.DataFrame({
        "id": ids,
        "v27_score": test_scores,
    })

    # Keep sparse columns for later analysis
    for c in sparse_cols:
        score_df[c] = test_sparse[c].astype("float32").to_numpy()

    score_df.to_parquet(OUT_SCORE, index=False)

    desc = score_df[["v27_score", "sparse_blend", "sparse_pct_rank"]].describe().T
    desc.to_csv(OUT_TEST_REPORT)
    log("test score desc")
    log(desc.to_string())

    del test, final_model
    gc.collect()

    # ------------------------------------------------------------
    # Build candidates
    # ------------------------------------------------------------
    log("loading anchor submissions")
    pred_v13 = load_sub("v13", sample, required=False)
    pred_v16 = load_sub("v16", sample)
    pred_v19 = load_sub("v19p1", sample)
    pred_v20 = load_sub("v20", sample)
    pred_add5 = load_sub("add5", sample)
    pred_base = load_sub("v22_base", sample)
    pred_v24 = load_sub("v24_anchor", sample)
    pred_sparse_supported2000 = load_sub("v26_sparse_supported2000", sample, required=False)
    pred_sparse_strict2000 = load_sub("v26_sparse_strict2000", sample, required=False)

    if pred_v13 is None:
        pred_v13 = pred_base.copy()
    if pred_sparse_supported2000 is None:
        pred_sparse_supported2000 = pred_v24.copy()
    if pred_sparse_strict2000 is None:
        pred_sparse_strict2000 = pred_v24.copy()

    # Extra model scores for safe ranking
    log("loading external test scores")
    v5_score = np.zeros(n_test, dtype=np.float32)
    if V5_SCORE.exists():
        v5 = pd.read_parquet(V5_SCORE, columns=["id", "proba_avg"])
        v5["id"] = v5["id"].astype(str)
        if not v5["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            raise RuntimeError("v5 id mismatch")
        v5_score = v5["proba_avg"].astype("float32").to_numpy()

    v18_score = np.zeros(n_test, dtype=np.float32)
    if V18_SCORE.exists():
        v18 = pd.read_parquet(V18_SCORE)
        v18["id"] = v18["id"].astype(str)
        idx = v18["id"].map(id_to_idx)
        ok = idx.notna()
        v18_score[idx.loc[ok].astype(int).to_numpy()] = v18.loc[ok, "v18_minilm_sigmoid"].astype("float32").to_numpy()

    old_v21_score = np.zeros(n_test, dtype=np.float32)
    if V21_SCORE.exists():
        oldv21 = pd.read_parquet(V21_SCORE)
        oldv21["id"] = oldv21["id"].astype(str)
        if not oldv21["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            raise RuntimeError("old v21 id mismatch")
        old_v21_score = oldv21["v21_score"].astype("float32").to_numpy()

    v27 = test_scores
    sparse_blend = score_df["sparse_blend"].astype("float32").to_numpy()
    sparse_pct = score_df["sparse_pct_rank"].astype("float32").to_numpy()

    variants = {}
    add_variant(variants, "v22_base_public080", pred_base)
    add_variant(variants, "v24_anchor", pred_v24)
    add_variant(variants, "v26_sparse_supported2000", pred_sparse_supported2000)
    add_variant(variants, "v26_sparse_strict2000", pred_sparse_strict2000)

    add_score = (
        0.55 * v27
        + 0.18 * sparse_blend
        + 0.10 * (1.0 - sparse_pct)
        + 0.08 * v5_score
        + 0.06 * v18_score
        + 0.03 * old_v21_score
    ).astype("float32")

    trim_keep_score = (
        0.55 * v27
        + 0.15 * v5_score
        + 0.15 * v18_score
        + 0.10 * old_v21_score
        + 0.05 * sparse_blend
    ).astype("float32")

    add_mask_general = pred_v24 == 0
    add_mask_sparse_agree = (
        (pred_v24 == 0)
        & (
            ((sparse_pct <= 0.06) & (sparse_blend >= 0.20))
            | (pred_sparse_supported2000 == 1)
            | (pred_sparse_strict2000 == 1)
        )
    )
    add_mask_v27_high = (pred_v24 == 0) & (v27 >= np.quantile(v27[pred_v24 == 0], 0.995))

    trim_mask_general = pred_v24 == 1

    add_idx_general = np.where(add_mask_general)[0]
    add_order_general = add_idx_general[np.argsort(-add_score[add_idx_general])]

    add_idx_sparse = np.where(add_mask_sparse_agree)[0]
    add_order_sparse = add_idx_sparse[np.argsort(-add_score[add_idx_sparse])]

    add_idx_v27high = np.where(add_mask_v27_high)[0]
    add_order_v27high = add_idx_v27high[np.argsort(-add_score[add_idx_v27high])]

    trim_idx = np.where(trim_mask_general)[0]
    trim_order = trim_idx[np.argsort(trim_keep_score[trim_idx])]

    log("add general", len(add_order_general), "add sparse", len(add_order_sparse), "add v27high", len(add_order_v27high), "trim", len(trim_order))

    # Add-only variants
    for budget in [500, 1000, 2000, 3000, 5000, 8000, 12000]:
        pred = pred_v24.copy()
        pred[add_order_general[:min(budget, len(add_order_general))]] = 1
        add_variant(variants, f"v27_addonly_general_b{budget}", pred)

    for budget in [500, 1000, 2000, 3000, 5000, 8000]:
        pred = pred_v24.copy()
        pred[add_order_sparse[:min(budget, len(add_order_sparse))]] = 1
        add_variant(variants, f"v27_addonly_sparseagree_b{budget}", pred)

    for budget in [500, 1000, 2000, 3000]:
        pred = pred_v24.copy()
        pred[add_order_v27high[:min(budget, len(add_order_v27high))]] = 1
        add_variant(variants, f"v27_addonly_v27high_b{budget}", pred)

    # Trim-only
    for budget in [500, 1000, 2000, 3000, 5000, 8000]:
        pred = pred_v24.copy()
        pred[trim_order[:min(budget, len(trim_order))]] = 0
        add_variant(variants, f"v27_trim_low_b{budget}", pred)

    # Swap variants
    for budget in [500, 1000, 2000, 3000, 5000, 8000]:
        pred = pred_v24.copy()
        pred[add_order_general[:min(budget, len(add_order_general))]] = 1
        pred[trim_order[:min(budget, len(trim_order))]] = 0
        add_variant(variants, f"v27_swap_general_b{budget}", pred)

    for budget in [500, 1000, 2000, 3000, 5000]:
        pred = pred_v24.copy()
        pred[add_order_sparse[:min(budget, len(add_order_sparse))]] = 1
        pred[trim_order[:min(budget, len(trim_order))]] = 0
        add_variant(variants, f"v27_swap_sparseagree_b{budget}", pred)

    # Start from v26 sparse2000 then swap/trim
    for base_name, base_pred in [
        ("supported2000", pred_sparse_supported2000),
        ("strict2000", pred_sparse_strict2000),
    ]:
        add_variant(variants, f"v27_base_{base_name}", base_pred)

        for budget in [500, 1000, 2000, 3000]:
            pred = base_pred.copy()
            pred[trim_order[:min(budget, len(trim_order))]] = 0
            add_variant(variants, f"v27_{base_name}_trim_b{budget}", pred)

        for add_budget, trim_budget in [(1000, 500), (2000, 1000), (3000, 1500), (3000, 3000)]:
            pred = base_pred.copy()
            pred[add_order_general[:min(add_budget, len(add_order_general))]] = 1
            pred[trim_order[:min(trim_budget, len(trim_order))]] = 0
            add_variant(variants, f"v27_{base_name}_add{add_budget}_trim{trim_budget}", pred)

    # Pure top-ratio risky variants
    for ratio in [0.31684, 0.31714, 0.31744, 0.31773, 0.31833]:
        k = int(round(ratio * n_test))
        order = np.argsort(-v27)
        pred = np.zeros(n_test, dtype=np.int8)
        pred[order[:k]] = 1
        add_variant(variants, f"v27_pure_topratio_{ratio:.5f}", pred)

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
            "diff_vs_v20": int((pred != pred_v20).sum()),
            "diff_vs_supported2000": int((pred != pred_sparse_supported2000).sum()),
            "diff_vs_strict2000": int((pred != pred_sparse_strict2000).sum()),
        })

    full = pd.DataFrame(full_rows)
    full.to_csv(OUT_FULL, index=False)

    # ------------------------------------------------------------
    # Manual validation
    # ------------------------------------------------------------
    log("evaluating variants")
    eval_rows = []

    for label_set, path in LABEL_FILES.items():
        if not path.exists():
            log("missing label file", label_set)
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

        if label_set in ["v20_active", "v21_active"] and "review_bucket" in lab.columns:
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
            yy = part["assistant_label"].to_numpy()

            for name, pred in variants.items():
                pp = pred[idx]
                row = {
                    "label_set": label_set,
                    "subset": subset_name,
                    "eval_key": f"{label_set}_{subset_name}",
                    "variant": name,
                    "n": len(part),
                }
                row.update(metrics(yy, pp))
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
        "v20_active_clean": 0.05,
        "v20_active_high_medium_clean": 0.04,
        "v21_active_clean": 0.18,
        "v21_active_high_medium_clean": 0.14,
        "v21_active_bucket_v21_add5k_added_over_v20": 0.12,
        "v21_active_bucket_v21_vote_added_over_v20": 0.20,
        "v21_active_bucket_v21_vote_removed_from_v20": 0.20,
        "v21_active_bucket_v21_high_but_v18_low_risky_add": 0.10,
    }

    agg_rows = []
    v24_pos = float(pred_v24.mean())

    for name, g in eval_df.groupby("variant"):
        legacy_score = 0.0
        legacy_wsum = 0.0
        active_score = 0.0
        active_wsum = 0.0
        legacy_vals = []
        active_vals = []

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

        if legacy_wsum == 0:
            continue

        legacy_weighted = legacy_score / legacy_wsum
        active_weighted = active_score / active_wsum if active_wsum else np.nan
        combined_score = 0.72 * legacy_weighted + 0.28 * active_weighted if active_wsum else legacy_weighted

        f = full[full["variant"] == name].iloc[0]
        pos_ratio = float(f["pos_ratio"])
        diff_v24 = int(f["diff_vs_v24"])

        pos_penalty = abs(pos_ratio - v24_pos) * 1.40
        diff_penalty = max(0, diff_v24 - 8000) / 900000.0
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
            "legacy_min_macro": float(np.min(legacy_vals)) if legacy_vals else np.nan,
            "active_min_macro": float(np.min(active_vals)) if active_vals else np.nan,
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

    log("\nTOP PUBLIC SAFE")
    log(agg.head(80).to_string(index=False))

    log("\nTOP COMBINED")
    log(agg.sort_values("combined_score", ascending=False).head(80).to_string(index=False))

    log("\nFULL SUMMARY NEAR V24")
    near = full[full["diff_vs_v24"] <= 20000].sort_values("diff_vs_v24")
    log(near.head(120).to_string(index=False))

    # Save top candidates
    top_names = []
    for col in ["public_safe_score", "combined_score", "legacy_weighted", "active_weighted", "main_min_macro"]:
        for nm in agg.sort_values(col, ascending=False).head(12)["variant"].tolist():
            if nm not in top_names:
                top_names.append(nm)

    for nm in ["v24_anchor", "v26_sparse_supported2000", "v26_sparse_strict2000"]:
        if nm in variants and nm not in top_names:
            top_names.append(nm)

    for name in top_names[:40]:
        out = ROOT / "submissions" / f"FINAL_CANDIDATE_v27_{clean_name(name)}.csv"
        pd.DataFrame({"id": ids, "prediction": variants[name].astype(np.int8)}).to_csv(out, index=False)
        log("saved candidate", out)

    log("saved:", OUT_SCORE)
    log("saved:", OUT_VALID_REPORT)
    log("saved:", OUT_TEST_REPORT)
    log("saved:", OUT_EVAL)
    log("saved:", OUT_AGG)
    log("saved:", OUT_FULL)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
