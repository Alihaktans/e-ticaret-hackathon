from pathlib import Path
import os
import gc
import time
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

TRAIN_FEAT = ROOT / "data/processed/v21_train_features.parquet"
TEST_FEAT = ROOT / "data/processed/v21_test_features.parquet"
FEATURE_LIST = ROOT / "data/processed/v21_feature_columns.txt"

MODEL_DIR = ROOT / "models"
MODEL_VALID = MODEL_DIR / "v21_catboost_valid_fold4.cbm"
MODEL_FINAL = MODEL_DIR / "v21_catboost_final.cbm"

OUT_TEST = ROOT / "data/processed/v21_catboost_test_scores.parquet"
OUT_VALID = ROOT / "data/processed/v21_catboost_valid_scores.parquet"

REPORT_VALID = ROOT / "reports/manual_review/v21_catboost_valid_report.csv"
REPORT_SCORE = ROOT / "reports/manual_review/v21_catboost_test_score_report.csv"
REPORT_IMPORTANCE = ROOT / "reports/manual_review/v21_catboost_feature_importance.csv"

SEED = 2026
VALID_FOLD = 4

# 0 = full train. İstersen PowerShell'de set edebilirsin:
# $env:V21_MAX_TRAIN_ROWS="1800000"
MAX_TRAIN_ROWS = int(os.environ.get("V21_MAX_TRAIN_ROWS", "0"))

# GPU patlarsa:
# $env:V21_USE_GPU="0"
USE_GPU = os.environ.get("V21_USE_GPU", "1") != "0"

# Daha hızlı deneme için:
# $env:V21_ITERATIONS="800"
ITERATIONS = int(os.environ.get("V21_ITERATIONS", "1200"))

CHUNK_PRED = 500000

def sigmoid_clip(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro"),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(np.mean(p)),
        "true_pos_ratio": float(np.mean(y)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def make_weights(df):
    y = df["label"].astype(int).to_numpy()
    w = np.ones(len(df), dtype=np.float32)

    # Pozitifleri yükseltiyoruz çünkü train candidate set pos_ratio ~0.099.
    w[y == 1] *= 4.0

    if "candidate_source" in df.columns:
        src = df["candidate_source"].astype(str).to_numpy()

        w[src == "lex_top"] *= 1.15
        w[src == "same_root"] *= 1.00
        w[src == "random_easy"] *= 0.35
        w[src == "positive"] *= 1.00

    return w

def prepare_X(df, feature_cols):
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    for c in feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(-1.0).astype("float32")
    return x

def build_params(iterations):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": iterations,
        "learning_rate": 0.045,
        "depth": 8,
        "l2_leaf_reg": 9.0,
        "random_strength": 0.8,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.82,
        "random_seed": SEED,
        "od_type": "Iter",
        "od_wait": 90,
        "verbose": 100,
        "allow_writing_files": False,
        "thread_count": max(1, os.cpu_count() - 1),
    }

    if USE_GPU:
        params.update({
            "task_type": "GPU",
            "devices": "0",
            "gpu_ram_part": 0.88,
        })
    else:
        params.update({
            "task_type": "CPU",
        })

    return params

def train_model(train_pool, valid_pool, iterations, model_path):
    params = build_params(iterations)
    print("CatBoost params:", params)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(str(model_path))
    return model

def predict_proba_chunks(model, df, feature_cols):
    scores = np.zeros(len(df), dtype=np.float32)

    for start in range(0, len(df), CHUNK_PRED):
        end = min(start + CHUNK_PRED, len(df))
        print(f"predict {start:,}-{end:,}/{len(df):,}")

        x = prepare_X(df.iloc[start:end], feature_cols)
        pool = Pool(x)
        scores[start:end] = model.predict_proba(pool)[:, 1].astype(np.float32)

        del x, pool
        gc.collect()

    return scores

def main():
    t0 = time.time()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_VALID.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEST.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = FEATURE_LIST.read_text(encoding="utf-8").splitlines()
    feature_cols = [c.strip() for c in feature_cols if c.strip()]

    print("feature cols:", len(feature_cols))
    print("loading train features...")
    train = pd.read_parquet(TRAIN_FEAT)

    print("train shape:", train.shape)
    print("label counts:")
    print(train["label"].value_counts().to_string())

    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise RuntimeError(f"Missing train feature columns: {missing[:20]} ... total={len(missing)}")

    if MAX_TRAIN_ROWS and MAX_TRAIN_ROWS < len(train):
        print("sampling train rows:", MAX_TRAIN_ROWS)

        pos = train[train["label"].astype(int) == 1]
        neg = train[train["label"].astype(int) == 0]

        target_pos = min(len(pos), max(200000, int(MAX_TRAIN_ROWS * 0.18)))
        target_neg = min(len(neg), MAX_TRAIN_ROWS - target_pos)

        pos_s = pos.sample(n=target_pos, random_state=SEED)
        neg_s = neg.sample(n=target_neg, random_state=SEED)

        train = pd.concat([pos_s, neg_s], ignore_index=True)
        train = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        print("sampled train shape:", train.shape)
        print(train["label"].value_counts().to_string())

    print("splitting fold...")
    train["fold"] = train["fold"].astype(int)

    tr = train[train["fold"] != VALID_FOLD].copy()
    va = train[train["fold"] == VALID_FOLD].copy()

    print("tr:", tr.shape, "va:", va.shape)
    print("valid label counts:")
    print(va["label"].value_counts().to_string())

    y_tr = tr["label"].astype(int).to_numpy()
    y_va = va["label"].astype(int).to_numpy()

    w_tr = make_weights(tr)

    print("preparing X train/valid...")
    X_tr = prepare_X(tr, feature_cols)
    X_va = prepare_X(va, feature_cols)

    train_pool = Pool(X_tr, label=y_tr, weight=w_tr)
    valid_pool = Pool(X_va, label=y_va)

    print("training validation model...")
    try:
        model = train_model(train_pool, valid_pool, ITERATIONS, MODEL_VALID)
    except Exception as e:
        if USE_GPU:
            print("GPU training failed. Retrying on CPU. Error:", repr(e))
            os.environ["V21_USE_GPU"] = "0"
            globals()["USE_GPU"] = False
            model = train_model(train_pool, valid_pool, min(900, ITERATIONS), MODEL_VALID)
        else:
            raise

    print("predicting validation...")
    va_score = model.predict_proba(valid_pool)[:, 1].astype(np.float32)

    try:
        auc = roc_auc_score(y_va, va_score)
    except Exception:
        auc = np.nan

    rows = []
    for th in np.arange(0.02, 0.981, 0.02):
        pred = (va_score >= th).astype(np.int8)
        row = {
            "threshold": float(th),
            "valid_auc": float(auc),
            "valid_rows": len(va),
            "train_rows": len(tr),
            "best_iteration": int(model.get_best_iteration() or ITERATIONS),
        }
        row.update(metrics(y_va, pred))
        rows.append(row)

    valid_report = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    valid_report.to_csv(REPORT_VALID, index=False)

    print("\nVALID TOP")
    print(valid_report.head(25).to_string(index=False))

    valid_out = va[["term_id", "item_id", "label", "fold", "candidate_source"]].copy()
    valid_out["v21_score"] = va_score
    valid_out.to_parquet(OUT_VALID, index=False)

    print("feature importance...")
    imp = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(REPORT_IMPORTANCE, index=False)
    print(imp_df.head(40).to_string(index=False))

    best_iter = int(model.get_best_iteration() or ITERATIONS)
    final_iter = int(min(max(best_iter * 1.15, 600), ITERATIONS))
    print("best_iter:", best_iter, "final_iter:", final_iter)

    del train_pool, valid_pool, X_tr, X_va, tr, va, y_tr, y_va, w_tr, model
    gc.collect()

    print("\ntraining FINAL model on all train rows...")
    y_all = train["label"].astype(int).to_numpy()
    w_all = make_weights(train)
    X_all = prepare_X(train, feature_cols)

    final_pool = Pool(X_all, label=y_all, weight=w_all)

    params_final = build_params(final_iter)
    params_final.pop("od_type", None)
    params_final.pop("od_wait", None)

    print("final params:", params_final)

    final_model = CatBoostClassifier(**params_final)
    final_model.fit(final_pool)
    final_model.save_model(str(MODEL_FINAL))

    del final_pool, X_all, y_all, w_all, train
    gc.collect()

    print("\nloading test features...")
    test = pd.read_parquet(TEST_FEAT)
    print("test shape:", test.shape)

    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise RuntimeError(f"Missing test feature columns: {missing[:20]} ... total={len(missing)}")

    print("predicting test...")
    test_score = predict_proba_chunks(final_model, test, feature_cols)

    out = test[["id", "term_id", "item_id"]].copy()
    out["v21_score"] = test_score
    out.to_parquet(OUT_TEST, index=False)

    score_report = pd.DataFrame([{
        "rows": len(out),
        "score_mean": float(np.mean(test_score)),
        "score_std": float(np.std(test_score)),
        "score_min": float(np.min(test_score)),
        "score_p01": float(np.quantile(test_score, 0.01)),
        "score_p05": float(np.quantile(test_score, 0.05)),
        "score_p10": float(np.quantile(test_score, 0.10)),
        "score_p25": float(np.quantile(test_score, 0.25)),
        "score_p50": float(np.quantile(test_score, 0.50)),
        "score_p75": float(np.quantile(test_score, 0.75)),
        "score_p90": float(np.quantile(test_score, 0.90)),
        "score_p95": float(np.quantile(test_score, 0.95)),
        "score_p99": float(np.quantile(test_score, 0.99)),
        "score_max": float(np.max(test_score)),
        "runtime_min": round((time.time() - t0) / 60, 3),
    }])

    score_report.to_csv(REPORT_SCORE, index=False)

    print("\nTEST SCORE REPORT")
    print(score_report.to_string(index=False))

    print("saved:", MODEL_VALID)
    print("saved:", MODEL_FINAL)
    print("saved:", OUT_VALID)
    print("saved:", OUT_TEST)
    print("saved:", REPORT_VALID)
    print("saved:", REPORT_SCORE)
    print("saved:", REPORT_IMPORTANCE)

if __name__ == "__main__":
    main()
