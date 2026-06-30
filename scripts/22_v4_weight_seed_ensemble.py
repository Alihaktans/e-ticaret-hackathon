from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report

from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

SOURCE_SCRIPT = ROOT / "scripts" / "18_train_v4_weighted_semantic_ranker.py"

RUN_NAME = "embedding_ranker_v4_weight_seed_ensemble"

SEMANTIC_WEIGHTS = [0.05, 0.10, 0.20, 0.35, 0.50]
SEEDS = [42, 2026, 777]

VALID_SIZE = 0.20

ENSEMBLE_BLEND_WEIGHTS = [0.90, 0.85, 0.80, 0.75, 0.70]
TOP_RATIOS = [0.10, 0.12, 0.14]

REPORT_DIR = ROOT / "reports" / "experiments"
MODEL_DIR = ROOT / "models"
SUB_DIR = ROOT / "submissions"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

BASE_FINAL = SUB_DIR / "FINAL_first_submission_v4_weighted_semantic_e080_m020_top012.csv"


def load_v4_module():
    spec = importlib.util.spec_from_file_location("v4mod", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {SOURCE_SCRIPT}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict:
    best = {"threshold": 0.5, "macro_f1": -1.0}

    for th in np.linspace(0.01, 0.99, 99):
        pred = (proba >= th).astype(int)
        score = f1_score(y_true, pred, average="macro")

        if score > best["macro_f1"]:
            best = {"threshold": float(th), "macro_f1": float(score)}

    return best


def make_topratio_submission(
    ids: pd.Series,
    term_ids: pd.Series,
    final_score: np.ndarray,
    ratio: float,
    out_path: Path,
) -> dict:
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

    df = pd.DataFrame(
        {
            "id": ids.values,
            "term_id": term_ids.values,
            "final_score": final_score.astype(np.float32),
        }
    )

    pred = np.zeros(len(df), dtype=np.int8)
    score_series = df["final_score"]

    for _, idx in tqdm(df.groupby("term_id").groups.items(), desc=f"topratio {ratio}"):
        n = len(idx)

        k = round(n * ratio)
        k = max(1, k)
        k = min(20, k)
        k = min(k, n)

        top_idx = score_series.loc[idx].nlargest(k).index
        pred[top_idx] = 1

    tmp = df[["id"]].copy()
    tmp["prediction"] = pred

    sub = sample[["id"]].merge(tmp, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)

    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])
    assert set(sub["prediction"].unique()) <= {0, 1}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    diff_vs_base = None
    diff_ratio_vs_base = None

    if BASE_FINAL.exists():
        base = pd.read_csv(BASE_FINAL)
        diff_vs_base = int((sub["prediction"] != base["prediction"]).sum())
        diff_ratio_vs_base = float(diff_vs_base / len(sub))

    result = {
        "path": str(out_path),
        "ratio": ratio,
        "rows": int(len(sub)),
        "ones": int(sub["prediction"].sum()),
        "zeros": int(len(sub) - sub["prediction"].sum()),
        "pos_ratio": float(sub["prediction"].mean()),
        "diff_vs_base_final": diff_vs_base,
        "diff_ratio_vs_base_final": diff_ratio_vs_base,
    }

    print(f"\nSaved: {out_path}")
    print(result)

    return result


def train_one_model(
    mod,
    train: pd.DataFrame,
    test: pd.DataFrame,
    semantic_weight: float,
    seed: int,
    model_index: int,
) -> tuple[dict, np.ndarray]:
    feature_cols = mod.FEATURE_COLS
    cat_features = mod.CAT_FEATURES
    cat_indices = [feature_cols.index(c) for c in cat_features]

    df = train.copy()

    df["grid_sample_weight"] = 1.0
    df.loc[df["negative_type"] == "semantic_hard", "grid_sample_weight"] = semantic_weight
    df["grid_sample_weight"] = df["grid_sample_weight"].astype(np.float32)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALID_SIZE,
        random_state=seed,
    )

    tr_idx, va_idx = next(splitter.split(df, df["label"], groups=df["term_id"]))

    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[va_idx].reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"MODEL {model_index}: semantic_weight={semantic_weight}, seed={seed}")
    print("=" * 100)
    print(f"Train rows: {len(tr):,}")
    print(f"Valid rows: {len(va):,}")
    print("\nTrain negative type:")
    print(tr["negative_type"].value_counts(normalize=True))
    print("\nValid negative type:")
    print(va["negative_type"].value_counts(normalize=True))
    print("\nSample weights:")
    print(df.groupby("negative_type")["grid_sample_weight"].describe())

    train_pool = Pool(
        tr[feature_cols],
        label=tr["label"].astype(int),
        weight=tr["grid_sample_weight"].astype(float),
        cat_features=cat_indices,
    )

    valid_pool = Pool(
        va[feature_cols],
        label=va["label"].astype(int),
        weight=va["grid_sample_weight"].astype(float),
        cat_features=cat_indices,
    )

    params = {
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "iterations": 1300,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 7.0,
        "random_seed": seed,
        "auto_class_weights": "Balanced",
        "verbose": 100,
        "allow_writing_files": False,
    }

    try:
        model = CatBoostClassifier(**params, task_type="GPU", devices="0")
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        task_type = "GPU"
    except Exception as exc:
        print("\nGPU CatBoost failed, falling back to CPU.")
        print(f"Reason: {exc}")
        model = CatBoostClassifier(**params, task_type="CPU", thread_count=-1)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        task_type = "CPU"

    valid_proba = model.predict_proba(va[feature_cols])[:, 1]
    best = find_best_threshold(va["label"].astype(int).to_numpy(), valid_proba)

    valid_pred = (valid_proba >= best["threshold"]).astype(int)
    valid_macro_f1 = f1_score(va["label"].astype(int), valid_pred, average="macro")
    report_text = classification_report(va["label"].astype(int), valid_pred, digits=5)

    print("\nValidation report:")
    print(report_text)
    print(f"Best threshold: {best['threshold']:.5f}")
    print(f"Valid Macro-F1: {valid_macro_f1:.5f}")
    print(f"Task type     : {task_type}")

    model_name = f"{RUN_NAME}_semw{int(round(semantic_weight * 100)):03d}_seed{seed}"
    model_path = MODEL_DIR / f"{model_name}.cbm"
    model.save_model(model_path)

    print("\nPredicting test proba...")
    test_proba = model.predict_proba(test[feature_cols])[:, 1].astype(np.float32)

    meta = {
        "model_index": model_index,
        "semantic_weight": float(semantic_weight),
        "seed": int(seed),
        "task_type": task_type,
        "valid_macro_f1": float(valid_macro_f1),
        "best_threshold": float(best["threshold"]),
        "model_path": str(model_path),
        "classification_report": report_text,
        "best_iteration": int(model.get_best_iteration() or -1),
    }

    meta_path = MODEL_DIR / f"{model_name}_meta.joblib"
    joblib.dump(meta, meta_path)
    meta["meta_path"] = str(meta_path)

    return meta, test_proba


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("22 V4 WEIGHT + SEED ENSEMBLE")
    print("=" * 100)

    mod = load_v4_module()

    print("\nBuilding train frame from V4 source script...")
    train = mod.build_train_frame()

    print("\nBuilding test frame from V4 source script...")
    test = mod.build_test_frame()

    print(f"\nTrain rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    test_ids = test["id"].copy()
    test_term_ids = test["term_id"].copy()
    embedding_rank_score = (1.0 - test["rank_pct"].astype(np.float32)).to_numpy(dtype=np.float32)

    ensemble_sum = np.zeros(len(test), dtype=np.float32)
    ensemble_weighted_sum = np.zeros(len(test), dtype=np.float32)
    ensemble_weight_total = 0.0

    model_reports = []
    model_index = 0

    for semantic_weight in SEMANTIC_WEIGHTS:
        for seed in SEEDS:
            model_index += 1

            meta, test_proba = train_one_model(
                mod=mod,
                train=train,
                test=test,
                semantic_weight=semantic_weight,
                seed=seed,
                model_index=model_index,
            )

            model_reports.append(meta)

            ensemble_sum += test_proba

            # Validation skoru yüksek olan modele biraz daha fazla güven.
            w = max(float(meta["valid_macro_f1"]), 1e-6)
            ensemble_weighted_sum += test_proba * w
            ensemble_weight_total += w

            # Belleği rahatlat.
            del test_proba

            partial_path = REPORT_DIR / f"{RUN_NAME}_partial_report.json"
            partial_path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "model_reports": model_reports,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    ensemble_avg_proba = ensemble_sum / float(len(model_reports))
    ensemble_weighted_proba = ensemble_weighted_sum / float(ensemble_weight_total)

    out_proba_path = PROCESSED_DIR / f"{RUN_NAME}_test_ensemble_proba.parquet"
    pd.DataFrame(
        {
            "id": test_ids.values,
            "term_id": test_term_ids.values,
            "embedding_rank_score": embedding_rank_score,
            "ensemble_avg_proba": ensemble_avg_proba,
            "ensemble_weighted_proba": ensemble_weighted_proba,
        }
    ).to_parquet(out_proba_path, index=False)

    print(f"\nSaved ensemble proba: {out_proba_path}")

    submissions = []

    for proba_name, proba in [
        ("avg", ensemble_avg_proba),
        ("weighted", ensemble_weighted_proba),
    ]:
        for embed_weight in ENSEMBLE_BLEND_WEIGHTS:
            model_weight = 1.0 - embed_weight

            final_score = (
                embed_weight * embedding_rank_score
                + model_weight * proba.astype(np.float32)
            ).astype(np.float32)

            for ratio in TOP_RATIOS:
                ew_tag = f"e{int(round(embed_weight * 100)):03d}"
                mw_tag = f"m{int(round(model_weight * 100)):03d}"
                ratio_tag = f"top{int(round(ratio * 100)):03d}"

                out_name = f"{RUN_NAME}_{proba_name}_{ew_tag}_{mw_tag}_{ratio_tag}.csv"
                out_path = SUB_DIR / out_name

                submissions.append(
                    make_topratio_submission(
                        ids=test_ids,
                        term_ids=test_term_ids,
                        final_score=final_score,
                        ratio=ratio,
                        out_path=out_path,
                    )
                )

    model_report_df = pd.DataFrame(model_reports).sort_values("valid_macro_f1", ascending=False)
    sub_report_df = pd.DataFrame(submissions)

    model_summary_path = REPORT_DIR / f"{RUN_NAME}_model_summary.csv"
    submission_summary_path = REPORT_DIR / f"{RUN_NAME}_submission_summary.csv"

    model_report_df.to_csv(model_summary_path, index=False)
    sub_report_df.to_csv(submission_summary_path, index=False)

    final_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "semantic_weights": SEMANTIC_WEIGHTS,
        "seeds": SEEDS,
        "blend_weights": ENSEMBLE_BLEND_WEIGHTS,
        "top_ratios": TOP_RATIOS,
        "n_models": len(model_reports),
        "ensemble_proba_path": str(out_proba_path),
        "model_summary_path": str(model_summary_path),
        "submission_summary_path": str(submission_summary_path),
        "best_models": model_report_df.head(10).to_dict(orient="records"),
        "submissions": submissions,
    }

    report_path = REPORT_DIR / f"{RUN_NAME}_report.json"
    report_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Saved model summary     : {model_summary_path}")
    print(f"Saved submission summary: {submission_summary_path}")
    print(f"Saved report            : {report_path}")

    print("\nTop models:")
    print(model_report_df[["model_index", "semantic_weight", "seed", "valid_macro_f1", "best_threshold", "task_type"]].head(20).to_string(index=False))

    print("\nSubmission summary:")
    print(sub_report_df.to_string(index=False))


if __name__ == "__main__":
    main()
