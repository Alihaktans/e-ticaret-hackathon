from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

SOURCE_SCRIPT = ROOT / "scripts" / "18_train_v4_weighted_semantic_ranker.py"

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
SUB_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports" / "experiments"

RUN_NAME = "v4_full_semw035_threshold_ensemble"

SEMANTIC_WEIGHT = 0.35
SEEDS = [42, 2026, 777]

# CV en iyi alan: semw=0.35, th≈0.96
THRESHOLDS = [0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980]

# Ensemble model summary'de semw035 seed42 best_iter ≈143 idi.
# Full train için biraz pay bırakıyoruz ama aşırı overfit etmiyoruz.
ITERATIONS = 220


def load_v4_module():
    spec = importlib.util.spec_from_file_location("v4mod", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {SOURCE_SCRIPT}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def threshold_tag(th: float) -> str:
    return f"{th:.3f}".replace(".", "p")


def make_threshold_submission(
    sample: pd.DataFrame,
    ids: pd.Series,
    proba: np.ndarray,
    name: str,
    threshold: float,
) -> dict:
    tmp = pd.DataFrame(
        {
            "id": ids.values,
            "prediction": (proba >= threshold).astype(np.int8),
        }
    )

    sub = sample[["id"]].merge(tmp, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)

    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])
    assert set(sub["prediction"].unique()) <= {0, 1}

    tag = threshold_tag(threshold)
    out_path = SUB_DIR / f"{RUN_NAME}_{name}_threshold_{tag}.csv"
    sub.to_csv(out_path, index=False)

    ones = int(sub["prediction"].sum())
    pos_ratio = float(sub["prediction"].mean())

    print(f"Saved: {out_path}")
    print(f"threshold={threshold:.3f}, ones={ones:,}, pos_ratio={pos_ratio:.6f}")

    return {
        "name": name,
        "threshold": float(threshold),
        "path": str(out_path),
        "ones": ones,
        "zeros": int(len(sub) - ones),
        "pos_ratio": pos_ratio,
    }


def compare_if_exists(summary_rows: list[dict]) -> list[dict]:
    reference_paths = [
        SUB_DIR / "v4_single_threshold_0p96.csv",
        SUB_DIR / "v4_ensemble_weighted_threshold_0p96.csv",
        SUB_DIR / "embedding_ranker_v4_weighted_semantic_e080_m020_topratio_012.csv",
    ]

    for ref_path in reference_paths:
        if not ref_path.exists():
            continue

        ref = pd.read_csv(ref_path)

        for row in summary_rows:
            cand_path = Path(row["path"])
            cand = pd.read_csv(cand_path)

            assert ref["id"].equals(cand["id"])

            diff = int((ref["prediction"] != cand["prediction"]).sum())
            row[f"diff_vs_{ref_path.stem}"] = diff
            row[f"diff_ratio_vs_{ref_path.stem}"] = float(diff / len(cand))

    return summary_rows


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("27 FULL SEMW035 THRESHOLD ENSEMBLE")
    print("=" * 100)

    mod = load_v4_module()

    print("\nBuilding full train frame...")
    train = mod.build_train_frame()

    print("\nBuilding test frame...")
    test = mod.build_test_frame()

    feature_cols = mod.FEATURE_COLS
    cat_features = mod.CAT_FEATURES
    cat_indices = [feature_cols.index(c) for c in cat_features]

    print(f"\nTrain rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    train = train.copy()
    train["full_weight"] = 1.0
    train.loc[train["negative_type"] == "semantic_hard", "full_weight"] = SEMANTIC_WEIGHT
    train["full_weight"] = train["full_weight"].astype(np.float32)

    print("\nWeight distribution:")
    print(train.groupby("negative_type")["full_weight"].describe())

    train_pool = Pool(
        train[feature_cols],
        label=train["label"].astype(int),
        weight=train["full_weight"].astype(float),
        cat_features=cat_indices,
    )

    proba_columns = {}
    proba_sum = np.zeros(len(test), dtype=np.float32)

    model_reports = []

    for seed in SEEDS:
        print("\n" + "=" * 100)
        print(f"Training full model seed={seed}, semw={SEMANTIC_WEIGHT}")
        print("=" * 100)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            iterations=ITERATIONS,
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=7.0,
            random_seed=seed,
            auto_class_weights="Balanced",
            verbose=50,
            allow_writing_files=False,
            task_type="GPU",
            devices="0",
        )

        model.fit(train_pool)

        model_path = MODEL_DIR / f"{RUN_NAME}_seed{seed}.cbm"
        model.save_model(model_path)

        print(f"Saved model: {model_path}")

        print("Predicting test proba...")
        proba = model.predict_proba(test[feature_cols])[:, 1].astype(np.float32)

        col = f"proba_seed{seed}"
        proba_columns[col] = proba
        proba_sum += proba

        model_reports.append(
            {
                "seed": int(seed),
                "model_path": str(model_path),
                "iterations": ITERATIONS,
                "semantic_weight": SEMANTIC_WEIGHT,
            }
        )

    avg_proba = proba_sum / float(len(SEEDS))
    proba_columns["proba_avg"] = avg_proba

    proba_df = pd.DataFrame(
        {
            "id": test["id"].values,
            "term_id": test["term_id"].values,
            "item_id": test["item_id"].values,
            "embedding_score": test["embedding_score"].values,
        }
    )

    for col, arr in proba_columns.items():
        proba_df[col] = arr

    proba_path = PROCESSED_DIR / f"{RUN_NAME}_test_proba.parquet"
    proba_df.to_parquet(proba_path, index=False)

    print(f"\nSaved proba: {proba_path}")

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")

    summary_rows = []

    for name, proba in proba_columns.items():
        for th in THRESHOLDS:
            summary_rows.append(
                make_threshold_submission(
                    sample=sample,
                    ids=test["id"],
                    proba=proba,
                    name=name,
                    threshold=th,
                )
            )

    summary_rows = compare_if_exists(summary_rows)

    summary = pd.DataFrame(summary_rows).sort_values(["name", "threshold"])
    summary_path = REPORT_DIR / f"{RUN_NAME}_summary.csv"
    summary.to_csv(summary_path, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "semantic_weight": SEMANTIC_WEIGHT,
        "seeds": SEEDS,
        "iterations": ITERATIONS,
        "thresholds": THRESHOLDS,
        "model_reports": model_reports,
        "proba_path": str(proba_path),
        "summary_path": str(summary_path),
    }

    report_path = REPORT_DIR / f"{RUN_NAME}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Saved summary: {summary_path}")
    print(f"Saved report : {report_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
