from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

V5_SCRIPT = ROOT / "scripts" / "28_v5_e5_embedding_cv_threshold.py"

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports" / "experiments"

RUN_NAME = "v5_e5base_full900"

SEMANTIC_WEIGHT = 0.35
SEEDS = [42, 2026, 777]
ITERATIONS = 900

THRESHOLDS = [
    0.930,
    0.935,
    0.940,
    0.945,
    0.950,
    0.955,
    0.960,
    0.965,
    0.970,
]

REFERENCE_FILES = [
    "v5_e5base_proba_avg_threshold_0p955.csv",
    "v5_e5base_proba_avg_threshold_0p950.csv",
    "v5_e5base_proba_avg_threshold_0p960.csv",
    "v6_dual_e5_minilm_proba_avg_threshold_0p940.csv",
]


def load_v5_module():
    spec = importlib.util.spec_from_file_location("v5mod", V5_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V5_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def threshold_tag(th: float) -> str:
    return f"{th:.3f}".replace(".", "p")


def make_submission(sample: pd.DataFrame, ids: pd.Series, proba: np.ndarray, name: str, threshold: float) -> dict:
    pred = (proba >= threshold).astype(np.int8)

    tmp = pd.DataFrame({"id": ids.values, "prediction": pred})
    sub = sample[["id"]].merge(tmp, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)

    assert list(sub.columns) == ["id", "prediction"]
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])
    assert set(sub["prediction"].unique()) <= {0, 1}

    out_path = SUBMISSIONS_DIR / f"{RUN_NAME}_{name}_threshold_{threshold_tag(threshold)}.csv"
    sub.to_csv(out_path, index=False)

    row = {
        "name": name,
        "threshold": float(threshold),
        "path": str(out_path),
        "ones": int(sub["prediction"].sum()),
        "zeros": int(len(sub) - sub["prediction"].sum()),
        "pos_ratio": float(sub["prediction"].mean()),
        "distance_to_0p132347": abs(float(sub["prediction"].mean()) - 0.132347),
        "distance_to_0p12967": abs(float(sub["prediction"].mean()) - 0.12967),
    }

    for ref_name in REFERENCE_FILES:
        ref_path = SUBMISSIONS_DIR / ref_name
        if ref_path.exists():
            ref = pd.read_csv(ref_path)
            assert ref["id"].equals(sub["id"])
            diff = int((ref["prediction"] != sub["prediction"]).sum())
            row[f"diff_vs_{ref_path.stem}"] = diff
            row[f"diff_ratio_vs_{ref_path.stem}"] = float(diff / len(sub))

    print(row)
    return row


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("31 V5 FULL 900 THRESHOLD")
    print("=" * 100)

    v5 = load_v5_module()

    mod = v5.load_v4_module()
    mod = v5.patch_v4_module_paths(mod)

    print("Building V5 train frame...")
    train = mod.build_train_frame()

    print("Building V5 test frame...")
    test = mod.build_test_frame()

    print(f"Train rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    feature_cols = mod.FEATURE_COLS
    cat_features = mod.CAT_FEATURES
    cat_indices = [feature_cols.index(c) for c in cat_features]

    tr = train.copy()
    tr["final_weight"] = 1.0
    tr.loc[tr["negative_type"] == "semantic_hard", "final_weight"] = SEMANTIC_WEIGHT
    tr["final_weight"] = tr["final_weight"].astype(np.float32)

    print("Weight distribution:")
    print(tr.groupby("negative_type")["final_weight"].describe())

    train_pool = Pool(
        tr[feature_cols],
        label=tr["label"].astype(int),
        weight=tr["final_weight"].astype(float),
        cat_features=cat_indices,
    )

    proba_sum = np.zeros(len(test), dtype=np.float32)
    proba_cols = {}

    for seed in SEEDS:
        print("\n" + "=" * 100)
        print(f"Training seed={seed}, iterations={ITERATIONS}, semw={SEMANTIC_WEIGHT}")
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
            verbose=100,
            allow_writing_files=False,
            task_type="GPU",
            devices="0",
        )

        model.fit(train_pool)

        model_path = MODELS_DIR / f"{RUN_NAME}_seed{seed}.cbm"
        model.save_model(model_path)
        print("Saved model:", model_path)

        proba = model.predict_proba(test[feature_cols])[:, 1].astype(np.float32)
        col = f"proba_seed{seed}"
        proba_cols[col] = proba
        proba_sum += proba

    proba_cols["proba_avg"] = proba_sum / float(len(SEEDS))

    proba_df = pd.DataFrame(
        {
            "id": test["id"].values,
            "term_id": test["term_id"].values,
            "item_id": test["item_id"].values,
        }
    )

    for col, arr in proba_cols.items():
        proba_df[col] = arr

    proba_path = PROCESSED_DIR / f"{RUN_NAME}_test_proba.parquet"
    proba_df.to_parquet(proba_path, index=False)
    print("Saved proba:", proba_path)

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    rows = []

    for name, proba in proba_cols.items():
        for th in THRESHOLDS:
            rows.append(make_submission(sample, test["id"], proba, name, float(th)))

    summary = pd.DataFrame(rows).sort_values(["distance_to_0p132347", "name", "threshold"])
    summary_path = REPORT_DIR / f"{RUN_NAME}_summary.csv"
    summary.to_csv(summary_path, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": RUN_NAME,
        "semantic_weight": SEMANTIC_WEIGHT,
        "seeds": SEEDS,
        "iterations": ITERATIONS,
        "proba_path": str(proba_path),
        "summary_path": str(summary_path),
        "top_candidates": summary.head(20).to_dict(orient="records"),
    }

    report_path = REPORT_DIR / f"{RUN_NAME}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved summary:", summary_path)
    print("Saved report :", report_path)
    print("\nTop candidates:")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
