"""Train a leakage-safe V106/V108 meta ensemble and create two-way predictions.

Required artifacts are produced by scripts 210 and 212. The meta layer uses the
existing term-grouped OOF predictions, then performs another fold split while fitting
the meta model. This prevents evaluating the meta learner on rows it fitted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from trendyol.meta_ensemble import add_meta_features, assert_same_rows, threshold_report


ROOT = Path(".")
OOF_V106 = ROOT / "data/processed/v106_generalized_oof_scores.parquet"
TEST_V106 = ROOT / "data/processed/v106_generalized_test_scores.parquet"
OOF_V108 = ROOT / "data/processed/v108_source_matched_oof_scores.parquet"
TEST_V108 = ROOT / "data/processed/v108_source_matched_test_scores.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"
ANCHOR = ROOT / "submissions/final_candidates_v108/FINAL_CANDIDATE_v108_bridge_top.csv"

OOF_OUT = ROOT / "data/processed/v218_meta_oof_scores.parquet"
TEST_OUT = ROOT / "data/processed/v218_meta_test_scores.parquet"
REPORT = ROOT / "reports/experiments/v218_meta_ensemble.json"
THRESHOLD_REPORT = ROOT / "reports/experiments/v218_threshold_report.csv"
OUT_DIR = ROOT / "submissions/final_candidates_v218"

SEED = 20260710


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [OOF_V106, TEST_V106, OOF_V108, TEST_V108, SAMPLE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V218 inputs:\n- " + "\n- ".join(missing))

    oof106 = pd.read_parquet(OOF_V106)
    test106 = pd.read_parquet(TEST_V106)
    oof108 = pd.read_parquet(OOF_V108)
    test108 = pd.read_parquet(TEST_V108)

    assert_same_rows(oof106, oof108, ["term_id", "item_id"], "OOF V106/V108")
    assert_same_rows(test106, test108, ["id", "term_id", "item_id"], "test V106/V108")

    if not oof106["label"].astype(np.int8).equals(oof108["label"].astype(np.int8)):
        raise ValueError("OOF label mismatch between V106 and V108")
    if not oof106["fold5"].astype(np.int8).equals(oof108["fold5"].astype(np.int8)):
        raise ValueError("OOF fold mismatch between V106 and V108")

    train = oof108[["term_id", "item_id", "label", "fold5"]].copy()
    train["v106_score"] = oof106["oof_score"].astype(np.float32).to_numpy()
    train["v108_score"] = oof108["v108_score"].astype(np.float32).to_numpy()

    test = test108[["id", "term_id", "item_id"]].copy()
    test["v106_score"] = test106["v106_score"].astype(np.float32).to_numpy()
    test["v108_score"] = test108["v108_score"].astype(np.float32).to_numpy()
    for frame in (train, test):
        frame["term_id"] = frame["term_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
    test["id"] = test["id"].astype(str)
    return train, test


def make_model() -> LogisticRegression:
    return LogisticRegression(
        C=float(os.environ.get("V218_C", "0.25")),
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        random_state=SEED,
    )


def save_submission(name: str, ids: np.ndarray, prediction: np.ndarray) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"FINAL_CANDIDATE_v218_{name}.csv"
    pd.DataFrame({"id": ids, "prediction": prediction.astype(np.int8)}).to_csv(path, index=False)
    print("saved", path, "ones", int(prediction.sum()), "ratio", float(prediction.mean()), flush=True)
    return path


def main() -> None:
    train, test = load_frames()
    train, features = add_meta_features(train)
    test, test_features = add_meta_features(test)
    if features != test_features:
        raise AssertionError("train/test meta features differ")

    x = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    xtest = test[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    y = train["label"].astype(np.int8).to_numpy()
    folds = train["fold5"].astype(np.int8).to_numpy()

    meta_oof = np.empty(len(train), dtype=np.float32)
    fold_models = []
    for fold in np.unique(folds):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = make_model()
        model.fit(x.loc[train_mask], y[train_mask])
        meta_oof[valid_mask] = model.predict_proba(x.loc[valid_mask])[:, 1].astype(np.float32)
        fold_models.append(model)
        print({"fold": int(fold), "train": int(train_mask.sum()), "valid": int(valid_mask.sum())}, flush=True)

    threshold_table = threshold_report(y, meta_oof, folds)
    best = threshold_table.iloc[0]
    threshold = float(best["threshold"])
    print("selected threshold", threshold, "OOF macro", float(best["macro_f1"]), flush=True)

    final_model = make_model()
    final_model.fit(x, y)
    test_probability = final_model.predict_proba(xtest)[:, 1].astype(np.float32)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    assert_same_rows(sample, test, ["id"], "sample/test")
    ids = sample["id"].to_numpy()
    pure_prediction = (test_probability >= threshold).astype(np.int8)

    outputs = {"pure_oof_threshold": str(save_submission("pure_oof_threshold", ids, pure_prediction))}
    anchor_changes = None
    if ANCHOR.exists():
        anchor = pd.read_csv(ANCHOR)
        anchor["id"] = anchor["id"].astype(str)
        assert_same_rows(sample, anchor, ["id"], "sample/anchor")
        anchor_pred = anchor["prediction"].astype(np.int8).to_numpy()

        # Only override the anchor when the calibrated model is outside its OOF
        # uncertainty band. The band is derived from the selected threshold.
        margin = float(os.environ.get("V218_CONFIDENCE_MARGIN", "0.08"))
        conservative = anchor_pred.copy()
        conservative[test_probability >= min(0.999, threshold + margin)] = 1
        conservative[test_probability <= max(0.001, threshold - margin)] = 0
        outputs["anchor_confident_twoway"] = str(
            save_submission("anchor_confident_twoway", ids, conservative)
        )
        anchor_changes = {
            "total": int((conservative != anchor_pred).sum()),
            "zero_to_one": int(((anchor_pred == 0) & (conservative == 1)).sum()),
            "one_to_zero": int(((anchor_pred == 1) & (conservative == 0)).sum()),
        }

    OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    train_out = train[["term_id", "item_id", "label", "fold5"]].copy()
    train_out["v218_meta_oof"] = meta_oof
    train_out.to_parquet(OOF_OUT, index=False)
    test_out = test[["id", "term_id", "item_id"]].copy()
    test_out["v218_meta_score"] = test_probability
    test_out.to_parquet(TEST_OUT, index=False)
    THRESHOLD_REPORT.parent.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(THRESHOLD_REPORT, index=False)

    report = {
        "features": features,
        "rows": {"train": len(train), "test": len(test)},
        "selected_threshold": threshold,
        "selected_metrics": {key: float(value) for key, value in best.to_dict().items()},
        "anchor_changes": anchor_changes,
        "outputs": outputs,
        "validation": "nested term-fold OOF meta predictions",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print("saved", REPORT, flush=True)


if __name__ == "__main__":
    main()
