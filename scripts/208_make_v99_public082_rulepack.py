"""Build small rule-pack refinements around the public-0.82 v95 anchor."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(".")
TRAIN = ROOT / "data/processed/v82_v34_features.parquet"
HOLD = ROOT / "models/v76_trendyol_contrastive/holdout_terms.txt"
FEATURE_REPORT = ROOT / "reports/experiments/v83_independent_pairlocal_classifier.json"
MODEL = ROOT / "models/v83_independent_pairlocal_classifier.cbm"

BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
SCORE = ROOT / "data/processed/v83_independent_test_scores.parquet"
FEAT = ROOT / "data/processed/v21_test_features.parquet"

OUT = ROOT / "submissions/final_candidates_v99"
REPORT = ROOT / "reports/experiments/v99_public082_rulepack.json"


def neg_precision(y: pd.Series) -> float:
    return float((1 - y.to_numpy(np.int8)).mean())


def save_variant(ids: pd.Series, pred: np.ndarray, name: str, base: np.ndarray, rows: list[dict]) -> None:
    path = OUT / f"FINAL_CANDIDATE_v99_{name}.csv"
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(path, index=False)
    rows.append(
        {
            "variant": name,
            "removed_vs_base": int((base == 1).sum() - pred.sum()),
            "positives": int(pred.sum()),
            "ratio": float(pred.mean()),
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )


def main() -> None:
    features = json.loads(FEATURE_REPORT.read_text())["features"]
    train = pd.read_parquet(TRAIN)
    train.term_id = train.term_id.astype(str)
    hold = set(HOLD.read_text().splitlines())
    valid = train[train.term_id.isin(hold)].copy()

    model = CatBoostClassifier()
    model.load_model(str(MODEL))
    valid["v83_score"] = model.predict_proba(valid[features].fillna(-1).astype("float32"))[:, 1]

    base = pd.read_csv(BASE)
    score = pd.read_parquet(SCORE, columns=["id", "v82_score"])
    feat = pd.read_parquet(
        FEAT,
        columns=[
            "id",
            "query_has_color",
            "query_has_known_brand",
            "gender_mismatch",
            "color_mismatch",
            "query_brand_mismatch",
            "title_cov_pct_rank",
            "weighted_overlap_pct_rank",
            "last_token_in_title",
            "brand_in_query_pair",
        ],
    )
    for df in (base, score, feat):
        df.id = df.id.astype(str)
    test = base.merge(score, on="id").merge(feat, on="id")
    if not test.id.equals(base.id):
        raise RuntimeError("alignment")

    valid_rules = {
        "gender_or_color": (
            valid.gender_mismatch.eq(1) & valid.v83_score.le(0.15)
        ) | (
            valid.query_has_color.eq(1) & valid.color_mismatch.eq(1) & valid.v83_score.le(0.10)
        ),
        "brand_tail006_tcov50_last0_biq0": (
            valid.query_has_known_brand.eq(1)
            & valid.query_brand_mismatch.eq(1)
            & valid.v83_score.le(0.06)
            & valid.title_cov_pct_rank.le(0.50)
            & valid.last_token_in_title.eq(0)
            & valid.brand_in_query_pair.eq(0)
        ),
        "brand_tail008_tcov50_last0_biq0": (
            valid.query_has_known_brand.eq(1)
            & valid.query_brand_mismatch.eq(1)
            & valid.v83_score.le(0.08)
            & valid.title_cov_pct_rank.le(0.50)
            & valid.last_token_in_title.eq(0)
            & valid.brand_in_query_pair.eq(0)
        ),
        "brand_tail008_wov50_last0_biq0": (
            valid.query_has_known_brand.eq(1)
            & valid.query_brand_mismatch.eq(1)
            & valid.v83_score.le(0.08)
            & valid.weighted_overlap_pct_rank.le(0.50)
            & valid.last_token_in_title.eq(0)
            & valid.brand_in_query_pair.eq(0)
        ),
    }
    test_rules = {
        "gender_or_color": (
            test.prediction.eq(1) & test.gender_mismatch.eq(1) & test.v82_score.le(0.15)
        ) | (
            test.prediction.eq(1) & test.query_has_color.eq(1) & test.color_mismatch.eq(1) & test.v82_score.le(0.10)
        ),
        "brand_tail006_tcov50_last0_biq0": (
            test.prediction.eq(1)
            & test.query_has_known_brand.eq(1)
            & test.query_brand_mismatch.eq(1)
            & test.v82_score.le(0.06)
            & test.title_cov_pct_rank.le(0.50)
            & test.last_token_in_title.eq(0)
            & test.brand_in_query_pair.eq(0)
        ),
        "brand_tail008_tcov50_last0_biq0": (
            test.prediction.eq(1)
            & test.query_has_known_brand.eq(1)
            & test.query_brand_mismatch.eq(1)
            & test.v82_score.le(0.08)
            & test.title_cov_pct_rank.le(0.50)
            & test.last_token_in_title.eq(0)
            & test.brand_in_query_pair.eq(0)
        ),
        "brand_tail008_wov50_last0_biq0": (
            test.prediction.eq(1)
            & test.query_has_known_brand.eq(1)
            & test.query_brand_mismatch.eq(1)
            & test.v82_score.le(0.08)
            & test.weighted_overlap_pct_rank.le(0.50)
            & test.last_token_in_title.eq(0)
            & test.brand_in_query_pair.eq(0)
        ),
    }

    valid_rules["gender_color_brand006"] = valid_rules["gender_or_color"] | valid_rules["brand_tail006_tcov50_last0_biq0"]
    valid_rules["gender_color_brand008"] = valid_rules["gender_or_color"] | valid_rules["brand_tail008_tcov50_last0_biq0"]
    valid_rules["gender_color_brandwov008"] = valid_rules["gender_or_color"] | valid_rules["brand_tail008_wov50_last0_biq0"]

    test_rules["gender_color_brand006"] = test_rules["gender_or_color"] | test_rules["brand_tail006_tcov50_last0_biq0"]
    test_rules["gender_color_brand008"] = test_rules["gender_or_color"] | test_rules["brand_tail008_tcov50_last0_biq0"]
    test_rules["gender_color_brandwov008"] = test_rules["gender_or_color"] | test_rules["brand_tail008_wov50_last0_biq0"]

    calibration = []
    for name, mask in valid_rules.items():
        rows = int(mask.sum())
        calibration.append(
            {
                "rule": name,
                "holdout_rows": rows,
                "holdout_positive_errors": int(valid.loc[mask, "label"].sum()),
                "holdout_negative_precision": neg_precision(valid.loc[mask, "label"]) if rows else np.nan,
                "test_positive_hits": int(test_rules[name].sum()),
            }
        )

    base_pred = base.prediction.to_numpy(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    for name, mask in test_rules.items():
        pred = base_pred.copy()
        pred[mask.to_numpy()] = 0
        save_variant(base.id, pred, name, base_pred, candidates)

    report = {
        "base_file": str(BASE),
        "base_positives": int(base_pred.sum()),
        "calibration": calibration,
        "candidates": candidates,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
