from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def clipped_logit(values: pd.Series | np.ndarray, eps: float = 1e-5) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.clip(array, eps, 1.0 - eps)
    return np.log(array / (1.0 - array)).astype(np.float32)


def assert_same_rows(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], label: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{label}: row count mismatch ({len(left):,} != {len(right):,})")
    for key in keys:
        left_values = left[key].astype(str).reset_index(drop=True)
        right_values = right[key].astype(str).reset_index(drop=True)
        if not left_values.equals(right_values):
            mismatch = np.flatnonzero(left_values.to_numpy() != right_values.to_numpy())
            row = int(mismatch[0]) if len(mismatch) else -1
            raise ValueError(f"{label}: {key!r} alignment mismatch at row {row}")


def add_meta_features(
    frame: pd.DataFrame,
    score_columns: tuple[str, str] = ("v106_score", "v108_score"),
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    first, second = score_columns
    for column in score_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.5).clip(0.0, 1.0).astype(np.float32)
        out[f"{column}_logit"] = clipped_logit(out[column])
        out[f"{column}_rank_pct"] = (
            out.groupby("term_id", sort=False)[column].rank(method="average", ascending=False, pct=True).astype(np.float32)
        )
        out[f"{column}_delta_max"] = (
            out.groupby("term_id", sort=False)[column].transform("max") - out[column]
        ).astype(np.float32)

    out["score_mean"] = ((out[first] + out[second]) * 0.5).astype(np.float32)
    out["score_min"] = out[[first, second]].min(axis=1).astype(np.float32)
    out["score_max"] = out[[first, second]].max(axis=1).astype(np.float32)
    out["score_abs_gap"] = (out[first] - out[second]).abs().astype(np.float32)
    out["rank_mean"] = (
        (out[f"{first}_rank_pct"] + out[f"{second}_rank_pct"]) * 0.5
    ).astype(np.float32)
    out["candidate_count"] = out.groupby("term_id", sort=False)["term_id"].transform("size").astype(np.float32)
    out["candidate_count_log1p"] = np.log1p(out["candidate_count"]).astype(np.float32)

    features = [
        first,
        second,
        f"{first}_logit",
        f"{second}_logit",
        f"{first}_rank_pct",
        f"{second}_rank_pct",
        f"{first}_delta_max",
        f"{second}_delta_max",
        "score_mean",
        "score_min",
        "score_max",
        "score_abs_gap",
        "rank_mean",
        "candidate_count_log1p",
    ]
    return out, features


def threshold_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    folds: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 193)

    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    folds = np.asarray(folds)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        prediction = (probabilities >= threshold).astype(np.int8)
        fold_scores = [
            f1_score(labels[folds == fold], prediction[folds == fold], average="macro", labels=[0, 1])
            for fold in np.unique(folds)
        ]
        macro = f1_score(labels, prediction, average="macro", labels=[0, 1])
        f1_negative, f1_positive = f1_score(labels, prediction, average=None, labels=[0, 1])
        fold_mean = float(np.mean(fold_scores))
        fold_std = float(np.std(fold_scores))
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(macro),
                "f1_negative": float(f1_negative),
                "f1_positive": float(f1_positive),
                "fold_mean_macro_f1": fold_mean,
                "fold_std_macro_f1": fold_std,
                "fold_min_macro_f1": float(np.min(fold_scores)),
                "selection_score": fold_mean - 0.25 * fold_std,
                "positive_ratio": float(prediction.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["selection_score", "fold_min_macro_f1", "macro_f1"], ascending=False
    ).reset_index(drop=True)
