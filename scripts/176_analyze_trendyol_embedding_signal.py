"""Measure independence and source-wise audit quality of the full Trendyol signal."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(".")
PROC = ROOT / "data/processed"
EMB = PROC / "v70_trendyol_embedding"
REPORT = ROOT / "reports/experiments/v70_trendyol_full_signal_audit.json"
FEATURES = EMB / "v70_compact_rank_features.parquet"
V22 = ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv"
LABELS = [
    ("v29", ROOT / "reports/manual_review/review_v29_qswap_targets_assistant_labeled.csv"),
    ("v33", ROOT / "reports/manual_review/review_v33_final_qswap_targets_assistant_labeled.csv"),
]


def assert_aligned(left: pd.Series, right: pd.Series, name: str) -> None:
    if len(left) != len(right) or not np.array_equal(left.to_numpy(), right.to_numpy()):
        raise RuntimeError(f"row alignment failed for {name}")


def q_percentile(frame: pd.DataFrame, column: str) -> np.ndarray:
    # High score -> percentile close to one.  Average ties deliberately share a rank.
    return frame.groupby("term_id", sort=False)[column].rank(method="average", pct=True).to_numpy(np.float32)


def load_clean_labels() -> pd.DataFrame:
    parts = []
    for source, path in LABELS:
        d = pd.read_csv(path)
        label = pd.to_numeric(d["assistant_swap_label"], errors="coerce")
        recheck = pd.to_numeric(d.get("needs_recheck", 0), errors="coerce").fillna(1)
        confidence = d.get("assistant_confidence", "").astype(str).str.lower()
        keep = label.isin([0, 1]) & recheck.eq(0) & confidence.isin(["high", "medium"])
        x = d.loc[keep, ["term_id", "item_id_add", "item_id_drop"]].copy()
        x["label"] = label.loc[keep].astype(np.int8)
        x["source"] = source
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def audit_swaps(features: pd.DataFrame) -> list[dict]:
    labels = load_clean_labels()
    cols = ["trendyol_score", "trendyol_qpct", "lex_qpct", "v21_qpct", "v69_qpct"]
    key = features[["term_id", "item_id"] + cols]
    add = key.rename(columns={"item_id": "item_id_add", **{c: c + "_add" for c in cols}})
    drop = key.rename(columns={"item_id": "item_id_drop", **{c: c + "_drop" for c in cols}})
    data = labels.merge(add, on=["term_id", "item_id_add"], how="left", validate="many_to_one")
    data = data.merge(drop, on=["term_id", "item_id_drop"], how="left", validate="many_to_one")
    if data.filter(regex="_(add|drop)$").isna().any().any():
        raise RuntimeError("manual audit item missing from full candidate features")
    for c in cols:
        data[c + "_margin"] = data[c + "_add"] - data[c + "_drop"]
    margins = {
        "trendyol_raw": data["trendyol_score_margin"],
        "trendyol_rank": data["trendyol_qpct_margin"],
        "lex_rank": data["lex_qpct_margin"],
        "v21_rank": data["v21_qpct_margin"],
        "v69_rank": data["v69_qpct_margin"],
        "trend_lex_equal": .50*data["trendyol_qpct_margin"] + .50*data["lex_qpct_margin"],
        "trend_v21_equal": .50*data["trendyol_qpct_margin"] + .50*data["v21_qpct_margin"],
        "trend_lex_v21_equal": (data["trendyol_qpct_margin"] + data["lex_qpct_margin"] + data["v21_qpct_margin"])/3,
        "trend50_lex30_v21_20": .50*data["trendyol_qpct_margin"] + .30*data["lex_qpct_margin"] + .20*data["v21_qpct_margin"],
        "trend40_lex25_v21_20_v69_15": .40*data["trendyol_qpct_margin"] + .25*data["lex_qpct_margin"] + .20*data["v21_qpct_margin"] + .15*data["v69_qpct_margin"],
    }
    rows = []
    for name, margin in margins.items():
        for source in ["v29", "v33", "all"]:
            mask = np.ones(len(data), dtype=bool) if source == "all" else data["source"].eq(source).to_numpy()
            y, score = data.loc[mask, "label"], margin[mask]
            rows.append({"signal": name, "source": source, "n": int(mask.sum()),
                         "auc": float(roc_auc_score(y, score)),
                         "ap": float(average_precision_score(y, score))})
    pd.DataFrame(rows).to_csv(REPORT.with_suffix(".csv"), index=False)
    return rows


def main() -> None:
    pairs = pd.read_csv(ROOT / "data/raw/submission_pairs.csv", dtype={"id": "string", "term_id": "string", "item_id": "string"})
    scores = np.load(EMB / "v70_trendyol_pair_scores.npy", mmap_mode="r")
    if len(scores) != len(pairs) or not np.isfinite(scores).all():
        raise RuntimeError("invalid Trendyol score vector")
    frame = pairs.copy()
    frame["trendyol_score"] = np.asarray(scores, dtype=np.float32)
    frame["trendyol_qpct"] = q_percentile(frame, "trendyol_score")

    v21 = pd.read_parquet(PROC / "v21_catboost_test_scores.parquet", columns=["id", "v21_score"])
    lex = pd.read_parquet(PROC / "v38_lexical_pair_scores.parquet", columns=["id", "v38_lex_score"])
    v69 = pd.read_parquet(PROC / "v69_fast_poison_guarded_scores.parquet", columns=["id", "v69_guarded_score"])
    for name, other in [("v21", v21), ("lex", lex), ("v69", v69)]:
        assert_aligned(frame["id"], other["id"].astype("string"), name)
    frame["v21_score"] = v21["v21_score"].to_numpy(np.float32)
    frame["lex_score"] = lex["v38_lex_score"].to_numpy(np.float32)
    frame["v69_score"] = v69["v69_guarded_score"].to_numpy(np.float32)
    frame["v21_qpct"] = q_percentile(frame, "v21_score")
    frame["lex_qpct"] = q_percentile(frame, "lex_score")
    frame["v69_qpct"] = q_percentile(frame, "v69_score")

    v22 = pd.read_csv(V22)
    assert_aligned(frame["id"], v22["id"].astype("string"), "v22")
    prediction = v22["prediction"].to_numpy(np.int8)
    sample = frame.sample(n=min(250_000, len(frame)), random_state=20260702)
    rank_cols = ["trendyol_qpct", "lex_qpct", "v21_qpct", "v69_qpct"]
    corr = sample[rank_cols].corr(method="spearman").round(6).to_dict()
    v22_auc = float(roc_auc_score(prediction, frame["trendyol_qpct"]))
    v22_ratio = float(prediction.mean())
    swap_rows = audit_swaps(frame)

    compact = frame[["id", "term_id", "item_id", "trendyol_score", "trendyol_qpct",
                     "lex_qpct", "v21_qpct", "v69_qpct"]]
    compact.to_parquet(FEATURES, index=False)
    result = {
        "rows": len(frame), "terms": int(frame["term_id"].nunique()), "v22_positive_ratio": v22_ratio,
        "trendyol_auc_for_v22_labels_not_truth": v22_auc,
        "trendyol_mean_v22_positive": float(frame.loc[prediction == 1, "trendyol_score"].mean()),
        "trendyol_mean_v22_negative": float(frame.loc[prediction == 0, "trendyol_score"].mean()),
        "sample_spearman": corr, "swap_audit": swap_rows,
        "warning": "Manual swap labels and V22 agreement are diagnostics, not hidden Kaggle truth."
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    table = pd.DataFrame(swap_rows)
    pivot = table.pivot(index="signal", columns="source", values="auc").sort_values(["v29", "v33"], ascending=False)
    print(pivot.to_string())
    print(json.dumps({k: result[k] for k in result if k != "swap_audit"}, ensure_ascii=False, indent=2))
    print("saved", FEATURES, REPORT)


if __name__ == "__main__":
    main()
