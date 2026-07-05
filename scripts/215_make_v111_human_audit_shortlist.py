"""Build a deduplicated human-audit shortlist from the v110 audit blockpack."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import random

import pandas as pd


ROOT = Path(".")
SRC = ROOT / "reports/manual_review/v110_audit_blockpack.csv"
OUT = ROOT / "reports/manual_review/v111_human_audit_shortlist.csv"
SUMMARY = ROOT / "reports/experiments/v111_human_audit_shortlist_summary.json"

RNG = random.Random(20260705)

BUCKET_CONFIG = {
    "v104_only_vs_v107": {"limit": 24, "recommended_decision": "lock_rule_family", "sort_cols": ["v108_score"]},
    "v108_only_removal": {"limit": 24, "recommended_decision": "verify_and_maybe_expand", "sort_cols": ["v108_score"]},
    "repair_general_nohead": {"limit": 16, "recommended_decision": "lock_repair_family", "sort_cols": ["min_margin", "comp_gap"]},
    "repair_same_head_swap": {"limit": 16, "recommended_decision": "lock_repair_family", "sort_cols": ["min_margin", "comp_gap"]},
    "repair_cross_root": {"limit": 16, "recommended_decision": "lock_repair_family", "sort_cols": ["min_margin", "comp_gap"]},
    "repair_device_domain": {"limit": 8, "recommended_decision": "lock_repair_family", "sort_cols": ["min_margin", "comp_gap"]},
    "repair_head_safe": {"limit": 11, "recommended_decision": "review_cautiously", "sort_cols": ["min_margin", "comp_gap"]},
}


def sample_group(df: pd.DataFrame, limit: int, sort_cols: list[str]) -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()
    top_n = min(max(8, limit // 2), len(df))
    top = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(top_n)
    remaining = df.drop(index=top.index)
    if len(top) >= limit or remaining.empty:
        return top.head(limit).copy()

    picked = []
    seen_terms = set()
    for row in remaining.itertuples():
        if row.term_id in seen_terms:
            continue
        seen_terms.add(row.term_id)
        picked.append(row.Index)
        if len(top) + len(picked) >= limit:
            break
    if len(top) + len(picked) < limit:
        extra = [idx for idx in remaining.index if idx not in picked]
        RNG.shuffle(extra)
        picked.extend(extra[: limit - len(top) - len(picked)])
    return pd.concat([top, remaining.loc[picked]], ignore_index=False).head(limit).copy()


def dedup_key(df: pd.DataFrame) -> pd.Series:
    if "add_id" in df.columns and "drop_id" in df.columns:
        pair_key = df["add_id"].fillna("").astype(str) + "|" + df["drop_id"].fillna("").astype(str)
        if pair_key.ne("|").any():
            return pair_key
    if "id" in df.columns:
        return df["id"].fillna("").astype(str)
    return df["term_id"].astype(str) + "|" + df["query"].astype(str)


def main():
    df = pd.read_csv(SRC)
    if df.empty:
        OUT.write_text("", encoding="utf8")
        SUMMARY.write_text(json.dumps({"source_file": str(SRC), "rows": 0}, indent=2), encoding="utf8")
        return

    rows = []
    bucket_summaries = []

    for bucket, cfg in BUCKET_CONFIG.items():
        grp = df[df["bucket"] == bucket].copy()
        if grp.empty:
            continue
        grp["_dedup"] = dedup_key(grp)
        grp = grp.drop_duplicates("_dedup", keep="first").copy()
        sample = sample_group(grp, cfg["limit"], cfg["sort_cols"])
        sample["recommended_decision"] = cfg["recommended_decision"]
        sample["human_label"] = ""
        sample["notes"] = ""
        rows.append(sample)
        bucket_summaries.append(
            {
                "bucket": bucket,
                "dedup_rows": int(len(grp)),
                "sample_rows": int(len(sample)),
                "recommended_decision": cfg["recommended_decision"],
            }
        )

    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    ordered_cols = [
        "bucket",
        "priority",
        "recommended_decision",
        "human_label",
        "notes",
        "source_family",
        "variant",
        "term_id",
        "query",
        "query_intent",
        "query_device_family",
        "query_heads",
        "id",
        "item_id",
        "item_title",
        "item_category",
        "item_brand",
        "item_gender",
        "v108_score",
        "reason_text",
        "add_id",
        "add_item_id",
        "add_title",
        "add_category",
        "add_brand",
        "add_gender",
        "add_heads",
        "drop_id",
        "drop_item_id",
        "drop_title",
        "drop_category",
        "drop_brand",
        "drop_gender",
        "drop_heads",
        "q_a_head_overlap",
        "q_d_head_overlap",
        "add_root",
        "drop_root",
        "add_v108_score",
        "drop_v108_score",
        "add_v82_score",
        "drop_v82_score",
        "min_margin",
        "comp_gap",
    ]
    keep_cols = [c for c in ordered_cols if c in out.columns] + [c for c in out.columns if c not in ordered_cols]
    out = out[keep_cols]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    summary = {
        "source_file": str(SRC),
        "output_file": str(OUT),
        "rows": int(len(out)),
        "buckets": bucket_summaries,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
