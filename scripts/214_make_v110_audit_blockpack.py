"""Build a high-signal audit blockpack from v108 removals and v109 repair attempts."""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import random

import numpy as np
import pandas as pd


ROOT = Path(".")

BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
V104 = ROOT / "submissions/final_candidates_v104/FINAL_CANDIDATE_v104_full_balanced.csv"
V107 = ROOT / "submissions/final_candidates_v107/FINAL_CANDIDATE_v107_objective_certified_top.csv"
V108 = ROOT / "submissions/final_candidates_v108/FINAL_CANDIDATE_v108_bridge_top.csv"
V109_REVIEW = ROOT / "reports/manual_review/v109_bridge_targeted_repair_review.csv"
PAIR = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
V108_SCORES = ROOT / "data/processed/v108_source_matched_test_scores.parquet"
SCRIPT_209 = ROOT / "scripts/209_make_v100_multi_filterpack.py"

OUT_CSV = ROOT / "reports/manual_review/v110_audit_blockpack.csv"
OUT_JSON = ROOT / "reports/experiments/v110_audit_blockpack_summary.json"

RNG = random.Random(20260705)
MAX_PER_BUCKET = 40


def load_mod209():
    spec = importlib.util.spec_from_file_location("mod209", SCRIPT_209)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_item_lookup(item_ids: set[str]) -> dict[str, dict]:
    items = pd.read_csv(
        ITEMS,
        usecols=["item_id", "title", "category", "brand", "gender", "age_group", "attributes"],
        low_memory=False,
    )
    items["item_id"] = items["item_id"].astype(str)
    items = items[items["item_id"].isin(item_ids)].copy()
    out = {}
    for row in items.itertuples(index=False):
        out[str(row.item_id)] = {
            "title": row.title,
            "category": row.category,
            "brand": row.brand,
            "gender": row.gender,
            "age_group": row.age_group,
            "attributes": row.attributes,
        }
    return out


def sample_bucket(df: pd.DataFrame, n: int = MAX_PER_BUCKET, sort_cols: list[str] | None = None) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    if sort_cols:
        top_n = min(max(12, n // 2), len(df))
        top = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(top_n)
        rem = df.drop(index=top.index)
        need = n - len(top)
        if need > 0 and len(rem) > 0:
            take = rem.groupby("term_id", group_keys=False).head(1)
            if len(take) < need:
                extra_idx = [idx for idx in rem.index if idx not in take.index]
                RNG.shuffle(extra_idx)
                take = pd.concat([take, rem.loc[extra_idx[: need - len(take)]]], ignore_index=False)
            else:
                take = take.head(need)
            return pd.concat([top, take]).head(n).copy()
        return top.head(n).copy()
    idx = list(df.index)
    RNG.shuffle(idx)
    return df.loc[idx[:n]].copy()


def build_v108_only_removals(mod209) -> pd.DataFrame:
    base = pd.read_csv(BASE)
    v107 = pd.read_csv(V107)
    v108 = pd.read_csv(V108)
    pairs = pd.read_csv(PAIR, usecols=["id", "term_id", "item_id"])
    scores = pd.read_parquet(V108_SCORES, columns=["id", "v108_score"])
    for df in (base, v107, v108, pairs, scores):
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        if "term_id" in df.columns:
            df["term_id"] = df["term_id"].astype(str)
        if "item_id" in df.columns:
            df["item_id"] = df["item_id"].astype(str)
    merged = (
        base[["id", "prediction"]]
        .rename(columns={"prediction": "base"})
        .merge(v107[["id", "prediction"]].rename(columns={"prediction": "v107"}), on="id")
        .merge(v108[["id", "prediction"]].rename(columns={"prediction": "v108"}), on="id")
        .merge(pairs, on="id")
        .merge(scores, on="id", how="left")
    )
    rem = merged[(merged["base"].eq(1)) & (merged["v107"].eq(1)) & (merged["v108"].eq(0))].copy()
    if rem.empty:
        return rem

    brand_tokens = mod209.build_brand_token_set()
    query_info = mod209.build_query_info(set(rem["term_id"]), brand_tokens)
    item_info = mod209.build_item_meta(set(rem["item_id"]))

    rows = rem[["id", "term_id", "item_id"]].copy()
    rows["score"] = rem["v108_score"].fillna(0).astype(np.float32)
    rows["title_cov_pct_rank"] = np.float32(0.5)
    rows["label"] = -1
    eval_df = mod209.evaluate_rows(
        rows[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]],
        query_info,
        item_info,
        {},
    )

    terms = pd.read_csv(TERMS, usecols=["term_id", "query"])
    terms["term_id"] = terms["term_id"].astype(str)
    item_lookup = load_item_lookup(set(rem["item_id"]))
    rem = rem.merge(eval_df[["id", "query_intent", "query_device_family", "reason_text"]], on="id", how="left")
    rem = rem.merge(terms, on="term_id", how="left")
    rem["bucket"] = "v108_only_removal"
    rem["priority"] = "verify_veto"
    rem["query_heads"] = rem["term_id"].map(lambda t: "|".join(sorted(query_info.get(str(t), {}).get("head_labels", []))))
    rem["item_heads"] = rem["item_id"].map(lambda i: "|".join(sorted(item_info.get(str(i), {}).get("head_labels", []))))
    rem["item_title"] = rem["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("title"))
    rem["item_category"] = rem["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("category"))
    rem["item_brand"] = rem["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("brand"))
    rem["item_gender"] = rem["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("gender"))
    cols = [
        "bucket",
        "priority",
        "id",
        "term_id",
        "query",
        "query_intent",
        "query_device_family",
        "query_heads",
        "item_id",
        "item_title",
        "item_category",
        "item_brand",
        "item_gender",
        "v108_score",
        "reason_text",
    ]
    return rem[cols].copy()


def build_v104_v107_diff(mod209) -> pd.DataFrame:
    base = pd.read_csv(BASE)
    v104 = pd.read_csv(V104)
    v107 = pd.read_csv(V107)
    pairs = pd.read_csv(PAIR, usecols=["id", "term_id", "item_id"])
    scores = pd.read_parquet(V108_SCORES, columns=["id", "v108_score"])
    for df in (base, v104, v107, pairs, scores):
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        if "term_id" in df.columns:
            df["term_id"] = df["term_id"].astype(str)
        if "item_id" in df.columns:
            df["item_id"] = df["item_id"].astype(str)
    merged = (
        base[["id", "prediction"]]
        .rename(columns={"prediction": "base"})
        .merge(v104[["id", "prediction"]].rename(columns={"prediction": "v104"}), on="id")
        .merge(v107[["id", "prediction"]].rename(columns={"prediction": "v107"}), on="id")
        .merge(pairs, on="id")
        .merge(scores, on="id", how="left")
    )
    diff = merged[(merged["base"].eq(1)) & (merged["v104"].eq(0)) & (merged["v107"].eq(1))].copy()
    if diff.empty:
        return diff

    brand_tokens = mod209.build_brand_token_set()
    query_info = mod209.build_query_info(set(diff["term_id"]), brand_tokens)
    item_info = mod209.build_item_meta(set(diff["item_id"]))
    rows = diff[["id", "term_id", "item_id"]].copy()
    rows["score"] = diff["v108_score"].fillna(0).astype(np.float32)
    rows["title_cov_pct_rank"] = np.float32(0.5)
    rows["label"] = -1
    eval_df = mod209.evaluate_rows(
        rows[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]],
        query_info,
        item_info,
        {},
    )
    terms = pd.read_csv(TERMS, usecols=["term_id", "query"])
    terms["term_id"] = terms["term_id"].astype(str)
    item_lookup = load_item_lookup(set(diff["item_id"]))
    diff = diff.merge(eval_df[["id", "query_intent", "query_device_family", "reason_text"]], on="id", how="left")
    diff = diff.merge(terms, on="term_id", how="left")
    diff["bucket"] = "v104_only_vs_v107"
    diff["priority"] = "lock_old_broad_rule"
    diff["query_heads"] = diff["term_id"].map(lambda t: "|".join(sorted(query_info.get(str(t), {}).get("head_labels", []))))
    diff["item_heads"] = diff["item_id"].map(lambda i: "|".join(sorted(item_info.get(str(i), {}).get("head_labels", []))))
    diff["item_title"] = diff["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("title"))
    diff["item_category"] = diff["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("category"))
    diff["item_brand"] = diff["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("brand"))
    diff["item_gender"] = diff["item_id"].map(lambda i: item_lookup.get(str(i), {}).get("gender"))
    cols = [
        "bucket",
        "priority",
        "id",
        "term_id",
        "query",
        "query_intent",
        "query_device_family",
        "query_heads",
        "item_id",
        "item_title",
        "item_category",
        "item_brand",
        "item_gender",
        "v108_score",
        "reason_text",
    ]
    return diff[cols].copy()


def build_v109_buckets(mod209) -> pd.DataFrame:
    rev = pd.read_csv(V109_REVIEW)
    rev = rev[rev["variant"].isin(["m0p5", "m1", "m1p5", "m2"])].copy()
    if rev.empty:
        return rev

    brand_tokens = mod209.build_brand_token_set()
    query_info = mod209.build_query_info(set(rev["term_id"].astype(str)), brand_tokens)
    item_info = mod209.build_item_meta(set(rev["add_item_id"].astype(str)).union(set(rev["drop_item_id"].astype(str))))

    bucket_rows = []
    for row in rev.itertuples(index=False):
        q = query_info.get(str(row.term_id), {})
        ai = item_info.get(str(row.add_item_id), {})
        di = item_info.get(str(row.drop_item_id), {})
        q_heads = set(q.get("head_labels", set()))
        a_heads = set(ai.get("head_labels", set()))
        d_heads = set(di.get("head_labels", set()))
        q_a_overlap = int(bool(q_heads & a_heads))
        q_d_overlap = int(bool(q_heads & d_heads))

        buckets = []
        if q.get("intent") == "general" and not q_heads:
            buckets.append(("repair_general_nohead", "lock_repair"))
        if q_a_overlap == 1 and q_d_overlap == 1:
            buckets.append(("repair_same_head_swap", "lock_repair"))
        if ai.get("root") != di.get("root"):
            buckets.append(("repair_cross_root", "review_repair"))
        if q.get("intent") in {"device_main", "device_accessory"}:
            buckets.append(("repair_device_domain", "lock_repair"))
        if q_a_overlap == 1 and q_d_overlap == 0:
            buckets.append(("repair_head_safe", "possible_safe"))

        if not buckets:
            buckets.append(("repair_other", "review_repair"))

        for bucket, priority in buckets:
            bucket_rows.append(
                {
                    "bucket": bucket,
                    "priority": priority,
                    "variant": row.variant,
                    "term_id": str(row.term_id),
                    "query": row.query,
                    "query_intent": q.get("intent"),
                    "query_device_family": q.get("device_family"),
                    "query_heads": "|".join(sorted(q_heads)),
                    "add_id": row.add_id,
                    "add_item_id": str(row.add_item_id),
                    "add_title": row.add_title,
                    "add_category": row.add_category,
                    "add_brand": row.add_brand,
                    "add_gender": row.add_gender,
                    "add_heads": "|".join(sorted(a_heads)),
                    "drop_id": row.drop_id,
                    "drop_item_id": str(row.drop_item_id),
                    "drop_title": row.drop_title,
                    "drop_category": row.drop_category,
                    "drop_brand": row.drop_brand,
                    "drop_gender": row.drop_gender,
                    "drop_heads": "|".join(sorted(d_heads)),
                    "q_a_head_overlap": q_a_overlap,
                    "q_d_head_overlap": q_d_overlap,
                    "add_root": ai.get("root"),
                    "drop_root": di.get("root"),
                    "add_v108_score": row.add_v108_score,
                    "drop_v108_score": row.drop_v108_score,
                    "add_v82_score": row.add_v82_score,
                    "drop_v82_score": row.drop_v82_score,
                    "min_margin": row.min_margin,
                    "comp_gap": row.comp_gap,
                }
            )
    return pd.DataFrame(bucket_rows)


def main():
    mod209 = load_mod209()
    v104_v107 = build_v104_v107_diff(mod209)
    v108_only = build_v108_only_removals(mod209)
    v109_buckets = build_v109_buckets(mod209)

    blocks = []
    summary_rows = []

    if not v104_v107.empty:
        sample = sample_bucket(v104_v107, sort_cols=["v108_score"])
        sample["source_family"] = "v104_v107"
        blocks.append(sample)
        summary_rows.append(
            {
                "bucket": "v104_only_vs_v107",
                "source_family": "v104_v107",
                "full_rows": int(len(v104_v107)),
                "sample_rows": int(len(sample)),
                "priority": "lock_old_broad_rule",
            }
        )

    if not v108_only.empty:
        sample = sample_bucket(v108_only, sort_cols=["v108_score"])
        sample["source_family"] = "v108"
        blocks.append(sample)
        summary_rows.append(
            {
                "bucket": "v108_only_removal",
                "source_family": "v108",
                "full_rows": int(len(v108_only)),
                "sample_rows": int(len(sample)),
                "priority": "verify_veto",
            }
        )

    if not v109_buckets.empty:
        for bucket, grp in v109_buckets.groupby("bucket", sort=True):
            sort_cols = ["min_margin", "comp_gap"] if "repair_" in bucket else None
            sample = sample_bucket(grp, sort_cols=sort_cols)
            sample["source_family"] = "v109"
            blocks.append(sample)
            summary_rows.append(
                {
                    "bucket": bucket,
                    "source_family": "v109",
                    "full_rows": int(len(grp)),
                    "sample_rows": int(len(sample)),
                    "priority": str(grp["priority"].mode().iloc[0]),
                }
            )

    audit = pd.concat(blocks, ignore_index=True, sort=False) if blocks else pd.DataFrame()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_CSV, index=False)

    summary = {
        "base_file": str(BASE),
        "v104_file": str(V104),
        "v107_file": str(V107),
        "v108_file": str(V108),
        "v109_review_file": str(V109_REVIEW),
        "audit_file": str(OUT_CSV),
        "buckets": summary_rows,
        "sha256": hashlib.sha256(OUT_CSV.read_bytes()).hexdigest() if OUT_CSV.exists() else "",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
