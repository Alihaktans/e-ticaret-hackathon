from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
CTX = runpy.run_path(str(ROOT / "scripts/145_make_v49_query_twin_transfer.py"))
SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"
OUT_REVIEW = OUT_DIR / "review_v51_diverse_consensus_swaps.csv"
OUT_SUMMARY = OUT_DIR / "v51_diverse_consensus_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v51_diverse_consensus_saved_candidates.csv"
OUT_FEATURES = ROOT / "data/processed/v51_diverse_consensus_features.parquet"

ANCHOR_PATHS = CTX["ANCHOR_PATHS"]
VOTER_SPECS = [
    ("v35_ce", "base", [
        ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    ]),
    ("v40_history", "base", [
        ROOT / "submissions/final_candidates_v40/A_v40_raw35_history_impact2200.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v40_rankstable_002_2ef0d106.csv",
    ]),
    ("v41_meta", "meta_graph", [
        ROOT / "submissions/FINAL_CANDIDATE_v41_meta_001_22655235.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v41_meta_001_459bc2fe.csv",
    ]),
    ("v42_graph", "meta_graph", [
        ROOT / "submissions/final_candidate_last/A_v42_qgraph_v41_balanced3600.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v42_qgraph_005_1d4dbf47.csv",
    ]),
    ("v43_popbc", "semantic", [
        ROOT / "submissions/FINAL_CANDIDATE_v43_popbc_001_b2acd38f.csv",
    ]),
    ("v46_consensus", "semantic", [
        ROOT / "submissions/FINAL_CANDIDATE_v46_consensus_003_33ebb61c.csv",
        ROOT / "submissions/FINAL_CANDIDATE_v46_consensus_001_46421930.csv",
    ]),
    ("v47_shadow", "structural", [
        ROOT / "submissions/FINAL_CANDIDATE_v47_shadow_001_380a2292.csv",
    ]),
    ("v48_memory", "structural", [
        ROOT / "submissions/FINAL_CANDIDATE_v48_memory_001_99baaf79.csv",
    ]),
    ("v49_twin", "structural", [
        ROOT / "submissions/FINAL_CANDIDATE_v49_twin_003_65248ed3.csv",
    ]),
]


def first_existing(paths):
    return next((p for p in paths if p.exists()), None)


def load_prediction(path: Path, sample: pd.DataFrame) -> np.ndarray:
    d = pd.read_csv(path, usecols=["id", "prediction"])
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def short_hash(value):
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8]


def build_vote_frame(sample, pairs, anchor):
    df = pairs.copy()
    df["anchor"] = anchor
    voters, blocks = [], {}
    for name, block, paths in VOTER_SPECS:
        path = first_existing(paths)
        if path is None:
            print("missing voter", name)
            continue
        pred = load_prediction(path, sample)
        df[name] = pred
        voters.append(name)
        blocks.setdefault(block, []).append(name)
        print("voter", name, "block", block, "path", path, "diff_anchor", int((pred != anchor).sum()))
    if len(voters) < 6:
        raise RuntimeError(f"too few voters: {len(voters)}")
    df["vote_ones"] = df[voters].sum(axis=1).astype(np.int8)
    df["vote_zeros"] = (len(voters) - df["vote_ones"]).astype(np.int8)
    for block, cols in blocks.items():
        df[f"{block}_one"] = df[cols].max(axis=1).astype(np.int8)
        df[f"{block}_zero"] = (df[cols].min(axis=1) == 0).astype(np.int8)
    one_cols = [f"{b}_one" for b in blocks]
    zero_cols = [f"{b}_zero" for b in blocks]
    df["one_block_count"] = df[one_cols].sum(axis=1).astype(np.int8)
    df["zero_block_count"] = df[zero_cols].sum(axis=1).astype(np.int8)
    # Reuse the model-support feature generated for V49 when IDs align.
    feature_path = ROOT / "data/processed/v49_query_twin_transfer_features.parquet"
    if feature_path.exists():
        support = pd.read_parquet(feature_path, columns=["id", "support"])
        support["id"] = support["id"].astype(str)
        if support["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
            df["support"] = support["support"].astype(np.float32).to_numpy()
        else:
            df = df.merge(support, on="id", how="left", validate="one_to_one")
    if "support" not in df:
        df["support"] = np.float32(.5)
    df["support"] = df["support"].fillna(.5).astype(np.float32)
    keep = [
        "id", "term_id", "item_id", "anchor", "support", "vote_ones", "vote_zeros",
        "one_block_count", "zero_block_count",
    ] + voters + [c for c in df.columns if c.endswith("_one") or c.endswith("_zero")]
    df[keep].to_parquet(OUT_FEATURES, index=False)
    return df, voters, blocks


def make_swaps(df, n_voters, mode):
    if mode == "ultra":
        min_votes, min_blocks, min_gain = n_voters - 1, 4, .04
    elif mode == "safe":
        min_votes, min_blocks, min_gain = n_voters - 2, 3, .02
    elif mode == "balanced":
        min_votes, min_blocks, min_gain = n_voters - 3, 3, .00
    else:
        raise ValueError(mode)

    adds = df[
        (df["anchor"] == 0)
        & (df["vote_ones"] >= min_votes)
        & (df["one_block_count"] >= min_blocks)
    ][[
        "term_id", "id", "item_id", "support", "vote_ones", "one_block_count",
    ]].rename(columns={
        "term_id": "dst_term_id", "id": "id_add", "support": "add_support",
        "vote_ones": "add_votes", "one_block_count": "add_blocks",
    })
    drops = df[
        (df["anchor"] == 1)
        & (df["vote_zeros"] >= min_votes)
        & (df["zero_block_count"] >= min_blocks)
    ][[
        "term_id", "id", "item_id", "support", "vote_zeros", "zero_block_count",
    ]].rename(columns={
        "term_id": "dst_term_id", "id": "id_drop", "item_id": "item_id_drop",
        "support": "drop_support", "vote_zeros": "drop_votes", "zero_block_count": "drop_blocks",
    })
    print(mode, "raw adds", len(adds), "raw drops", len(drops))
    adds = adds.sort_values(
        ["dst_term_id", "add_votes", "add_blocks", "add_support"], ascending=[True, False, False, False]
    )
    drops = drops.sort_values(
        ["dst_term_id", "drop_votes", "drop_blocks", "drop_support"], ascending=[True, False, False, True]
    )
    adds["pair_rank"] = adds.groupby("dst_term_id").cumcount()
    drops["pair_rank"] = drops.groupby("dst_term_id").cumcount()
    sw = adds.merge(drops, on=["dst_term_id", "pair_rank"], how="inner")
    sw["support_gain"] = sw["add_support"] - sw["drop_support"]
    sw = sw[sw["support_gain"] >= min_gain].copy()
    sw["v51_swap_score"] = (
        .75 * sw["add_votes"] + .40 * sw["add_blocks"] + .75 * sw["drop_votes"]
        + .40 * sw["drop_blocks"] + 1.50 * sw["support_gain"]
    )
    sw["mode"] = mode
    return sw.sort_values("v51_swap_score", ascending=False).reset_index(drop=True)


def apply_swaps(sample, anchor, swaps, cap):
    take = swaps.head(cap).copy()
    pred = anchor.copy()
    index = pd.Series(np.arange(len(sample)), index=sample["id"])
    ai, di = take["id_add"].map(index), take["id_drop"].map(index)
    if ai.isna().any() or di.isna().any():
        raise RuntimeError("swap mapping failed")
    pred[ai.astype(int)] = 1
    pred[di.astype(int)] = 0
    return pred, take


def enrich_review(swaps, vote_df, voters):
    if swaps.empty:
        return swaps
    terms = CTX["load_terms"]()[["term_id", "query"]].rename(
        columns={"term_id": "dst_term_id", "query": "dst_query"}
    )
    items = CTX["load_item_meta"]()
    add = items.rename(columns={"title": "add_title", "brand": "add_brand", "category": "add_category"})
    drop = items.rename(columns={
        "item_id": "item_id_drop", "title": "drop_title", "brand": "drop_brand", "category": "drop_category"
    })
    add_votes = vote_df[["id"] + voters].rename(
        columns={"id": "id_add", **{x: f"add_{x}" for x in voters}}
    )
    drop_votes = vote_df[["id"] + voters].rename(
        columns={"id": "id_drop", **{x: f"drop_{x}" for x in voters}}
    )
    out = swaps.merge(terms, on="dst_term_id", how="left")
    out = out.merge(add, on="item_id", how="left").merge(drop, on="item_id_drop", how="left")
    out = out.merge(add_votes, on="id_add", how="left").merge(drop_votes, on="id_drop", how="left")
    front = [
        "mode", "dst_query", "id_add", "item_id", "add_title", "add_brand", "add_category",
        "id_drop", "item_id_drop", "drop_title", "drop_brand", "drop_category",
        "add_votes", "drop_votes", "add_blocks", "drop_blocks", "add_support", "drop_support",
        "support_gain", "v51_swap_score",
    ]
    return out[front + [c for c in out.columns if c not in front]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    for col in ["id", "term_id", "item_id"]:
        pairs[col] = pairs[col].astype(str)
    if not sample["id"].reset_index(drop=True).equals(pairs["id"].reset_index(drop=True)):
        raise RuntimeError("sample/pairs id mismatch")
    anchor_path = first_existing(ANCHOR_PATHS)
    anchor = load_prediction(anchor_path, sample)
    vote_df, voters, blocks = build_vote_frame(sample, pairs, anchor)
    print("active voters", voters, "blocks", blocks)

    summaries, saved, reviews, variants = [], [], [], {}
    caps = {"ultra": [100, 250, 500, 1000], "safe": [250, 500, 1000, 2000], "balanced": [500, 1000, 2000, 3500]}
    for mode, mode_caps in caps.items():
        swaps = make_swaps(vote_df, len(voters), mode)
        print(mode, "accepted_pool", len(swaps))
        if not swaps.empty:
            reviews.append(swaps.head(500))
        for cap in mode_caps:
            if swaps.empty:
                continue
            pred, take = apply_swaps(sample, anchor, swaps, min(cap, len(swaps)))
            variant = f"v51_diverse_{mode}_cap{cap}"
            variants[variant] = pred
            summaries.append({
                "variant": variant, "source": "four_block_diverse_vote", "mode": mode,
                "cap": cap, "used_swaps": int(len(take)), "accepted_pool": int(len(swaps)),
                "add_votes_mean": float(take["add_votes"].mean()),
                "drop_votes_mean": float(take["drop_votes"].mean()),
                "add_blocks_mean": float(take["add_blocks"].mean()),
                "drop_blocks_mean": float(take["drop_blocks"].mean()),
                "support_gain_mean": float(take["support_gain"].mean()),
                "v51_swap_score_mean": float(take["v51_swap_score"].mean()),
                "ones": int(pred.sum()), "pos_ratio": float(pred.mean()),
                "diff_vs_anchor": int((pred != anchor).sum()),
            })

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["v51_decision_score"] = (
            .004 * summary["add_votes_mean"] + .004 * summary["drop_votes_mean"]
            + .005 * summary["add_blocks_mean"] + .005 * summary["drop_blocks_mean"]
            + .010 * summary["support_gain_mean"]
            - .000001 * np.maximum(0, summary["diff_vs_anchor"] - 3000)
        )
        summary = summary.sort_values("v51_decision_score", ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if reviews:
        review = pd.concat(reviews, ignore_index=True).drop_duplicates(["id_add", "id_drop"])
        enrich_review(review, vote_df, voters).to_csv(OUT_REVIEW, index=False)
    else:
        pd.DataFrame().to_csv(OUT_REVIEW, index=False)

    for rank, row in summary.head(12).reset_index(drop=True).iterrows():
        variant = row["variant"]
        path = SUB_DIR / f"FINAL_CANDIDATE_v51_diverse_{rank+1:03d}_{short_hash(variant)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": variants[variant].astype(np.int8)}).to_csv(path, index=False)
        saved.append({**row.to_dict(), "file": str(path)})
        print("saved", path)
    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)
    print("\nTOP V51")
    print(summary.head(30).to_string(index=False) if len(summary) else "no candidates")
    print("outputs", OUT_FEATURES, OUT_SUMMARY, OUT_REVIEW, OUT_SAVED, sep="\n")


if __name__ == "__main__":
    main()
