from __future__ import annotations

import hashlib
import re
import runpy
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
CTX = runpy.run_path(str(ROOT / "scripts/145_make_v49_query_twin_transfer.py"))
SAMPLE, PAIRS, ITEMS = ROOT / "data/raw/sample_submission.csv", ROOT / "data/raw/submission_pairs.csv", ROOT / "data/raw/items.csv"
ANCHOR_PATHS = CTX["ANCHOR_PATHS"]
VOTE_FEATURES = ROOT / "data/processed/v51_diverse_consensus_features.parquet"
OUT_FEATURES = ROOT / "data/processed/v52_item_twin_features.parquet"
OUT_REVIEW = ROOT / "reports/manual_review/review_v52_item_twin_swaps.csv"
OUT_SUMMARY = ROOT / "reports/manual_review/v52_item_twin_candidate_summary.csv"
OUT_SAVED = ROOT / "reports/manual_review/v52_item_twin_saved_candidates.csv"
SUB_DIR = ROOT / "submissions"

TR = str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c", "İ": "i", "Ş": "s", "Ğ": "g", "Ü": "u", "Ö": "o", "Ç": "c"})


def norm(value):
    text = unicodedata.normalize("NFKD", str(value).lower().translate(TR))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


def load_pred(path, sample):
    d = pd.read_csv(path, usecols=["id", "prediction"])
    d["id"] = d["id"].astype(str)
    if not d.id.reset_index(drop=True).equals(sample.id.reset_index(drop=True)):
        raise RuntimeError(f"id mismatch: {path}")
    return d.prediction.astype(np.int8).to_numpy()


def build_candidates(frame):
    frame["title_norm"] = frame.title.fillna("").map(norm)
    frame["brand_norm"] = frame.brand.fillna("").map(norm)
    frame["cat_norm"] = frame.category.fillna("").map(norm)
    frame["token_sig"] = frame.title_norm.map(lambda x: " ".join(sorted(set(x.split()))))
    parts = []
    for twin_type, signature in [("exact_title", "title_norm"), ("token_signature", "token_sig")]:
        keys = ["term_id", "brand_norm", "cat_norm", signature]
        stats = frame.groupby(keys, as_index=False).anchor.agg(twin_group_size="size", twin_positive_count="sum")
        mixed = stats[
            (stats.twin_group_size >= 2)
            & (stats.twin_positive_count >= 1)
            & (stats.twin_positive_count < stats.twin_group_size)
        ]
        cand = frame.merge(mixed, on=keys, how="inner")
        cand = cand[cand.anchor.eq(0)].copy()
        cand["twin_type"] = twin_type
        cand["twin_priority"] = 2 if twin_type == "exact_title" else 1
        parts.append(cand)
    candidates = pd.concat(parts, ignore_index=True)
    candidates = candidates.sort_values(
        ["twin_priority", "twin_positive_count", "support", "vote_ones"], ascending=False
    ).drop_duplicates("id", keep="first")
    candidates = candidates.rename(columns={
        "id": "id_add", "support": "add_support", "vote_ones": "add_votes",
        "one_block_count": "add_blocks",
    })
    return candidates


def make_swaps(frame, candidates, mode):
    if mode == "exact":
        c = candidates[
            candidates.twin_type.eq("exact_title")
            & (candidates.add_support >= .35)
            & (candidates.add_votes >= 4)
            & (candidates.add_blocks >= 2)
        ].copy()
        min_gain, min_vote_gain = -.03, 0
    elif mode == "safe":
        c = candidates[
            (candidates.add_support >= .40)
            & (candidates.add_votes >= 5)
            & (candidates.add_blocks >= 3)
        ].copy()
        min_gain, min_vote_gain = 0, 1
    elif mode == "balanced":
        c = candidates[
            (candidates.add_support >= .35)
            & (candidates.add_votes >= 4)
            & (candidates.add_blocks >= 2)
        ].copy()
        min_gain, min_vote_gain = -.02, 0
    else:
        raise ValueError(mode)
    if c.empty:
        return c
    drops = frame[frame.anchor.eq(1)][[
        "term_id", "id", "item_id", "support", "vote_ones", "one_block_count",
    ]].rename(columns={
        "id": "id_drop", "item_id": "item_id_drop", "support": "drop_support",
        "vote_ones": "drop_one_votes", "one_block_count": "drop_one_blocks",
    })
    c = c.sort_values(
        ["term_id", "twin_priority", "twin_positive_count", "add_votes", "add_support"],
        ascending=[True, False, False, False, False],
    )
    drops = drops.sort_values(["term_id", "drop_one_votes", "drop_support"], ascending=[True, True, True])
    c["pair_rank"] = c.groupby("term_id").cumcount()
    drops["pair_rank"] = drops.groupby("term_id").cumcount()
    sw = c.merge(drops, on=["term_id", "pair_rank"], how="inner")
    sw["support_gain"] = sw.add_support - sw.drop_support
    sw["vote_gain"] = sw.add_votes - sw.drop_one_votes
    sw = sw[(sw.support_gain >= min_gain) & (sw.vote_gain >= min_vote_gain)].copy()
    sw["v52_swap_score"] = (
        .85 * sw.twin_priority + .50 * np.log1p(sw.twin_positive_count)
        + .55 * sw.add_votes + .30 * sw.add_blocks + 1.25 * sw.support_gain + .30 * sw.vote_gain
    )
    sw["mode"] = mode
    return sw.sort_values("v52_swap_score", ascending=False).reset_index(drop=True)


def apply_swaps(sample, anchor, swaps, cap):
    take = swaps.head(cap).copy()
    pred = anchor.copy()
    idx = pd.Series(np.arange(len(sample)), index=sample.id)
    ai, di = take.id_add.map(idx), take.id_drop.map(idx)
    if ai.isna().any() or di.isna().any():
        raise RuntimeError("swap map failed")
    pred[ai.astype(int)] = 1
    pred[di.astype(int)] = 0
    return pred, take


def main():
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE, usecols=["id"]); sample.id = sample.id.astype(str)
    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    for c in ["id", "term_id", "item_id"]: pairs[c] = pairs[c].astype(str)
    if not sample.id.reset_index(drop=True).equals(pairs.id.reset_index(drop=True)):
        raise RuntimeError("sample/pairs mismatch")
    anchor_path = next(p for p in ANCHOR_PATHS if p.exists())
    anchor = load_pred(anchor_path, sample)
    votes = pd.read_parquet(VOTE_FEATURES)
    votes.id = votes.id.astype(str)
    if not votes.id.reset_index(drop=True).equals(sample.id.reset_index(drop=True)):
        raise RuntimeError("vote feature order mismatch")
    frame = pairs.copy()
    for c in ["anchor", "support", "vote_ones", "one_block_count"]:
        frame[c] = votes[c].to_numpy()
    items = pd.read_csv(ITEMS, usecols=["item_id", "title", "brand", "category"])
    items.item_id = items.item_id.astype(str)
    frame = frame.merge(items, on="item_id", how="left", validate="many_to_one")
    candidates = build_candidates(frame)
    candidates.to_parquet(OUT_FEATURES, index=False)
    print("item twin candidates", len(candidates), candidates.twin_type.value_counts().to_dict())

    summaries, reviews, variants = [], [], {}
    for mode in ["exact", "safe", "balanced"]:
        swaps = make_swaps(frame, candidates, mode)
        print(mode, "accepted_pool", len(swaps))
        if not swaps.empty: reviews.append(swaps.head(400))
        for cap in [50, 100, 200, 500]:
            if swaps.empty: continue
            pred, take = apply_swaps(sample, anchor, swaps, min(cap, len(swaps)))
            name = f"v52_item_twin_{mode}_cap{cap}"; variants[name] = pred
            summaries.append({
                "variant": name, "mode": mode, "cap": cap, "used_swaps": len(take),
                "accepted_pool": len(swaps), "exact_rate": float(take.twin_type.eq('exact_title').mean()),
                "twin_positive_mean": float(take.twin_positive_count.mean()),
                "add_votes_mean": float(take.add_votes.mean()), "support_gain_mean": float(take.support_gain.mean()),
                "vote_gain_mean": float(take.vote_gain.mean()), "ones": int(pred.sum()),
                "pos_ratio": float(pred.mean()), "diff_vs_anchor": int((pred != anchor).sum()),
                "v52_decision_score": float(take.v52_swap_score.mean()),
            })
    summary = pd.DataFrame(summaries).sort_values(
        ["exact_rate", "v52_decision_score"], ascending=False
    ) if summaries else pd.DataFrame()
    summary.to_csv(OUT_SUMMARY, index=False)

    if reviews:
        rev = pd.concat(reviews).drop_duplicates(["id_add", "id_drop"])
        terms = CTX["load_terms"]()[["term_id", "query"]]
        drop_meta = items.rename(columns={
            "item_id": "item_id_drop", "title": "drop_title", "brand": "drop_brand", "category": "drop_category"
        })
        rev = rev.merge(terms, on="term_id", how="left").merge(drop_meta, on="item_id_drop", how="left")
        front = [
            "mode", "query", "twin_type", "twin_group_size", "twin_positive_count",
            "title", "brand", "category", "drop_title", "drop_brand", "drop_category",
            "add_votes", "drop_one_votes", "add_blocks", "support_gain", "vote_gain", "v52_swap_score",
        ]
        rev[front + [c for c in rev.columns if c not in front]].to_csv(OUT_REVIEW, index=False)
    else:
        pd.DataFrame().to_csv(OUT_REVIEW, index=False)

    saved = []
    for rank, row in summary.head(10).reset_index(drop=True).iterrows():
        name = row.variant
        path = SUB_DIR / f"FINAL_CANDIDATE_v52_itemtwin_{rank+1:03d}_{short_hash(name)}.csv"
        pd.DataFrame({"id": sample.id, "prediction": variants[name].astype(np.int8)}).to_csv(path, index=False)
        saved.append({**row.to_dict(), "file": str(path)})
        print("saved", path)
    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)
    print("\nTOP V52")
    print(summary.head(20).to_string(index=False) if len(summary) else "no candidates")
    print("outputs", OUT_FEATURES, OUT_SUMMARY, OUT_REVIEW, OUT_SAVED, sep="\n")


if __name__ == "__main__":
    main()
