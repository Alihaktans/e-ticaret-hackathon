from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
CTX = runpy.run_path(str(ROOT / "scripts/145_make_v49_query_twin_transfer.py"))

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions"
OUT_FEATURES = ROOT / "data/processed/v50_query_family_features.parquet"
OUT_REVIEW = OUT_DIR / "review_v50_query_family_swaps.csv"
OUT_SUMMARY = OUT_DIR / "v50_query_family_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v50_query_family_saved_candidates.csv"


def short_hash(x: object) -> str:
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


class UnionFind:
    def __init__(self, values):
        self.parent = {x: x for x in values}
        self.size = {x: 1 for x in values}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def family_map(term_ids: pd.Series, edges: pd.DataFrame, threshold: float) -> pd.DataFrame:
    values = term_ids.astype(str).unique().tolist()
    uf = UnionFind(values)
    use = edges[edges["query_sim"] >= threshold]
    for row in use[["src_term_id", "dst_term_id"]].itertuples(index=False):
        if row.src_term_id in uf.parent and row.dst_term_id in uf.parent:
            uf.union(row.src_term_id, row.dst_term_id)
    roots = {x: uf.find(x) for x in values}
    root_values = sorted(set(roots.values()))
    root_id = {x: i for i, x in enumerate(root_values)}
    counts = pd.Series(list(roots.values())).value_counts().to_dict()
    return pd.DataFrame(
        {
            "term_id": values,
            "family_id": [root_id[roots[x]] for x in values],
            "family_size": [counts[roots[x]] for x in values],
        }
    )


def undirected_edges(qpairs: pd.DataFrame) -> pd.DataFrame:
    base = qpairs[[
        "src_term_id", "dst_term_id", "query_sim", "src_query", "dst_query"
    ]].copy()
    rev = base.rename(
        columns={
            "src_term_id": "dst_term_id",
            "dst_term_id": "src_term_id",
            "src_query": "dst_query",
            "dst_query": "src_query",
        }
    )
    edges = pd.concat([base, rev], ignore_index=True)
    edges = edges.sort_values("query_sim", ascending=False).drop_duplicates(
        ["src_term_id", "dst_term_id"], keep="first"
    )
    return edges[edges["src_term_id"] != edges["dst_term_id"]].reset_index(drop=True)


def build_family_candidates(
    df: pd.DataFrame,
    terms: pd.DataFrame,
    edges: pd.DataFrame,
    edge_threshold: float,
) -> pd.DataFrame:
    use_edges = edges[edges["query_sim"] >= edge_threshold].copy()
    fmap = family_map(terms["term_id"], use_edges, edge_threshold)
    query_map = terms.set_index("term_id")["query"].astype(str).to_dict()

    positives = df[df["anchor"].eq(1)][[
        "term_id", "item_id", "support"
    ]].rename(
        columns={"term_id": "src_term_id", "support": "src_support"}
    )
    destinations = df[[
        "term_id", "item_id", "id", "anchor", "support",
        "v33", "v34", "v38", "v41", "v42", "v43",
    ]].rename(
        columns={
            "term_id": "dst_term_id", "id": "id_add", "anchor": "add_anchor",
            "support": "add_support", "v33": "add_v33", "v34": "add_v34",
            "v38": "add_v38", "v41": "add_v41", "v42": "add_v42", "v43": "add_v43",
        }
    )

    evidence = use_edges.merge(positives, on="src_term_id", how="inner")
    evidence = evidence.merge(destinations, on=["dst_term_id", "item_id"], how="inner")
    evidence = evidence[evidence["add_anchor"].eq(0)].copy()
    if evidence.empty:
        return evidence

    evidence["src_query_text"] = evidence["src_term_id"].map(query_map).fillna("")
    grouped = evidence.groupby(
        ["dst_term_id", "item_id", "id_add"], as_index=False
    ).agg(
        sibling_positive_count=("src_term_id", "nunique"),
        sibling_queries=("src_query_text", lambda x: " | ".join(sorted(set(map(str, x)))[:12])),
        source_term_ids=("src_term_id", lambda x: " | ".join(sorted(set(map(str, x)))[:12])),
        query_sim_min=("query_sim", "min"),
        query_sim_mean=("query_sim", "mean"),
        query_sim_max=("query_sim", "max"),
        src_support_mean=("src_support", "mean"),
        src_support_max=("src_support", "max"),
        add_support=("add_support", "first"),
        add_v33=("add_v33", "first"),
        add_v34=("add_v34", "first"),
        add_v38=("add_v38", "first"),
        add_v41=("add_v41", "first"),
        add_v42=("add_v42", "first"),
        add_v43=("add_v43", "first"),
    )
    grouped = grouped.merge(
        fmap.rename(columns={"term_id": "dst_term_id"}), on="dst_term_id", how="left"
    )
    grouped["dst_query"] = grouped["dst_term_id"].map(query_map).fillna("")
    grouped["edge_threshold"] = edge_threshold
    return grouped


def make_swaps(df: pd.DataFrame, candidates: pd.DataFrame, mode: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    if mode == "ultra":
        c = candidates[
            (candidates["sibling_positive_count"] >= 2)
            & (candidates["family_size"] >= 3)
            & (candidates["query_sim_min"] >= .95)
            & (candidates["add_support"] >= .58)
            & (candidates["src_support_mean"] >= .58)
        ].copy()
        min_gain, min_wins = .07, 4
    elif mode == "safe":
        c = candidates[
            (candidates["sibling_positive_count"] >= 2)
            & (candidates["family_size"] >= 3)
            & (candidates["query_sim_mean"] >= .93)
            & (candidates["query_sim_min"] >= .90)
            & (candidates["add_support"] >= .54)
            & (candidates["src_support_mean"] >= .55)
        ].copy()
        min_gain, min_wins = .05, 3
    elif mode == "balanced":
        c = candidates[
            (candidates["sibling_positive_count"] >= 2)
            & (candidates["family_size"] >= 3)
            & (candidates["query_sim_mean"] >= .90)
            & (candidates["query_sim_min"] >= .88)
            & (candidates["add_support"] >= .50)
            & (candidates["src_support_mean"] >= .52)
        ].copy()
        min_gain, min_wins = .03, 3
    else:
        raise ValueError(mode)
    if c.empty:
        return c

    drops = df[df["anchor"].eq(1)][[
        "term_id", "id", "item_id", "support", "v33", "v34", "v38", "v41", "v42", "v43",
    ]].rename(
        columns={
            "term_id": "dst_term_id", "id": "id_drop", "item_id": "item_id_drop",
            "support": "drop_support", "v33": "drop_v33", "v34": "drop_v34",
            "v38": "drop_v38", "v41": "drop_v41", "v42": "drop_v42", "v43": "drop_v43",
        }
    )
    c = c.sort_values(
        ["dst_term_id", "sibling_positive_count", "add_support"], ascending=[True, False, False]
    )
    drops = drops.sort_values(["dst_term_id", "drop_support"], ascending=[True, True])
    c["pair_rank"] = c.groupby("dst_term_id").cumcount()
    drops["pair_rank"] = drops.groupby("dst_term_id").cumcount()
    sw = c.merge(drops, on=["dst_term_id", "pair_rank"], how="inner")
    if sw.empty:
        return sw
    sw["support_gain"] = sw["add_support"] - sw["drop_support"]
    sw["model_win_count"] = sum(
        (sw[f"add_{name}"] > sw[f"drop_{name}"]).astype(np.int8)
        for name in ["v33", "v34", "v38", "v41", "v43"]
    )
    sw = sw[(sw["support_gain"] >= min_gain) & (sw["model_win_count"] >= min_wins)].copy()
    sw["v50_swap_score"] = (
        1.10 * np.log1p(sw["sibling_positive_count"])
        + .65 * sw["query_sim_mean"]
        + .25 * sw["query_sim_min"]
        + .80 * sw["add_support"]
        + 1.35 * sw["support_gain"]
        + .20 * sw["model_win_count"]
    )
    sw["mode"] = mode
    return sw.sort_values("v50_swap_score", ascending=False).reset_index(drop=True)


def apply_swaps(sample, anchor, swaps, cap):
    take = swaps.head(cap).copy()
    pred = anchor.copy()
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    add_idx = take["id_add"].map(id_to_idx)
    drop_idx = take["id_drop"].map(id_to_idx)
    if add_idx.isna().any() or drop_idx.isna().any():
        raise RuntimeError("swap id mapping failed")
    pred[add_idx.astype(int)] = 1
    pred[drop_idx.astype(int)] = 0
    return pred, take


def enrich_review(swaps):
    if swaps.empty:
        return swaps
    items = CTX["load_item_meta"]()
    terms = CTX["load_terms"]()[["term_id", "query"]].rename(
        columns={"term_id": "dst_term_id", "query": "dst_query_real"}
    )
    add = items.rename(
        columns={"title": "add_title", "brand": "add_brand", "category": "add_category"}
    )
    drop = items.rename(
        columns={
            "item_id": "item_id_drop", "title": "drop_title",
            "brand": "drop_brand", "category": "drop_category",
        }
    )
    out = swaps.merge(terms, on="dst_term_id", how="left")
    out = out.merge(add, on="item_id", how="left").merge(drop, on="item_id_drop", how="left")
    front = [
        "mode", "family_id", "family_size", "dst_term_id", "dst_query_real",
        "sibling_queries", "query_sim_min", "query_sim_mean", "query_sim_max",
        "sibling_positive_count", "id_add", "item_id", "add_title", "add_brand", "add_category",
        "id_drop", "item_id_drop", "drop_title", "drop_brand", "drop_category",
        "add_support", "drop_support", "support_gain", "model_win_count", "v50_swap_score",
    ]
    return out[front + [c for c in out.columns if c not in front]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(CTX["SAMPLE"], usecols=["id"])
    sample["id"] = sample["id"].astype(str)
    pairs = pd.read_csv(CTX["PAIRS"], usecols=["id", "term_id", "item_id"])
    for col in ["id", "term_id", "item_id"]:
        pairs[col] = pairs[col].astype(str)
    if not sample["id"].reset_index(drop=True).equals(pairs["id"].reset_index(drop=True)):
        raise RuntimeError("sample/pairs id order mismatch")
    anchor_path = CTX["first_existing"](CTX["ANCHOR_PATHS"])
    anchor = CTX["load_pred"](anchor_path, sample)
    terms = CTX["load_terms"]()
    df = CTX["build_features"](sample, pairs, anchor)
    qpairs = CTX["build_query_twin_pairs"](terms, min_sim=.88, topn=6)
    edges = undirected_edges(qpairs)

    candidate_parts = []
    for threshold in [.97, .94, .91, .88]:
        part = build_family_candidates(df, terms, edges, threshold)
        print("threshold", threshold, "family add candidates", len(part))
        if not part.empty:
            candidate_parts.append(part)
    candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["sibling_positive_count", "query_sim_mean", "add_support"], ascending=False
        ).drop_duplicates(["dst_term_id", "item_id"], keep="first")
        candidates.to_parquet(OUT_FEATURES, index=False)
    else:
        pd.DataFrame().to_parquet(OUT_FEATURES, index=False)
    print("unique family candidates", len(candidates))

    baselines = {"anchor": anchor}
    for name, paths in CTX["BASELINE_CSV_CANDIDATES"].items():
        path = CTX["first_existing"](paths)
        if path is not None:
            baselines[name] = CTX["load_pred"](path, sample)

    summaries, reviews, saved = [], [], []
    variants = {}
    mode_caps = {"ultra": [50, 100, 200, 500], "safe": [100, 250, 500, 1000], "balanced": [250, 500, 1000, 2000]}
    for mode, caps in mode_caps.items():
        sw = make_swaps(df, candidates, mode)
        print(mode, "accepted_pool", len(sw))
        if not sw.empty:
            reviews.append(sw.head(500))
        for cap in caps:
            if sw.empty:
                continue
            pred, take = apply_swaps(sample, anchor, sw, min(cap, len(sw)))
            variant = f"v50_family_{mode}_cap{cap}"
            variants[variant] = pred
            summaries.append(
                {
                    "variant": variant, "source": "direct_query_family_consensus", "mode": mode,
                    "cap": cap, "used_swaps": int(len(take)), "accepted_pool": int(len(sw)),
                    "family_size_mean": float(take["family_size"].mean()),
                    "sibling_positive_count_mean": float(take["sibling_positive_count"].mean()),
                    "query_sim_min_mean": float(take["query_sim_min"].mean()),
                    "query_sim_mean": float(take["query_sim_mean"].mean()),
                    "support_gain_mean": float(take["support_gain"].mean()),
                    "add_support_mean": float(take["add_support"].mean()),
                    "drop_support_mean": float(take["drop_support"].mean()),
                    "model_win_mean": float(take["model_win_count"].mean()),
                    "v50_swap_score_mean": float(take["v50_swap_score"].mean()),
                }
            )

    summary = pd.DataFrame(summaries)
    for i, row in summary.iterrows():
        pred = variants[row["variant"]]
        summary.loc[i, "ones"] = int(pred.sum())
        summary.loc[i, "pos_ratio"] = float(pred.mean())
        summary.loc[i, "diff_vs_anchor"] = int((pred != anchor).sum())
        for bname in ["raw_v35_b5000", "v41_meta_same_quota", "v42_qgraph_v41_balanced3600"]:
            summary.loc[i, f"diff_vs_{bname}"] = int((pred != baselines[bname]).sum()) if bname in baselines else -1
        summary.loc[i, "v50_decision_score"] = (
            .010 * row["query_sim_mean"]
            + .008 * row["support_gain_mean"]
            + .004 * row["model_win_mean"]
            + .004 * np.log1p(row["sibling_positive_count_mean"])
            - .000001 * max(0, summary.loc[i, "diff_vs_anchor"] - 3000)
        )
    if not summary.empty:
        summary = summary.sort_values("v50_decision_score", ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if reviews:
        review = pd.concat(reviews, ignore_index=True).drop_duplicates(["id_add", "id_drop"])
        enrich_review(review).to_csv(OUT_REVIEW, index=False)
    else:
        pd.DataFrame().to_csv(OUT_REVIEW, index=False)

    for rank, row in summary.head(12).reset_index(drop=True).iterrows():
        variant = row["variant"]
        pred = variants[variant]
        filename = f"FINAL_CANDIDATE_v50_family_{rank+1:03d}_{short_hash(variant)}.csv"
        path = SUB_DIR / filename
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(path, index=False)
        saved.append({**row.to_dict(), "file": str(path)})
        print("saved", path, "used_swaps", int(row["used_swaps"]))
    pd.DataFrame(saved).to_csv(OUT_SAVED, index=False)
    print("\nTOP V50")
    print(summary.head(30).to_string(index=False) if len(summary) else "no candidates")
    print("outputs", OUT_FEATURES, OUT_SUMMARY, OUT_REVIEW, OUT_SAVED, sep="\n")


if __name__ == "__main__":
    main()
