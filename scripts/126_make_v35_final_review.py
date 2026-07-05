from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")

SAMPLE = ROOT / "data/raw/sample_submission.csv"
PAIRS = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

V33 = ROOT / "data/processed/v33_ft_pair_scores.parquet"
V34 = ROOT / "data/processed/v34/v34_cross_encoder_pair_scores.parquet"

ANCHOR_CANDIDATES = [
    ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv",
]

CAND_A_PATTERNS = [
    "FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    "FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv",
    "FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv",
]

CAND_B_PATTERNS = [
    "FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b10000.csv",
    "FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b10000.csv",
    "FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b10000.csv",
]

OUT_DIR = ROOT / "reports/manual_review"
OUT_B5000 = OUT_DIR / "review_v35_b5000_stratified.csv"
OUT_B10000_EXTRA = OUT_DIR / "review_v35_b10000_extra_stratified.csv"
OUT_SUMMARY = OUT_DIR / "review_v35_final_review_summary.csv"


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def first_existing_name(names):
    for name in names:
        p = ROOT / "submissions" / name
        if p.exists():
            return p
    return None


def clean_text(x, lim=300):
    if pd.isna(x):
        return ""
    x = str(x).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = " ".join(x.split())
    return x[:lim]


def sigmoid(x):
    x = np.clip(x, -12, 12)
    return 1.0 / (1.0 + np.exp(-x))


def load_pred(path, sample):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str)
    if not d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError(f"id order mismatch: {path}")
    return d["prediction"].astype(np.int8).to_numpy()


def align_score(score_path, sample, cols):
    d = pd.read_parquet(score_path)
    d["id"] = d["id"].astype(str)
    if d["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        return d[["id"] + cols].copy()
    return sample[["id"]].merge(d[["id"] + cols], on="id", how="left", validate="one_to_one")


def make_pair_table(anchor_pred, cand_pred, name, sample, pairs, terms, items, v33, v34):
    changed_add = (anchor_pred == 0) & (cand_pred == 1)
    changed_drop = (anchor_pred == 1) & (cand_pred == 0)

    add = pairs.loc[changed_add, ["id", "term_id", "item_id"]].copy()
    drop = pairs.loc[changed_drop, ["id", "term_id", "item_id"]].copy()

    add = add.merge(v33, on="id", how="left").merge(v34, on="id", how="left")
    drop = drop.merge(v33, on="id", how="left").merge(v34, on="id", how="left")

    for d in [add, drop]:
        d["v33_rs"] = (1.0 - d["v33_ft_pct_rank"]).clip(0, 1)
        d["v34_rs"] = (1.0 - d["v34_ce_pct_rank"]).clip(0, 1)
        d["v33_zsig"] = sigmoid(d["v33_ft_term_z"].fillna(0).to_numpy(np.float32) / 1.8)
        d["v34_zsig"] = sigmoid(d["v34_ce_term_z"].fillna(0).to_numpy(np.float32) / 1.8)
        d["score"] = (
            0.42 * d["v34_rs"].to_numpy(np.float32) +
            0.33 * d["v33_rs"].to_numpy(np.float32) +
            0.11 * d["v34_zsig"].to_numpy(np.float32) +
            0.08 * d["v33_zsig"].to_numpy(np.float32)
        ).astype(np.float32)

    add = add.sort_values(["term_id", "score"], ascending=[True, False])
    drop = drop.sort_values(["term_id", "score"], ascending=[True, True])

    add["pair_rank"] = add.groupby("term_id").cumcount()
    drop["pair_rank"] = drop.groupby("term_id").cumcount()

    paired = add.merge(
        drop,
        on=["term_id", "pair_rank"],
        suffixes=("_add", "_drop"),
        how="inner",
    )

    paired["score_gain"] = paired["score_add"] - paired["score_drop"]
    paired = paired.sort_values("score_gain", ascending=False).reset_index(drop=True)
    paired["candidate_name"] = name
    paired["global_swap_rank"] = np.arange(1, len(paired) + 1)

    # Text merge
    terms_small = terms[["term_id", "query"]].copy()
    item_cols = ["item_id", "title", "category", "brand", "gender", "age_group"]
    for c in item_cols:
        if c not in items.columns:
            items[c] = ""
    items_small = items[item_cols].copy()

    paired = paired.merge(terms_small, on="term_id", how="left", validate="many_to_one")
    paired = paired.merge(items_small.add_suffix("_add"), left_on="item_id_add", right_on="item_id_add", how="left")
    paired = paired.merge(items_small.add_suffix("_drop"), left_on="item_id_drop", right_on="item_id_drop", how="left")

    # Compact review columns
    out = pd.DataFrame({
        "candidate_name": paired["candidate_name"],
        "review_bucket": "",
        "global_swap_rank": paired["global_swap_rank"],
        "term_id": paired["term_id"],
        "query": paired["query"].map(lambda x: clean_text(x, 220)),

        "id_add": paired["id_add"],
        "item_id_add": paired["item_id_add"],
        "title_add": paired["title_add"].map(lambda x: clean_text(x, 260)),
        "category_add": paired["category_add"].map(lambda x: clean_text(x, 220)),
        "brand_add": paired["brand_add"].map(lambda x: clean_text(x, 80)),
        "gender_add": paired["gender_add"].map(lambda x: clean_text(x, 60)),
        "v33_score_add": paired["v33_ft_score_add"],
        "v33_rank_pct_add": paired["v33_ft_pct_rank_add"],
        "v34_score_add": paired["v34_ce_score_add"],
        "v34_rank_pct_add": paired["v34_ce_pct_rank_add"],

        "id_drop": paired["id_drop"],
        "item_id_drop": paired["item_id_drop"],
        "title_drop": paired["title_drop"].map(lambda x: clean_text(x, 260)),
        "category_drop": paired["category_drop"].map(lambda x: clean_text(x, 220)),
        "brand_drop": paired["brand_drop"].map(lambda x: clean_text(x, 80)),
        "gender_drop": paired["gender_drop"].map(lambda x: clean_text(x, 60)),
        "v33_score_drop": paired["v33_ft_score_drop"],
        "v33_rank_pct_drop": paired["v33_ft_pct_rank_drop"],
        "v34_score_drop": paired["v34_ce_score_drop"],
        "v34_rank_pct_drop": paired["v34_ce_pct_rank_drop"],

        "score_add": paired["score_add"],
        "score_drop": paired["score_drop"],
        "score_gain": paired["score_gain"],

        # assistant will fill these if uploaded
        "assistant_swap_label": "",
        "assistant_confidence": "",
        "assistant_reason": "",
    })

    return out


def stratified_review(df, top_n=400, mid_n=300, tail_n=300, seed=2026):
    if len(df) == 0:
        return df

    rng = np.random.default_rng(seed)
    parts = []

    top = df.head(min(top_n, len(df))).copy()
    top["review_bucket"] = "top_gain"
    parts.append(top)

    if len(df) > top_n:
        mid_start = max(top_n, len(df) // 2 - mid_n // 2)
        mid_end = min(len(df), mid_start + mid_n)
        mid = df.iloc[mid_start:mid_end].copy()
        mid["review_bucket"] = "mid_gain"
        parts.append(mid)

    if len(df) > top_n + mid_n:
        tail = df.tail(min(tail_n, len(df))).copy()
        tail["review_bucket"] = "tail_gain"
        parts.append(tail)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["candidate_name", "id_add", "id_drop"], keep="first")
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"])
    pairs["id"] = pairs["id"].astype(str)
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)

    if not pairs["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("submission_pairs order mismatch sample")

    terms = pd.read_csv(TERMS)
    terms["term_id"] = terms["term_id"].astype(str)

    items = pd.read_csv(ITEMS, low_memory=False)
    items["item_id"] = items["item_id"].astype(str)

    v33 = align_score(V33, sample, ["v33_ft_score", "v33_ft_pct_rank", "v33_ft_term_z"])
    v34 = align_score(V34, sample, ["v34_ce_score", "v34_ce_pct_rank", "v34_ce_term_z"])

    anchor_path = first_existing(ANCHOR_CANDIDATES)
    cand_a_path = first_existing_name(CAND_A_PATTERNS)
    cand_b_path = first_existing_name(CAND_B_PATTERNS)

    if anchor_path is None:
        raise FileNotFoundError("v33 qprob top2000 anchor not found")
    if cand_a_path is None:
        raise FileNotFoundError("b5000 candidate not found")
    if cand_b_path is None:
        raise FileNotFoundError("b10000 candidate not found")

    print("anchor:", anchor_path)
    print("candidate A:", cand_a_path)
    print("candidate B:", cand_b_path)

    anchor = load_pred(anchor_path, sample)
    cand_a = load_pred(cand_a_path, sample)
    cand_b = load_pred(cand_b_path, sample)

    table_a = make_pair_table(anchor, cand_a, "v35_b5000", sample, pairs, terms, items, v33, v34)
    table_b = make_pair_table(anchor, cand_b, "v35_b10000", sample, pairs, terms, items, v33, v34)

    # b10000 extra = b10000 changes excluding b5000 changes
    key_a = set(zip(table_a["id_add"].astype(str), table_a["id_drop"].astype(str)))

    mask_extra = []
    for a, d in zip(table_b["id_add"].astype(str), table_b["id_drop"].astype(str)):
        mask_extra.append((a, d) not in key_a)
    extra_b = table_b.loc[mask_extra].copy().reset_index(drop=True)
    extra_b["candidate_name"] = "v35_b10000_extra_only"

    review_a = stratified_review(table_a, top_n=400, mid_n=300, tail_n=300, seed=2026)
    review_b = stratified_review(extra_b, top_n=300, mid_n=250, tail_n=250, seed=2027)

    review_a.to_csv(OUT_B5000, index=False)
    review_b.to_csv(OUT_B10000_EXTRA, index=False)

    summary = pd.DataFrame([
        {
            "name": "b5000_all",
            "rows": len(table_a),
            "diff_rows_vs_anchor": int((anchor != cand_a).sum()),
            "adds": int(((anchor == 0) & (cand_a == 1)).sum()),
            "drops": int(((anchor == 1) & (cand_a == 0)).sum()),
            "score_gain_mean": float(table_a["score_gain"].mean()),
            "score_gain_min": float(table_a["score_gain"].min()),
            "score_gain_max": float(table_a["score_gain"].max()),
            "review_file": str(OUT_B5000),
            "candidate_file": str(cand_a_path),
        },
        {
            "name": "b10000_all",
            "rows": len(table_b),
            "diff_rows_vs_anchor": int((anchor != cand_b).sum()),
            "adds": int(((anchor == 0) & (cand_b == 1)).sum()),
            "drops": int(((anchor == 1) & (cand_b == 0)).sum()),
            "score_gain_mean": float(table_b["score_gain"].mean()),
            "score_gain_min": float(table_b["score_gain"].min()),
            "score_gain_max": float(table_b["score_gain"].max()),
            "review_file": str(OUT_B10000_EXTRA),
            "candidate_file": str(cand_b_path),
        },
        {
            "name": "b10000_extra_only",
            "rows": len(extra_b),
            "diff_rows_vs_anchor": -1,
            "adds": -1,
            "drops": -1,
            "score_gain_mean": float(extra_b["score_gain"].mean()) if len(extra_b) else np.nan,
            "score_gain_min": float(extra_b["score_gain"].min()) if len(extra_b) else np.nan,
            "score_gain_max": float(extra_b["score_gain"].max()) if len(extra_b) else np.nan,
            "review_file": str(OUT_B10000_EXTRA),
            "candidate_file": str(cand_b_path),
        },
    ])

    summary.to_csv(OUT_SUMMARY, index=False)

    print(summary.to_string(index=False))
    print("saved:", OUT_B5000)
    print("saved:", OUT_B10000_EXTRA)
    print("saved:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
