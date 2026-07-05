"""Targeted repair on top of v108 bridge candidates using dual intent models."""
from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(".")

BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
CURRENT = ROOT / "submissions/final_candidates_v108/FINAL_CANDIDATE_v108_bridge_top.csv"
PAIR = ROOT / "data/raw/submission_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"
RANK = ROOT / "data/processed/v72_anchorless_trendlex_features.parquet"
LEX = ROOT / "data/processed/v83_independent_test_scores.parquet"
V108 = ROOT / "data/processed/v108_source_matched_test_scores.parquet"
TEST_FEATURES = ROOT / "data/processed/v21_test_features.parquet"
SCRIPT_209 = ROOT / "scripts/209_make_v100_multi_filterpack.py"

OLD_CACHE = ROOT / "data/processed/v97_targeted_replacement_scores.parquet"
CACHE = ROOT / "data/processed/v109_bridge_targeted_repair_scores.parquet"
SAFE_ADD_CACHE = ROOT / "data/processed/v109_bridge_safe_add_ids_v2.parquet"

OUT = ROOT / "submissions/final_candidates_v109"
REPORT = ROOT / "reports/experiments/v109_bridge_targeted_repair.json"
REVIEW = ROOT / "reports/manual_review/v109_bridge_targeted_repair_review.csv"

MODELS = {
    "v88": ROOT / "models/v88_intent_berturk_from_v62v29/best",
    "v90": ROOT / "models/v90_intent_berturk_from_v62v33/best",
}

THRESHOLDS = [
    {"name": "m0p5", "dual_margin": 0.5, "comp_gap_min": 0.02},
    {"name": "m1", "dual_margin": 1.0, "comp_gap_min": 0.04},
    {"name": "m1p5", "dual_margin": 1.5, "comp_gap_min": 0.05},
    {"name": "m2", "dual_margin": 2.0, "comp_gap_min": 0.08},
]


def repair_text(x):
    x = "" if pd.isna(x) else str(x)
    try:
        return x.encode("latin1").decode("utf8")
    except Exception:
        return x


def item_text(row: pd.Series) -> str:
    parts = []
    fields = [
        ("title", "baslik", 260),
        ("category", "kategori", 180),
        ("brand", "marka", 80),
        ("gender", "cinsiyet", 50),
        ("age_group", "yas", 50),
        ("attributes", "ozellik", 350),
    ]
    for col, label, max_len in fields:
        val = row.get(col, "")
        if pd.isna(val):
            continue
        val = str(val)
        if val in {"", "nan", "unknown"}:
            continue
        parts.append(f"{label}: {repair_text(val)[:max_len]}")
    return " | ".join(parts)


@torch.inference_mode()
def score_rows(df: pd.DataFrame, model_path: Path) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64 if dev == "cuda" else 12
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    out = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), batch_size):
        end = min(len(df), start + batch_size)
        enc = tok(
            df["query_text"].fillna("").iloc[start:end].tolist(),
            df["item_text"].fillna("").iloc[start:end].tolist(),
            padding=True,
            truncation=True,
            max_length=176,
            return_tensors="pt",
        ).to(dev)
        if dev == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc).logits.view(-1).float().cpu().numpy()
        else:
            logits = model(**enc).logits.view(-1).float().cpu().numpy()
        out[start:end] = logits
        if end % 5000 < batch_size:
            print({"model": str(model_path), "scored": end, "total": len(df)}, flush=True)
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def build_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(BASE)
    current = pd.read_csv(CURRENT)
    pairs = pd.read_csv(PAIR, usecols=["id", "term_id", "item_id"])
    rank = pd.read_parquet(RANK, columns=["id", "trendyol_qpct", "lex_qpct"])
    lex = pd.read_parquet(LEX, columns=["id", "v82_score"])
    v108 = pd.read_parquet(V108, columns=["id", "v108_score"])
    for df in (base, current, pairs, rank, lex, v108):
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        if "term_id" in df.columns:
            df["term_id"] = df["term_id"].astype(str)
        if "item_id" in df.columns:
            df["item_id"] = df["item_id"].astype(str)

    if not base["id"].equals(current["id"]) or not base["id"].equals(pairs["id"]):
        raise RuntimeError("alignment failure across base/current/pairs")

    full = pairs.merge(base[["id", "prediction"]].rename(columns={"prediction": "base"}), on="id")
    full = full.merge(current[["id", "prediction"]].rename(columns={"prediction": "current"}), on="id")
    full = full.merge(rank, on="id", how="left").merge(lex, on="id", how="left").merge(v108, on="id", how="left")
    full["trendyol_qpct"] = full["trendyol_qpct"].fillna(0).astype(np.float32)
    full["lex_qpct"] = full["lex_qpct"].fillna(0).astype(np.float32)
    full["v82_score"] = full["v82_score"].fillna(0).astype(np.float32)
    full["v108_score"] = full["v108_score"].fillna(0).astype(np.float32)
    full["comp"] = (
        0.45 * full["v108_score"]
        + 0.35 * full["v82_score"]
        + 0.10 * full["trendyol_qpct"]
        + 0.10 * full["lex_qpct"]
    ).astype(np.float32)

    y_base = full["base"].to_numpy(np.int8)
    y_cur = full["current"].to_numpy(np.int8)
    removed = (y_base == 1) & (y_cur == 0)
    removed_terms = set(full.loc[removed, "term_id"])
    safe_add_ids = build_safe_add_ids(full, removed_terms)

    rows = []
    for term_id, idx in full.groupby("term_id", sort=False).indices.items():
        if term_id not in removed_terms:
            continue
        idx = np.asarray(idx)
        drop = idx[removed[idx]]
        add = idx[((y_base[idx] == 0) & full.iloc[idx]["id"].isin(safe_add_ids).to_numpy())]
        if not len(drop) or not len(add):
            continue
        drop = drop[np.argsort(full["comp"].to_numpy()[drop])]
        add = add[np.argsort(-full["comp"].to_numpy()[add])]
        used_add = set()
        for di in drop:
            for ai in add:
                if ai == di or ai in used_add:
                    continue
                used_add.add(int(ai))
                rows.append((str(term_id), int(ai), int(di)))
                break

    pair_df = pd.DataFrame(rows, columns=["term_id", "add_row", "drop_row"])
    if pair_df.empty:
        return full, pair_df

    pair_df["add_id"] = full.iloc[pair_df["add_row"].to_numpy()]["id"].to_numpy()
    pair_df["drop_id"] = full.iloc[pair_df["drop_row"].to_numpy()]["id"].to_numpy()
    pair_df["add_item_id"] = full.iloc[pair_df["add_row"].to_numpy()]["item_id"].to_numpy()
    pair_df["drop_item_id"] = full.iloc[pair_df["drop_row"].to_numpy()]["item_id"].to_numpy()
    pair_df["add_comp"] = full.iloc[pair_df["add_row"].to_numpy()]["comp"].to_numpy()
    pair_df["drop_comp"] = full.iloc[pair_df["drop_row"].to_numpy()]["comp"].to_numpy()
    pair_df["comp_gap"] = pair_df["add_comp"] - pair_df["drop_comp"]
    pair_df["add_v108_score"] = full.iloc[pair_df["add_row"].to_numpy()]["v108_score"].to_numpy()
    pair_df["drop_v108_score"] = full.iloc[pair_df["drop_row"].to_numpy()]["v108_score"].to_numpy()
    pair_df["add_v82_score"] = full.iloc[pair_df["add_row"].to_numpy()]["v82_score"].to_numpy()
    pair_df["drop_v82_score"] = full.iloc[pair_df["drop_row"].to_numpy()]["v82_score"].to_numpy()
    return full, pair_df


def load_mod209():
    spec = importlib.util.spec_from_file_location("mod209", SCRIPT_209)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_safe_add_ids(full: pd.DataFrame, term_ids: set[str]) -> set[str]:
    candidate = full[(full["term_id"].isin(term_ids)) & full["base"].eq(0)][["id", "term_id", "item_id", "v108_score"]].copy()
    candidate["id"] = candidate["id"].astype(str)
    candidate["term_id"] = candidate["term_id"].astype(str)
    candidate["item_id"] = candidate["item_id"].astype(str)
    if SAFE_ADD_CACHE.exists():
        cache = pd.read_parquet(SAFE_ADD_CACHE)
        cache["id"] = cache["id"].astype(str)
        if len(cache) == len(candidate) and set(cache["id"]) == set(candidate["id"]):
            return set(cache.loc[cache["safe_add"].eq(1), "id"])

    mod209 = load_mod209()
    brand_tokens = mod209.build_brand_token_set()
    query_info = mod209.build_query_info(set(candidate["term_id"]), brand_tokens)
    item_info = mod209.build_item_meta(set(candidate["item_id"]))
    feat_cols = [
        "id",
        "age_mismatch",
        "digit_mismatch",
        "size_mismatch",
        "query_has_color",
        "color_mismatch",
        "query_has_known_brand",
        "query_brand_match",
        "brand_in_query_pair",
    ]
    feat = pd.read_parquet(TEST_FEATURES, columns=feat_cols)
    feat["id"] = feat["id"].astype(str)
    rows = candidate.rename(columns={"v108_score": "score"}).copy()
    rows = rows.merge(feat, on="id", how="left")
    rows["title_cov_pct_rank"] = np.float32(0.5)
    rows["label"] = -1
    eval_df = mod209.evaluate_rows(
        rows[["id", "term_id", "item_id", "score", "title_cov_pct_rank", "label"]],
        query_info,
        item_info,
        {},
    )
    safe = eval_df["head_augmented_veto"].eq(0)
    safe &= rows["age_mismatch"].fillna(0).eq(0).to_numpy()
    safe &= rows["digit_mismatch"].fillna(0).eq(0).to_numpy()
    safe &= rows["size_mismatch"].fillna(0).eq(0).to_numpy()
    safe &= (~rows["query_has_color"].fillna(0).eq(1) | rows["color_mismatch"].fillna(0).eq(0)).to_numpy()
    safe &= (
        ~rows["query_has_known_brand"].fillna(0).eq(1)
        | rows["query_brand_match"].fillna(0).eq(1)
        | rows["brand_in_query_pair"].fillna(0).eq(1)
    ).to_numpy()
    eval_df["safe_add"] = safe.astype(np.int8)
    SAFE_ADD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    eval_df[["id", "safe_add", "reason_text"]].to_parquet(SAFE_ADD_CACHE, index=False)
    return set(eval_df.loc[eval_df["safe_add"].eq(1), "id"])


def build_scoring_frame(full: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    need_rows = np.unique(np.r_[pair_df["add_row"].to_numpy(), pair_df["drop_row"].to_numpy()])
    d = full.iloc[need_rows][["id", "term_id", "item_id"]].copy()
    d["row"] = need_rows

    terms = pd.read_csv(TERMS, usecols=["term_id", "query"])
    terms["term_id"] = terms["term_id"].astype(str)

    items = pd.read_csv(ITEMS, usecols=["item_id", "title", "category", "brand", "gender", "age_group", "attributes"], low_memory=False)
    items["item_id"] = items["item_id"].astype(str)
    items = items[items["item_id"].isin(set(d["item_id"]))].copy()
    items["item_text"] = items.apply(item_text, axis=1)

    d = d.merge(terms, on="term_id", how="left").merge(items[["item_id", "item_text"]], on="item_id", how="left")
    d["query_text"] = "sorgu: " + d["query"].map(repair_text).str.slice(0, 180)
    d = d.sort_values("row").reset_index(drop=True)

    if OLD_CACHE.exists():
        old_full = pd.read_parquet(OLD_CACHE)
        old_cols = [c for c in ["id", "v88_score", "v90_score"] if c in old_full.columns]
        old = old_full[old_cols].copy()
        old["id"] = old["id"].astype(str)
        d = d.merge(old, on="id", how="left")

    if CACHE.exists():
        cur = pd.read_parquet(CACHE)
        cur["id"] = cur["id"].astype(str)
        keep = ["id"] + [c for c in cur.columns if c.endswith("_score")]
        d = d.drop(columns=[c for c in d.columns if c.endswith("_score")], errors="ignore").merge(cur[keep], on="id", how="left")

    for name, model_path in MODELS.items():
        col = f"{name}_score"
        if col not in d.columns:
            d[col] = np.nan
        miss = d[col].isna()
        if miss.any():
            d.loc[miss, col] = score_rows(d.loc[miss, ["query_text", "item_text"]].copy(), model_path)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d[["id", "row", "term_id", "item_id", "query", "item_text", "query_text", "v88_score", "v90_score"]].to_parquet(CACHE, index=False)
    return d


def save_candidates(full: pd.DataFrame, pair_df: pd.DataFrame) -> dict:
    row_map = build_scoring_frame(full, pair_df).set_index("row")
    for name in MODELS:
        pair_df[f"margin_{name}"] = row_map.loc[pair_df["add_row"], f"{name}_score"].to_numpy() - row_map.loc[pair_df["drop_row"], f"{name}_score"].to_numpy()
    pair_df["min_margin"] = pair_df[[f"margin_{name}" for name in MODELS]].min(axis=1)

    base_pred = full["current"].to_numpy(np.int8)
    outs = []
    review_rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for cfg in THRESHOLDS:
        ok = (
            pair_df["margin_v88"].gt(cfg["dual_margin"])
            & pair_df["margin_v90"].gt(cfg["dual_margin"])
            & pair_df["comp_gap"].gt(cfg["comp_gap_min"])
        )
        pred = base_pred.copy()
        pred[pair_df.loc[ok, "add_row"].to_numpy()] = 1
        out_path = OUT / f"FINAL_CANDIDATE_v109_bridge_repair_{cfg['name']}.csv"
        pd.DataFrame({"id": full["id"], "prediction": pred}).to_csv(out_path, index=False)
        accepted = pair_df.loc[ok].copy()
        accepted["variant"] = cfg["name"]
        review_rows.append(accepted)
        outs.append(
            {
                "variant": cfg["name"],
                "dual_margin": cfg["dual_margin"],
                "comp_gap_min": cfg["comp_gap_min"],
                "accepted": int(ok.sum()),
                "positives": int(pred.sum()),
                "ratio": float(pred.mean()),
                "file": str(out_path),
                "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            }
        )

    review = pd.concat(review_rows, ignore_index=True) if review_rows else pd.DataFrame()
    if not review.empty:
        terms = pd.read_csv(TERMS, usecols=["term_id", "query"])
        terms["term_id"] = terms["term_id"].astype(str)
        items = pd.read_csv(ITEMS, usecols=["item_id", "title", "category", "brand", "gender"], low_memory=False)
        items["item_id"] = items["item_id"].astype(str)
        review = review.merge(terms, on="term_id", how="left")
        review = review.merge(items.add_prefix("add_"), left_on="add_item_id", right_on="add_item_id", how="left")
        review = review.merge(items.add_prefix("drop_"), left_on="drop_item_id", right_on="drop_item_id", how="left")
        review = review.sort_values(["variant", "min_margin", "comp_gap"], ascending=[True, False, False])
        REVIEW.parent.mkdir(parents=True, exist_ok=True)
        review.to_csv(REVIEW, index=False)

    report = {
        "base_file": str(BASE),
        "current_file": str(CURRENT),
        "initial_removed_vs_base": int(((full["base"] == 1) & (full["current"] == 0)).sum()),
        "proposed_pairs": int(len(pair_df)),
        "pair_comp_gap_mean": float(pair_df["comp_gap"].mean()) if len(pair_df) else 0.0,
        "margin_correlation": float(pair_df[["margin_v88", "margin_v90"]].corr().iloc[0, 1]) if len(pair_df) else float("nan"),
        "candidates": outs,
        "review_file": str(REVIEW),
        "cache_file": str(CACHE),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
    return report


def main():
    full, pair_df = build_pairs()
    if pair_df.empty:
        report = {
            "base_file": str(BASE),
            "current_file": str(CURRENT),
            "initial_removed_vs_base": 0,
            "proposed_pairs": 0,
            "candidates": [],
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")
        print(json.dumps(report, indent=2))
        return
    save_candidates(full, pair_df)


if __name__ == "__main__":
    main()
