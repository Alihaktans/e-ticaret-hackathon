"""Use Trendyol embeddings to audit V104/V219/V220 disagreements.

This script treats the 0.831 V104 submission as an anchor, not ground truth. It
encodes only rows where the anchor and V220 disagree, requires V219/V220
consensus, and creates small prior-preserving swaps.

Stages:
    python scripts/221_build_v104_embedding_guarded_swaps.py prepare
    python scripts/221_build_v104_embedding_guarded_swaps.py embed
    python scripts/221_build_v104_embedding_guarded_swaps.py swap
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
REPORTS = ROOT / "reports/experiments"
OUT_DIR = ROOT / "submissions/final_candidates_v221"

TEACHER = ROOT / "submissions/reference/FINAL_CANDIDATE_v104_full_balanced.csv"
V219_SCORE = PROC / "v219_test_scores.parquet"
V220_SCORE = PROC / "v220_crossfit_hard_negative_test_scores.parquet"
PAIRS = RAW / "submission_pairs.csv"
TERMS = RAW / "terms.csv"
ITEMS = RAW / "items.csv"

DISAGREEMENT = PROC / "v221_v104_v220_disagreement.parquet"
QUERY_IDS = PROC / "v221_query_ids.parquet"
ITEM_IDS = PROC / "v221_item_ids.parquet"
QUERY_EMB = PROC / "v221_query_embeddings_128.npy"
ITEM_EMB = PROC / "v221_item_embeddings_128.npy"
SCORED = PROC / "v221_embedding_guarded_disagreement.parquet"
REPORT = REPORTS / "v221_embedding_guarded_swaps.json"

MODEL_NAME = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
EMBED_DIM = 128
BATCH_SIZE = int(os.environ.get("V221_BATCH_SIZE", "64"))
BUDGETS = [500, 1000, 3000, 5000, 10000]


def assert_ids(left: pd.Series, right: pd.Series, label: str) -> None:
    if len(left) != len(right) or not left.astype(str).reset_index(drop=True).equals(
        right.astype(str).reset_index(drop=True)
    ):
        raise ValueError(f"{label} ID alignment mismatch")


def score_column(frame: pd.DataFrame) -> str:
    candidates = [column for column in frame.columns if column.endswith("_score")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one score column, found {candidates}")
    return candidates[0]


def prepare_disagreement() -> None:
    required = [TEACHER, V219_SCORE, V220_SCORE, PAIRS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V221 inputs:\n- " + "\n- ".join(missing))

    teacher = pd.read_csv(TEACHER, usecols=["id", "prediction"])
    if teacher["id"].duplicated().any() or teacher["prediction"].isna().any():
        raise ValueError("Teacher contains duplicate IDs or missing predictions")
    teacher["id"] = teacher["id"].astype(str)
    teacher["prediction"] = teacher["prediction"].astype(np.int8)
    teacher_ratio = float(teacher["prediction"].mean())

    v219 = pd.read_parquet(V219_SCORE)
    v220 = pd.read_parquet(V220_SCORE)
    v219_col = score_column(v219)
    v220_col = score_column(v220)
    assert_ids(teacher["id"], v219["id"], "teacher/V219")
    assert_ids(teacher["id"], v220["id"], "teacher/V220")

    p219 = v219[v219_col].astype(np.float32).to_numpy()
    p220 = v220[v220_col].astype(np.float32).to_numpy()
    threshold219 = float(np.quantile(p219, 1.0 - teacher_ratio))
    threshold220 = float(np.quantile(p220, 1.0 - teacher_ratio))
    pred219 = (p219 >= threshold219).astype(np.int8)
    pred220 = (p220 >= threshold220).astype(np.int8)
    teacher_pred = teacher["prediction"].to_numpy(np.int8)

    disagreement_mask = teacher_pred != pred220
    consensus_mask = pred219 == pred220
    selected = disagreement_mask & consensus_mask

    pairs = pd.read_csv(PAIRS, usecols=["id", "term_id", "item_id"], dtype=str)
    assert_ids(teacher["id"], pairs["id"], "teacher/pairs")
    out = pairs.loc[selected].copy()
    out["teacher"] = teacher_pred[selected]
    out["v219_pred"] = pred219[selected]
    out["v220_pred"] = pred220[selected]
    out["v219_score"] = p219[selected]
    out["v220_score"] = p220[selected]
    out["direction"] = np.where(out["teacher"].eq(0), "add", "remove")
    out.to_parquet(DISAGREEMENT, index=False, compression="zstd")

    summary = {
        "teacher_ratio": teacher_ratio,
        "threshold_v219_at_teacher_ratio": threshold219,
        "threshold_v220_at_teacher_ratio": threshold220,
        "all_disagreements": int(disagreement_mask.sum()),
        "student_consensus_disagreements": int(selected.sum()),
        "add_candidates": int(out["direction"].eq("add").sum()),
        "remove_candidates": int(out["direction"].eq("remove").sum()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    print("saved", DISAGREEMENT, flush=True)


def item_text(frame: pd.DataFrame) -> pd.Series:
    for column in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        frame[column] = frame[column].fillna("").astype(str)
    return (
        "urun: "
        + frame["title"]
        + ". kategori: "
        + frame["category"]
        + ". marka: "
        + frame["brand"]
        + ". cinsiyet: "
        + frame["gender"]
        + ". yas: "
        + frame["age_group"]
        + ". ozellikler: "
        + frame["attributes"].str.slice(0, 900)
    )


def save_embeddings(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float16, copy=False))
    temporary.replace(path)


def encode_disagreement() -> None:
    if not DISAGREEMENT.exists():
        raise FileNotFoundError(f"Run prepare first: {DISAGREEMENT}")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install the NLP dependencies: pip install -e \".[nlp]\"") from exc

    rows = pd.read_parquet(DISAGREEMENT, columns=["term_id", "item_id"])
    query_ids = pd.Series(rows["term_id"].astype(str).unique(), name="term_id")
    item_ids = pd.Series(rows["item_id"].astype(str).unique(), name="item_id")
    query_set = set(query_ids)
    item_set = set(item_ids)

    terms = pd.read_csv(TERMS, usecols=["term_id", "query"], dtype=str)
    terms = terms[terms["term_id"].isin(query_set)].copy()
    terms = terms.drop_duplicates("term_id").set_index("term_id").reindex(query_ids).reset_index()
    if terms["query"].isna().any():
        raise ValueError("Missing query text in V221 disagreement set")

    item_frames = []
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    for chunk in pd.read_csv(ITEMS, usecols=columns, dtype=str, chunksize=100000, low_memory=False):
        keep = chunk[chunk["item_id"].isin(item_set)]
        if not keep.empty:
            item_frames.append(keep)
    items = pd.concat(item_frames, ignore_index=True).drop_duplicates("item_id")
    items = items.set_index("item_id").reindex(item_ids).reset_index()
    if items["title"].isna().any():
        raise ValueError("Missing item metadata in V221 disagreement set")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": device, "queries": len(terms), "items": len(items), "batch": BATCH_SIZE}, flush=True)
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        truncate_dim=EMBED_DIM,
        device=device,
    )
    query_embeddings = model.encode(
        terms["query"].fillna("").tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    product_texts = item_text(items).tolist()
    item_embeddings = model.encode(
        product_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    pd.DataFrame({"term_id": query_ids}).to_parquet(QUERY_IDS, index=False)
    pd.DataFrame({"item_id": item_ids}).to_parquet(ITEM_IDS, index=False)
    save_embeddings(QUERY_EMB, query_embeddings)
    save_embeddings(ITEM_EMB, item_embeddings)
    print("saved embedding caches", flush=True)
    del model, query_embeddings, item_embeddings, items, terms, product_texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def percentile(values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.rank(method="average", pct=True, ascending=ascending).astype(np.float32)


def score_and_swap() -> None:
    required = [DISAGREEMENT, QUERY_IDS, ITEM_IDS, QUERY_EMB, ITEM_EMB, TEACHER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V221 caches:\n- " + "\n- ".join(missing))

    rows = pd.read_parquet(DISAGREEMENT)
    query_ids = pd.read_parquet(QUERY_IDS)["term_id"].astype(str)
    item_ids = pd.read_parquet(ITEM_IDS)["item_id"].astype(str)
    query_embeddings = np.load(QUERY_EMB).astype(np.float32)
    item_embeddings = np.load(ITEM_EMB).astype(np.float32)
    query_map = pd.Series(np.arange(len(query_ids), dtype=np.int32), index=query_ids)
    item_map = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)
    query_index = rows["term_id"].astype(str).map(query_map)
    item_index = rows["item_id"].astype(str).map(item_map)
    if query_index.isna().any() or item_index.isna().any():
        raise ValueError("Embedding ID mapping failed")
    semantic = np.sum(
        query_embeddings[query_index.to_numpy(np.int32)] * item_embeddings[item_index.to_numpy(np.int32)],
        axis=1,
    ).astype(np.float32)
    rows["trendyol_embedding"] = semantic

    add = rows[rows["direction"].eq("add")].copy()
    remove = rows[rows["direction"].eq("remove")].copy()
    add["priority"] = (
        0.55 * percentile(add["trendyol_embedding"])
        + 0.25 * percentile(add["v220_score"])
        + 0.20 * percentile(add["v219_score"])
    )
    remove["priority"] = (
        0.55 * percentile(remove["trendyol_embedding"], ascending=False)
        + 0.25 * percentile(remove["v220_score"], ascending=False)
        + 0.20 * percentile(remove["v219_score"], ascending=False)
    )
    # Require semantic evidence in the expected direction, not merely lexical consensus.
    add_semantic_cutoff = float(add["trendyol_embedding"].quantile(0.75))
    remove_semantic_cutoff = float(remove["trendyol_embedding"].quantile(0.25))
    add = add[add["trendyol_embedding"].ge(add_semantic_cutoff)].sort_values("priority", ascending=False)
    remove = remove[remove["trendyol_embedding"].le(remove_semantic_cutoff)].sort_values(
        "priority", ascending=False
    )
    rows.to_parquet(SCORED, index=False, compression="zstd")

    teacher = pd.read_csv(TEACHER, usecols=["id", "prediction"])
    teacher["id"] = teacher["id"].astype(str)
    teacher["prediction"] = teacher["prediction"].astype(np.int8)
    base_ratio = float(teacher["prediction"].mean())
    summaries = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for requested_budget in BUDGETS:
        budget = min(requested_budget, len(add), len(remove))
        if budget == 0:
            continue
        add_ids = set(add.head(budget)["id"].astype(str))
        remove_ids = set(remove.head(budget)["id"].astype(str))
        prediction = teacher["prediction"].to_numpy(copy=True)
        prediction[teacher["id"].isin(add_ids).to_numpy()] = 1
        prediction[teacher["id"].isin(remove_ids).to_numpy()] = 0
        path = OUT_DIR / f"FINAL_CANDIDATE_v221_embedding_guarded_swap_{budget}.csv"
        pd.DataFrame({"id": teacher["id"], "prediction": prediction}).to_csv(path, index=False)
        summaries.append(
            {
                "budget_each_direction": budget,
                "total_changes": int((prediction != teacher["prediction"].to_numpy()).sum()),
                "positive_ratio": float(prediction.mean()),
                "ratio_delta": float(prediction.mean() - base_ratio),
                "file": str(path),
            }
        )

    report = {
        "anchor": str(TEACHER),
        "anchor_positive_ratio": base_ratio,
        "model": MODEL_NAME,
        "embedding_dimension": EMBED_DIM,
        "disagreement_rows": len(rows),
        "eligible_add_rows": len(add),
        "eligible_remove_rows": len(remove),
        "add_semantic_cutoff_q75": add_semantic_cutoff,
        "remove_semantic_cutoff_q25": remove_semantic_cutoff,
        "outputs": summaries,
        "policy": "V104 anchor; V219/V220 consensus; Trendyol embedding guard; equal add/remove swaps",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "embed", "swap", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare_disagreement()
    if args.stage in {"embed", "all"}:
        encode_disagreement()
    if args.stage in {"swap", "all"}:
        score_and_swap()


if __name__ == "__main__":
    main()
