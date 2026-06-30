from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

from trendyol.text_preprocess import extract_root_category


@dataclass
class NegativeSamplingConfig:
    random_state: int = 42
    random_negatives_per_positive: int = 1
    same_root_negatives_per_positive: int = 1


def prepare_item_sampling_table(items: pd.DataFrame) -> pd.DataFrame:
    out = items[["item_id", "category"]].copy()
    out["root_category"] = out["category"].map(extract_root_category)
    return out


def build_positive_set(train_pairs: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(train_pairs["term_id"].values, train_pairs["item_id"].values))


def sample_one_valid_item(
    rng: np.random.Generator,
    candidates: np.ndarray,
    term_id: str,
    positive_set: set[tuple[str, str]],
    forbidden_item_id: str | None = None,
    max_tries: int = 50,
) -> str | None:
    if len(candidates) == 0:
        return None

    for _ in range(max_tries):
        item_id = str(rng.choice(candidates))

        if forbidden_item_id is not None and item_id == forbidden_item_id:
            continue

        if (term_id, item_id) in positive_set:
            continue

        return item_id

    return None


def generate_negative_pairs(
    train_pairs: pd.DataFrame,
    items: pd.DataFrame,
    cfg: NegativeSamplingConfig,
) -> pd.DataFrame:
    """
    Generate synthetic negative examples from positive-only training pairs.

    Important:
    These labels are synthetic and may contain noise.
    They must be used with group-based validation and careful thresholding.
    """
    rng = np.random.default_rng(cfg.random_state)

    train = train_pairs[["term_id", "item_id"]].copy()
    item_table = prepare_item_sampling_table(items)

    item_to_root = dict(zip(item_table["item_id"], item_table["root_category"]))

    all_item_ids = item_table["item_id"].values

    root_to_items = {
        root: group["item_id"].values
        for root, group in item_table.groupby("root_category", sort=False)
    }

    positive_set = build_positive_set(train_pairs)

    rows = []
    neg_id = 0

    print("Generating synthetic negatives...")

    for row in tqdm(train.itertuples(index=False), total=len(train), desc="negative sampling"):
        term_id = str(row.term_id)
        pos_item_id = str(row.item_id)

        # 1) Easy random negatives
        for _ in range(cfg.random_negatives_per_positive):
            item_id = sample_one_valid_item(
                rng=rng,
                candidates=all_item_ids,
                term_id=term_id,
                positive_set=positive_set,
                forbidden_item_id=pos_item_id,
            )

            if item_id is None:
                continue

            rows.append(
                {
                    "id": f"NEG_RANDOM_{neg_id}",
                    "term_id": term_id,
                    "item_id": item_id,
                    "label": 0,
                    "negative_type": "easy_random",
                }
            )
            neg_id += 1

        # 2) Same-root category hard negatives
        pos_root = item_to_root.get(pos_item_id, "unknown")
        same_root_candidates = root_to_items.get(pos_root, np.array([], dtype=object))

        for _ in range(cfg.same_root_negatives_per_positive):
            item_id = sample_one_valid_item(
                rng=rng,
                candidates=same_root_candidates,
                term_id=term_id,
                positive_set=positive_set,
                forbidden_item_id=pos_item_id,
            )

            if item_id is None:
                continue

            rows.append(
                {
                    "id": f"NEG_SAME_ROOT_{neg_id}",
                    "term_id": term_id,
                    "item_id": item_id,
                    "label": 0,
                    "negative_type": "same_root_hard",
                }
            )
            neg_id += 1

    negatives = pd.DataFrame(rows)

    if negatives.empty:
        raise RuntimeError("No negatives generated.")

    return negatives
