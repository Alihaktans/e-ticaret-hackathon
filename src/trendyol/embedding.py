from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sentence_model(model_name: str) -> SentenceTransformer:
    device = get_device()
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        print(f"Loading cache: {cache_path}")
        return np.load(cache_path)

    print(f"Encoding {len(texts):,} texts...")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float16)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    print(f"Saved cache: {cache_path}")

    return emb


def cosine_by_index(
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> np.ndarray:
    left = left_emb.astype(np.float32)
    right = right_emb.astype(np.float32)
    return np.sum(left[left_idx] * right[right_idx], axis=1).astype(np.float32)
