from __future__ import annotations

import re
import pandas as pd


TR_TRANSLATION = str.maketrans(
    {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c",
        "İ": "i",
        "Ğ": "g",
        "Ü": "u",
        "Ş": "s",
        "Ö": "o",
        "Ç": "c",
    }
)


def normalize_text(value: object) -> str:
    """Normalize Turkish e-commerce text for lexical ML features."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).lower().translate(TR_TRANSLATION)
    text = re.sub(r"[^a-z0-9\s/:.,%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_root_category(category: object) -> str:
    text = normalize_text(category)
    if not text:
        return "unknown"
    return text.split("/")[0].strip() or "unknown"


def build_item_text(row: pd.Series) -> str:
    title = normalize_text(row.get("title", ""))
    category = normalize_text(row.get("category", ""))
    brand = normalize_text(row.get("brand", ""))
    gender = normalize_text(row.get("gender", ""))
    age_group = normalize_text(row.get("age_group", ""))
    attributes = normalize_text(row.get("attributes", ""))

    return (
        f"urun: {title}. "
        f"kategori: {category}. "
        f"marka: {brand}. "
        f"cinsiyet: {gender}. "
        f"yas: {age_group}. "
        f"ozellikler: {attributes}"
    )


def build_pair_text(query: object, item_text: object) -> str:
    return f"arama: {normalize_text(query)} [SEP] {normalize_text(item_text)}"
