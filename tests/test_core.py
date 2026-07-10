import numpy as np

from trendyol.metrics import find_best_threshold, macro_f1
from trendyol.text_preprocess import extract_root_category, normalize_text


def test_normalize_text_handles_turkish_characters() -> None:
    assert normalize_text("Çığ ŞÖLENİ") == "cig soleni"


def test_extract_root_category() -> None:
    assert extract_root_category("Elektronik / Telefon") == "elektronik"
    assert extract_root_category(None) == "unknown"


def test_macro_f1_and_threshold_search() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    result = find_best_threshold(y_true, y_score)
    prediction = (y_score >= result["threshold"]).astype(int)
    assert macro_f1(y_true, prediction) == 1.0

