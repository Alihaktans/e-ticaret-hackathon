import numpy as np
import pandas as pd

from trendyol.metrics import find_best_threshold, macro_f1
from trendyol.meta_ensemble import add_meta_features, assert_same_rows, threshold_report
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


def test_meta_features_and_alignment() -> None:
    frame = pd.DataFrame(
        {
            "term_id": ["q1", "q1", "q2"],
            "item_id": ["a", "b", "c"],
            "v106_score": [0.9, 0.2, 0.7],
            "v108_score": [0.8, 0.4, 0.6],
        }
    )
    enriched, features = add_meta_features(frame)
    assert "score_abs_gap" in features
    assert enriched.loc[0, "v108_score_rank_pct"] < enriched.loc[1, "v108_score_rank_pct"]
    assert_same_rows(frame, frame.copy(), ["term_id", "item_id"], "test")


def test_stable_threshold_report() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])
    folds = np.array([0, 0, 1, 1])
    report = threshold_report(labels, probabilities, folds, np.array([0.5]))
    assert report.iloc[0]["macro_f1"] == 1.0
