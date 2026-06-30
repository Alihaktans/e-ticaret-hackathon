from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier

MODEL_PATH = Path("models/embedding_ranker_v4_weighted_semantic.cbm")
OUT_PATH = Path("reports/experiments/embedding_ranker_v4_feature_importance.csv")

FEATURE_COLS = [
    "embedding_score",
    "candidate_count",
    "score_mean",
    "score_std",
    "score_max",
    "score_min",
    "score_range",
    "score_z",
    "rank_desc",
    "rank_pct",
    "gap_to_top",
    "query_char_len",
    "title_char_len",
    "category_char_len",
    "query_token_count",
    "title_token_count",
    "category_token_count",
    "qt_overlap_count",
    "qt_overlap_query_ratio",
    "qt_overlap_title_ratio",
    "qc_overlap_count",
    "qc_overlap_query_ratio",
    "qc_overlap_category_ratio",
    "query_in_title",
    "title_in_query",
    "brand_in_query",
    "brand",
    "root_category",
    "gender",
    "age_group",
]

def main():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
