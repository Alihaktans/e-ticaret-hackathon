from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT_DIR / "reports"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
EXPERIMENT_REPORTS_DIR = REPORTS_DIR / "experiments"

MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

REQUIRED_RAW_FILES = {
    "items": "items.csv",
    "terms": "terms.csv",
    "training_pairs": "training_pairs.csv",
    "submission_pairs": "submission_pairs.csv",
    "sample_submission": "sample_submission.csv",
}

EXPECTED_COLUMNS = {
    "items": ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"],
    "terms": ["term_id", "query"],
    "training_pairs": ["id", "term_id", "item_id", "label"],
    "submission_pairs": ["id", "term_id", "item_id"],
    "sample_submission": ["id", "prediction"],
}


def ensure_output_dirs() -> None:
    """Create local output directories without touching raw input data."""
    for path in (
        PROCESSED_DIR,
        EDA_REPORTS_DIR,
        EXPERIMENT_REPORTS_DIR,
        MODELS_DIR,
        SUBMISSIONS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
