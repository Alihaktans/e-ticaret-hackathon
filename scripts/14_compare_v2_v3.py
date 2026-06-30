from pathlib import Path
import pandas as pd

FILES = [
    (
        "FINAL_first_submission_v2_e080_m020_top012.csv",
        "embedding_ranker_v3_catboost_text_no_brand_blend_topratio_012.csv",
    ),
    (
        "embedding_v1_minilm_topratio_012.csv",
        "embedding_ranker_v3_catboost_text_no_brand_blend_topratio_012.csv",
    ),
    (
        "embedding_v1_minilm_topratio_012.csv",
        "FINAL_first_submission_v2_e080_m020_top012.csv",
    ),
]

def compare(a_name: str, b_name: str):
    a = pd.read_csv(Path("submissions") / a_name)
    b = pd.read_csv(Path("submissions") / b_name)

    diff = int((a["prediction"] != b["prediction"]).sum())

    print("=" * 100)
    print(a_name)
    print("vs")
    print(b_name)
    print("same ids:", a["id"].equals(b["id"]))
    print("different predictions:", diff)
    print("diff ratio:", diff / len(a))
    print("A counts:")
    print(a["prediction"].value_counts())
    print("B counts:")
    print(b["prediction"].value_counts())


def main():
    for a, b in FILES:
        compare(a, b)


if __name__ == "__main__":
    main()
