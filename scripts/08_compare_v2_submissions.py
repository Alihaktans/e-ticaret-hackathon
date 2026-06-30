from pathlib import Path
import pandas as pd

pairs = [
    (
        "embedding_v1_minilm_topratio_012.csv",
        "embedding_ranker_v2_catboost_text_blend_topratio_012.csv",
    ),
    (
        "embedding_v1_minilm_topratio_008.csv",
        "embedding_ranker_v2_catboost_text_blend_topratio_008.csv",
    ),
]

for a_name, b_name in pairs:
    a_path = Path("submissions") / a_name
    b_path = Path("submissions") / b_name

    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)

    diff = (a["prediction"] != b["prediction"]).sum()

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
