import pandas as pd

base = pd.read_csv("submissions/FINAL_first_submission_v2_e080_m020_top012.csv")
v4 = pd.read_csv("submissions/embedding_ranker_v4_weighted_semantic_e080_m020_topratio_012.csv")

scores = pd.read_parquet(
    "data/processed/embedding_v1_minilm_topratio_008_scores.parquet",
    columns=["id", "score"],
)

v4_proba = pd.read_parquet(
    "data/processed/embedding_ranker_v4_weighted_semantic_test_proba.parquet",
    columns=["id", "embedding_score", "proba", "embedding_rank_score", "final_score"],
)

df = (
    base.rename(columns={"prediction": "base_pred"})
    .merge(v4.rename(columns={"prediction": "v4_pred"}), on="id")
    .merge(scores, on="id")
    .merge(v4_proba, on="id")
)

print("Total diff:", int((df["base_pred"] != df["v4_pred"]).sum()))
print("Diff ratio:", float((df["base_pred"] != df["v4_pred"]).mean()))

print("\nbase 1 -> v4 0")
print(
    df[(df.base_pred == 1) & (df.v4_pred == 0)]
    [["score", "embedding_score", "proba", "embedding_rank_score", "final_score"]]
    .describe()
)

print("\nbase 0 -> v4 1")
print(
    df[(df.base_pred == 0) & (df.v4_pred == 1)]
    [["score", "embedding_score", "proba", "embedding_rank_score", "final_score"]]
    .describe()
)
