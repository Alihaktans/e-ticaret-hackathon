import pandas as pd

emb = pd.read_csv("submissions/embedding_v1_minilm_topratio_012.csv")
v2 = pd.read_csv("submissions/embedding_ranker_v2_catboost_text_blend_topratio_012.csv")

scores = pd.read_parquet(
    "data/processed/embedding_v1_minilm_topratio_008_scores.parquet",
    columns=["id", "score"],
)

proba = pd.read_parquet(
    "data/processed/embedding_ranker_v2_catboost_text_test_proba.parquet",
    columns=["id", "embedding_score", "proba", "embedding_rank_score", "final_score"],
)

df = (
    emb.rename(columns={"prediction": "emb_pred"})
    .merge(v2.rename(columns={"prediction": "v2_pred"}), on="id")
    .merge(scores, on="id")
    .merge(proba, on="id")
)

print("Total diff:", int((df["emb_pred"] != df["v2_pred"]).sum()))
print("Diff ratio:", float((df["emb_pred"] != df["v2_pred"]).mean()))

print("\nembedding 1 -> v2 0")
print(
    df[(df.emb_pred == 1) & (df.v2_pred == 0)]
    [["score", "embedding_score", "proba", "embedding_rank_score", "final_score"]]
    .describe()
)

print("\nembedding 0 -> v2 1")
print(
    df[(df.emb_pred == 0) & (df.v2_pred == 1)]
    [["score", "embedding_score", "proba", "embedding_rank_score", "final_score"]]
    .describe()
)
