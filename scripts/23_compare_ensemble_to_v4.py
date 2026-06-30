from pathlib import Path
import pandas as pd

BASE = "embedding_ranker_v4_weighted_semantic_e080_m020_topratio_012.csv"

CANDIDATES = [
    "embedding_ranker_v4_weight_seed_ensemble_avg_e090_m010_top012.csv",
    "embedding_ranker_v4_weight_seed_ensemble_avg_e080_m020_top012.csv",
    "embedding_ranker_v4_weight_seed_ensemble_avg_e070_m030_top012.csv",
    "embedding_ranker_v4_weight_seed_ensemble_weighted_e090_m010_top012.csv",
    "embedding_ranker_v4_weight_seed_ensemble_weighted_e080_m020_top012.csv",
    "embedding_ranker_v4_weight_seed_ensemble_weighted_e070_m030_top012.csv",
    "embedding_ranker_v4_weighted_semantic_e080_m020_topratio_010.csv",
    "embedding_ranker_v4_weighted_semantic_e080_m020_topratio_014.csv",
]

def read(name):
    path = Path("submissions") / name
    if not path.exists():
        print("MISSING:", path)
        return None
    return pd.read_csv(path)

def main():
    base = read(BASE)
    if base is None:
        raise FileNotFoundError(BASE)

    rows = []

    for name in CANDIDATES:
        cand = read(name)
        if cand is None:
            continue

        assert base["id"].equals(cand["id"])

        diff = int((base["prediction"] != cand["prediction"]).sum())

        rows.append(
            {
                "candidate": name,
                "ones": int(cand["prediction"].sum()),
                "pos_ratio": float(cand["prediction"].mean()),
                "diff_vs_base": diff,
                "diff_ratio_vs_base": diff / len(cand),
            }
        )

    out = pd.DataFrame(rows).sort_values(["diff_ratio_vs_base", "candidate"])
    print(out.to_string(index=False))

    out_path = Path("reports/experiments/ensemble_vs_v4_compare.csv")
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
