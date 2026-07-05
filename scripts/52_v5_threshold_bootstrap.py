from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(".")
LABEL_PATH = ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv"
SCORE_PATH = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
OUT = ROOT / "reports/manual_review/v5_threshold_bootstrap.csv"

THRESHOLDS = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
N_BOOT = 1000
SEED = 2026

def macro(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1])

labels = pd.read_csv(LABEL_PATH)
labels["id"] = labels["id"].astype(str)
labels = labels[labels["assistant_label"].isin([0, 1])].copy()
labels["assistant_label"] = labels["assistant_label"].astype(int)

scores = pd.read_parquet(SCORE_PATH, columns=["id", "proba_avg"])
scores["id"] = scores["id"].astype(str)

df = labels.merge(scores, on="id", how="left", validate="many_to_one")

subsets = {
    "all": df,
    "clean": df[
        (df["needs_recheck"].fillna(0).astype(int) == 0)
        & (df["assistant_confidence"].astype(str).str.lower().isin(["high", "medium"]))
    ].copy(),
    "high_clean": df[
        (df["needs_recheck"].fillna(0).astype(int) == 0)
        & (df["assistant_confidence"].astype(str).str.lower().eq("high"))
    ].copy(),
}

rng = np.random.default_rng(SEED)
rows = []

for subset_name, part in subsets.items():
    part = part.dropna(subset=["proba_avg"]).copy()
    y = part["assistant_label"].to_numpy()
    s = part["proba_avg"].to_numpy()
    n = len(part)

    wins = {th: 0 for th in THRESHOLDS}
    scores_by_th = {th: [] for th in THRESHOLDS}

    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        ss = s[idx]

        local = []
        for th in THRESHOLDS:
            pred = (ss >= th).astype(int)
            val = macro(yy, pred)
            scores_by_th[th].append(val)
            local.append((val, th))

        best_val, best_th = max(local)
        wins[best_th] += 1

    for th in THRESHOLDS:
        vals = np.array(scores_by_th[th])
        rows.append({
            "subset": subset_name,
            "threshold": th,
            "mean_macro_f1": float(vals.mean()),
            "std_macro_f1": float(vals.std()),
            "p05_macro_f1": float(np.quantile(vals, 0.05)),
            "p50_macro_f1": float(np.quantile(vals, 0.50)),
            "p95_macro_f1": float(np.quantile(vals, 0.95)),
            "win_count": int(wins[th]),
            "win_rate": float(wins[th] / N_BOOT),
            "n": int(n),
        })

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

print("BOOTSTRAP SUMMARY")
for subset in subsets:
    print("=" * 100)
    print(subset)
    x = res[res["subset"].eq(subset)].sort_values(
        ["win_rate", "mean_macro_f1"], ascending=False
    )
    print(x.to_string(index=False))

print("Saved:", OUT)
