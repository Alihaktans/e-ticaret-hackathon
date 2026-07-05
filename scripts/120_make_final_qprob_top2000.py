from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")

sample_path = ROOT / "data/raw/sample_submission.csv"
anchor_path = ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv"
swaps_path = ROOT / "reports/manual_review/v33_perfected_all_swaps_scored.csv"

out_main = ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv"
out_candidate = ROOT / "submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv"

sample = pd.read_csv(sample_path)
sample["id"] = sample["id"].astype(str)

anchor = pd.read_csv(anchor_path)
anchor["id"] = anchor["id"].astype(str)

if not anchor["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
    raise RuntimeError("anchor id order mismatch")

swaps = pd.read_csv(swaps_path)
swaps["id_add"] = swaps["id_add"].astype(str)
swaps["id_drop"] = swaps["id_drop"].astype(str)

pred = anchor["prediction"].astype(np.int8).to_numpy().copy()
id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])

top = swaps.sort_values("qprob", ascending=False).head(2000).copy()

add_idx = top["id_add"].map(id_to_idx)
drop_idx = top["id_drop"].map(id_to_idx)

if add_idx.isna().any() or drop_idx.isna().any():
    raise RuntimeError("swap id not found in sample")

pred[add_idx.astype(int).to_numpy()] = 1
pred[drop_idx.astype(int).to_numpy()] = 0

out = pd.DataFrame({
    "id": sample["id"],
    "prediction": pred.astype(np.int8),
})

out.to_csv(out_candidate, index=False)
out.to_csv(out_main, index=False)

print("saved:", out_main)
print("shape:", out.shape)
print(out["prediction"].value_counts())
print("pos_ratio:", out["prediction"].mean())
print("id_order_ok:", out["id"].astype(str).equals(sample["id"].astype(str)))
print("diff_vs_v24:", int((out["prediction"].to_numpy() != anchor["prediction"].astype(np.int8).to_numpy()).sum()))
print("qprob_min:", float(top["qprob"].min()))
print("qprob_mean:", float(top["qprob"].mean()))
