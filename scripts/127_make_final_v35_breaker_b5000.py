from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")

sample_path = ROOT / "data/raw/sample_submission.csv"

candidate_paths = [
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv",
    ROOT / "submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv",
]

v24_path = ROOT / "submissions/FINAL_MAIN_v24_swap_rE_tE_b6500.csv"
v33_path = ROOT / "submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv"

out_path = ROOT / "submissions/FINAL_MAIN_v35_BREAKER_b5000.csv"

cand_path = None
for p in candidate_paths:
    if p.exists():
        cand_path = p
        break

if cand_path is None:
    raise FileNotFoundError("b5000 candidate bulunamadı")

sample = pd.read_csv(sample_path)
cand = pd.read_csv(cand_path)

sample["id"] = sample["id"].astype(str)
cand["id"] = cand["id"].astype(str)

if not cand["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
    raise RuntimeError("id_order mismatch")

cand["prediction"] = cand["prediction"].astype(np.int8)
cand.to_csv(out_path, index=False)

print("saved:", out_path)
print("source:", cand_path)
print("shape:", cand.shape)
print(cand["prediction"].value_counts())
print("pos_ratio:", cand["prediction"].mean())
print("id_order_ok:", cand["id"].astype(str).equals(sample["id"].astype(str)))

if v24_path.exists():
    v24 = pd.read_csv(v24_path)
    print("diff_vs_v24:", int((cand["prediction"].to_numpy() != v24["prediction"].astype(np.int8).to_numpy()).sum()))

if v33_path.exists():
    v33 = pd.read_csv(v33_path)
    print("diff_vs_v33_qprob2000:", int((cand["prediction"].to_numpy() != v33["prediction"].astype(np.int8).to_numpy()).sum()))
