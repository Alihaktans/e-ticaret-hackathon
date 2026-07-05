from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

V13 = ROOT / "submissions/FINAL_MAIN_v13_aggressive_w025_t0275_GSB.csv"
V15 = ROOT / "submissions/FINAL_MAIN_v15_bge_w020_t018_GSB.csv"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

V5S = ROOT / "data/processed/v5_e5base_full900_test_proba.parquet"
BGE = ROOT / "data/processed/v15_bge_reranker_candidate_scores.parquet"
CE = ROOT / "data/processed/v13_cross_encoder_candidate_scores.parquet"

CHANGE_LABELS = ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "change_review_600": CHANGE_LABELS,
}

OUT_EVAL = ROOT / "reports/manual_review/v16_bge_hybrid_eval.csv"
OUT_FULL = ROOT / "reports/manual_review/v16_bge_hybrid_full_summary.csv"
OUT_AGG = ROOT / "reports/manual_review/v16_bge_hybrid_aggregate.csv"

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1]),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def clean_name(x):
    return (
        x.replace(".", "p")
         .replace("<=", "le")
         .replace(">=", "ge")
         .replace(" ", "_")
         .replace("/", "_")
    )

print("loading submissions...")
v13 = pd.read_csv(V13)
v15 = pd.read_csv(V15)
sample = pd.read_csv(SAMPLE)

for d in [v13, v15, sample]:
    d["id"] = d["id"].astype(str)

assert v13["id"].equals(sample["id"])
assert v15["id"].equals(sample["id"])

ids = v13["id"].to_numpy()
v13_pred = v13["prediction"].astype(np.int8).to_numpy()
v15_pred = v15["prediction"].astype(np.int8).to_numpy()

added_by_v15 = (v13_pred == 0) & (v15_pred == 1)
removed_by_v15 = (v13_pred == 1) & (v15_pred == 0)

print("added_by_v15:", int(added_by_v15.sum()))
print("removed_by_v15:", int(removed_by_v15.sum()))

v5s = pd.read_parquet(V5S, columns=["id", "proba_avg"])
v5s["id"] = v5s["id"].astype(str)

bge = pd.read_parquet(BGE, columns=["id", "bge_sigmoid", "bge_blend_w020"])
bge["id"] = bge["id"].astype(str)

ce = pd.read_parquet(CE, columns=["id", "cross_sigmoid"])
ce["id"] = ce["id"].astype(str)

score = v5s.merge(bge, on="id", how="left").merge(ce, on="id", how="left")
assert score["id"].equals(v13["id"])

proba = score["proba_avg"].to_numpy()
bge_sig = score["bge_sigmoid"].fillna(-1).to_numpy()
bge_blend = score["bge_blend_w020"].fillna(-1).to_numpy()
ce_sig = score["cross_sigmoid"].fillna(-1).to_numpy()

variants = {}

def add_variant(name, pred):
    variants[name] = pred.astype(np.int8)

# Baselines
add_variant("v13_base_public_0p77", v13_pred.copy())
add_variant("v15_full_bge_w020_t018", v15_pred.copy())

# v13 taban: v15'in kaldırdıklarını uygula, ekleme yapma
pred = v13_pred.copy()
pred[removed_by_v15] = 0
add_variant("v16_remove_all_add_none", pred)

# v13 taban: removal hepsi, addition sadece yüksek BGE ise
for th in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = v13_pred.copy()
    pred[removed_by_v15] = 0
    pred[added_by_v15 & (bge_sig >= th)] = 1
    add_variant(f"v16_remove_all_add_bge_ge_{th}", pred)

# v13 taban: removal hepsi, addition sadece yüksek BGE blend ise
for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
    pred = v13_pred.copy()
    pred[removed_by_v15] = 0
    pred[added_by_v15 & (bge_blend >= th)] = 1
    add_variant(f"v16_remove_all_add_blend_ge_{th}", pred)

# Daha temkinli removal: sadece BGE gerçekten düşükse kaldır
for rem_th in [0.05, 0.10, 0.20, 0.30]:
    for add_th in [0.60, 0.70, 0.80]:
        pred = v13_pred.copy()
        pred[removed_by_v15 & (bge_sig <= rem_th)] = 0
        pred[added_by_v15 & (bge_sig >= add_th)] = 1
        add_variant(f"v16_rem_bge_le_{rem_th}_add_bge_ge_{add_th}", pred)

# Addition için hem BGE hem V5 yüksek olsun
for bge_th in [0.60, 0.70, 0.80]:
    for v5_th in [0.60, 0.65, 0.70]:
        pred = v13_pred.copy()
        pred[removed_by_v15] = 0
        pred[added_by_v15 & (bge_sig >= bge_th) & (proba >= v5_th)] = 1
        add_variant(f"v16_remove_all_add_bge_ge_{bge_th}_v5_ge_{v5_th}", pred)

# BGE ve MiniLM CE aynı anda destekliyorsa ekle
for bge_th in [0.50, 0.60, 0.70]:
    for ce_th in [0.10, 0.20, 0.30]:
        pred = v13_pred.copy()
        pred[removed_by_v15] = 0
        pred[added_by_v15 & (bge_sig >= bge_th) & (ce_sig >= ce_th)] = 1
        add_variant(f"v16_remove_all_add_bge_ge_{bge_th}_ce_ge_{ce_th}", pred)

full_rows = []
for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v13": int((pred != v13_pred).sum()),
        "diff_vs_v13_ratio": float((pred != v13_pred).mean()),
        "diff_vs_v15": int((pred != v15_pred).sum()),
        "diff_vs_v15_ratio": float((pred != v15_pred).mean()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

print("\nFULL SUMMARY TOP")
print(full.sort_values("diff_vs_v13").head(30).to_string(index=False))

eval_rows = []
id_to_idx = pd.Series(np.arange(len(ids)), index=ids)

for label_name, path in LABEL_FILES.items():
    if not path.exists():
        print("missing label file:", path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)

    subsets = {"all": lab}

    if "needs_recheck" in lab.columns:
        try:
            nr = lab["needs_recheck"].fillna(0).astype(int)
            subsets["clean"] = lab[nr == 0].copy()
        except Exception:
            pass

    if "assistant_confidence" in lab.columns and "needs_recheck" in lab.columns:
        conf = lab["assistant_confidence"].astype(str).str.lower()
        nr = lab["needs_recheck"].fillna(0).astype(int)
        subsets["high_medium_clean"] = lab[(nr == 0) & (conf.isin(["high", "medium"]))].copy()
        subsets["high_clean"] = lab[(nr == 0) & (conf.eq("high"))].copy()

    for subset_name, part in subsets.items():
        if len(part) < 30 or part["assistant_label"].nunique() < 2:
            continue

        mapped = part["id"].map(id_to_idx)
        ok = mapped.notna()
        part = part.loc[ok].copy()
        idx = mapped.loc[ok].astype(int).to_numpy()
        y = part["assistant_label"].to_numpy()

        for var_name, pred in variants.items():
            p = pred[idx]
            row = {
                "label_set": label_name,
                "subset": subset_name,
                "eval_key": f"{label_name}_{subset_name}",
                "variant": var_name,
                "n": len(part),
            }
            row.update(metrics(y, p))
            eval_rows.append(row)

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT_EVAL, index=False)

print("\nTOP CHANGE REVIEW")
print(
    eval_df[eval_df["eval_key"].str.contains("change_review_600")]
    .sort_values("macro_f1", ascending=False)
    .head(40)
    .to_string(index=False)
)

# Aggregate: public'e en yakın karar için ana setler
use_keys = [
    "random_clean_v2_all",
    "manual_v1_clean",
    "manual_v1_high_clean",
    "change_review_600_high_medium_clean",
    "change_review_600_clean",
]

agg_src = eval_df[eval_df["eval_key"].isin(use_keys)].copy()

agg = (
    agg_src.groupby("variant")
    .agg(
        mean_macro=("macro_f1", "mean"),
        min_macro=("macro_f1", "min"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_pred_pos_ratio=("pred_pos_ratio", "mean"),
        sets=("eval_key", "nunique"),
    )
    .reset_index()
    .sort_values(["min_macro", "mean_macro"], ascending=False)
)

agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP")
print(agg.head(40).to_string(index=False))

# Sadece aggregate top 5 için submission yaz
top_variants = agg.head(5)["variant"].tolist()
if "v13_base_public_0p77" not in top_variants:
    top_variants.append("v13_base_public_0p77")

for name in top_variants:
    pred = variants[name]
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v16_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)
    print("saved submission:", out)

print("\nsaved:", OUT_EVAL)
print("saved:", OUT_FULL)
print("saved:", OUT_AGG)
