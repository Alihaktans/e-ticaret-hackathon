from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")

FEATURES = ROOT / "data/processed/v17_query_level_features.parquet"
SAMPLE = ROOT / "data/raw/sample_submission.csv"

LABEL_FILES = {
    "random_clean_v2": ROOT / "reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv",
    "manual_v1": ROOT / "reports/manual_review/manual_review_set_v1_assistant_labeled.csv",
    "review_v13_vs_v5": ROOT / "reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv",
    "review_v15_vs_v13": ROOT / "reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv",
}

OUT_EVAL = ROOT / "reports/manual_review/v17p2_query_rules_eval.csv"
OUT_AGG = ROOT / "reports/manual_review/v17p2_query_rules_aggregate.csv"
OUT_FULL = ROOT / "reports/manual_review/v17p2_query_rules_full_summary.csv"

def clean_name(x):
    return (
        x.replace(".", "p")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace("<", "lt")
        .replace(">", "gt")
    )

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
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

print("loading features...")
df = pd.read_parquet(FEATURES)
df["id"] = df["id"].astype(str)

sample = pd.read_csv(SAMPLE, usecols=["id"])
sample["id"] = sample["id"].astype(str)
assert df["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True))

ids = df["id"].to_numpy()

pred_v13 = df["pred_v13"].astype(np.int8).to_numpy()
pred_v15 = df["pred_v15"].astype(np.int8).to_numpy()
pred_v16 = df["pred_v16"].astype(np.int8).to_numpy()

score = df["bge_blend_score"].astype("float32").to_numpy()
bge = df["bge_rank_score"].astype("float32").to_numpy()
ce = df["ce_rank_score"].astype("float32").to_numpy()
v5 = df["proba_avg"].astype("float32").to_numpy()

rank = df["bge_blend_score_rank"].astype("float32").to_numpy()
rank_bge = df["bge_rank_score_rank"].astype("float32").to_numpy()
rank_v5 = df["proba_avg_rank"].astype("float32").to_numpy()

brand_q = df["query_has_brand"].astype(bool).to_numpy()
specific_q = df["query_is_specific"].astype(bool).to_numpy()
generic_q = df["query_is_generic"].astype(bool).to_numpy()

term_count = df["candidate_count"].astype("int16").to_numpy()

variants = {}

def add_variant(name, pred):
    variants[name] = pred.astype(np.int8)

add_variant("v13_public_0p77", pred_v13.copy())
add_variant("v15_full_bge", pred_v15.copy())
add_variant("v16_current", pred_v16.copy())

def make_rule_variant(base, base_name, cap_brand, cap_spec, cap_generic, min_brand, min_spec, min_generic, add_mode):
    pred = base.copy()

    # Query-level removal: markalı/spesifik query'lerde düşük rank/score pozitifleri kes.
    remove_brand = (
        (pred == 1)
        & brand_q
        & ((rank > cap_brand) | (score < min_brand))
    )

    remove_spec = (
        (pred == 1)
        & (~brand_q)
        & specific_q
        & ((rank > cap_spec) | (score < min_spec))
    )

    remove_generic = (
        (pred == 1)
        & generic_q
        & ((rank > cap_generic) | (score < min_generic))
    )

    pred[remove_brand | remove_spec | remove_generic] = 0

    # Çok seçici addition. Sadece v13'ün kaçırdığı ama BGE/V5 birlikte yüksek gördüğü ürünler.
    if add_mode == "none":
        pass
    elif add_mode == "safe":
        add = (
            (pred == 0)
            & (pred_v13 == 0)
            & (rank <= 5)
            & (rank_v5 <= 12)
            & (score >= 0.55)
            & (bge >= 0.45)
        )
        pred[add] = 1
    elif add_mode == "safer":
        add = (
            (pred == 0)
            & (pred_v13 == 0)
            & (rank <= 3)
            & (rank_v5 <= 8)
            & (score >= 0.65)
            & (bge >= 0.55)
        )
        pred[add] = 1

    name = (
        f"v17p2_{base_name}"
        f"_cb{cap_brand}_cs{cap_spec}_cg{cap_generic}"
        f"_mb{min_brand}_ms{min_spec}_mg{min_generic}"
        f"_add_{add_mode}"
    )
    add_variant(name, pred)

bases = {
    "base_v13": pred_v13,
    "base_v16": pred_v16,
}

for base_name, base in bases.items():
    for cap_brand in [6, 10, 15, 25]:
        for cap_spec in [12, 20, 30, 45]:
            for cap_generic in [35, 50, 75, 999]:
                for min_brand in [0.12, 0.18, 0.25, 0.32]:
                    for min_spec in [0.10, 0.15, 0.20]:
                        for min_generic in [0.05, 0.08, 0.12]:
                            # Çok fazla variant çıkmasın diye aşırı kombinasyonları ele.
                            if cap_brand <= 10 and min_brand >= 0.25 and cap_spec <= 12:
                                continue
                            if cap_generic < 50 and min_generic >= 0.12:
                                continue

                            make_rule_variant(
                                base=base.copy(),
                                base_name=base_name,
                                cap_brand=cap_brand,
                                cap_spec=cap_spec,
                                cap_generic=cap_generic,
                                min_brand=min_brand,
                                min_spec=min_spec,
                                min_generic=min_generic,
                                add_mode="none",
                            )

# Az sayıda addition'lı varyant
for base_name, base in bases.items():
    for add_mode in ["safe", "safer"]:
        make_rule_variant(
            base=base.copy(),
            base_name=base_name,
            cap_brand=15,
            cap_spec=30,
            cap_generic=75,
            min_brand=0.18,
            min_spec=0.12,
            min_generic=0.05,
            add_mode=add_mode,
        )
        make_rule_variant(
            base=base.copy(),
            base_name=base_name,
            cap_brand=25,
            cap_spec=45,
            cap_generic=999,
            min_brand=0.12,
            min_spec=0.10,
            min_generic=0.05,
            add_mode=add_mode,
        )

print("variants:", len(variants))

# Full summary
full_rows = []
for name, pred in variants.items():
    full_rows.append({
        "variant": name,
        "ones": int(pred.sum()),
        "pos_ratio": float(pred.mean()),
        "diff_vs_v13": int((pred != pred_v13).sum()),
        "diff_vs_v13_ratio": float((pred != pred_v13).mean()),
        "diff_vs_v16": int((pred != pred_v16).sum()),
        "diff_vs_v16_ratio": float((pred != pred_v16).mean()),
    })

full = pd.DataFrame(full_rows)
full.to_csv(OUT_FULL, index=False)

# Labels
print("loading labels...")
labs = []
for label_set, path in LABEL_FILES.items():
    if not path.exists():
        print("missing:", path)
        continue

    lab = pd.read_csv(path)
    lab["id"] = lab["id"].astype(str)
    lab = lab[lab["assistant_label"].isin([0, 1, "0", "1"])].copy()
    lab["assistant_label"] = lab["assistant_label"].astype(int)
    lab["label_set"] = label_set
    labs.append(lab)

all_labels = pd.concat(labs, ignore_index=True)

id_to_idx = pd.Series(np.arange(len(ids)), index=ids)

eval_rows = []

for label_set, lab in all_labels.groupby("label_set"):
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
        subsets["high_medium_clean"] = lab[(nr == 0) & conf.isin(["high", "medium"])].copy()
        subsets["high_clean"] = lab[(nr == 0) & conf.eq("high")].copy()

    for subset_name, part in subsets.items():
        if len(part) < 30 or part["assistant_label"].nunique() < 2:
            continue

        idx = part["id"].map(id_to_idx)
        ok = idx.notna()
        part = part.loc[ok].copy()
        idx = idx.loc[ok].astype(int).to_numpy()
        y = part["assistant_label"].to_numpy()

        for name, pred in variants.items():
            p = pred[idx]
            row = {
                "label_set": label_set,
                "subset": subset_name,
                "eval_key": f"{label_set}_{subset_name}",
                "variant": name,
                "n": len(part),
            }
            row.update(metrics(y, p))
            eval_rows.append(row)

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT_EVAL, index=False)

# Aggregate scoring
# Ana validation setlere daha fazla ağırlık, change-review'a kontrollü ağırlık.
weights = {
    "random_clean_v2_all": 0.30,
    "random_clean_v2_clean": 0.30,
    "manual_v1_clean": 0.30,
    "manual_v1_high_clean": 0.20,
    "manual_v1_all": 0.15,
    "review_v15_vs_v13_clean": 0.15,
    "review_v15_vs_v13_high_medium_clean": 0.15,
    "review_v13_vs_v5_clean": 0.10,
    "review_v13_vs_v5_high_medium_clean": 0.10,
}

agg_rows = []
for name, g in eval_df.groupby("variant"):
    score = 0.0
    wsum = 0.0
    vals = []

    for _, r in g.iterrows():
        w = weights.get(r["eval_key"], 0.0)
        if w > 0:
            score += w * r["macro_f1"]
            wsum += w
            vals.append(r["macro_f1"])

    if wsum == 0:
        continue

    main = g[g["eval_key"].isin([
        "random_clean_v2_all",
        "manual_v1_clean",
        "manual_v1_high_clean",
        "review_v15_vs_v13_high_medium_clean",
        "review_v15_vs_v13_clean",
    ])]

    agg_rows.append({
        "variant": name,
        "weighted_macro": score / wsum,
        "min_macro_used": float(np.min(vals)) if vals else np.nan,
        "mean_macro_used": float(np.mean(vals)) if vals else np.nan,
        "main_min_macro": float(main["macro_f1"].min()) if len(main) else np.nan,
        "main_mean_macro": float(main["macro_f1"].mean()) if len(main) else np.nan,
        "mean_precision": float(g["precision"].mean()),
        "mean_recall": float(g["recall"].mean()),
        "mean_pred_pos_ratio": float(g["pred_pos_ratio"].mean()),
        "eval_count": int(len(g)),
    })

agg = pd.DataFrame(agg_rows)

agg = agg.merge(full, on="variant", how="left")
agg = agg.sort_values(
    ["weighted_macro", "main_min_macro", "main_mean_macro"],
    ascending=False,
)

agg.to_csv(OUT_AGG, index=False)

print("\nAGG TOP")
print(agg.head(40).to_string(index=False))

# Top 8 submission yaz
top_names = agg.head(8)["variant"].tolist()

# Baseline'ları da yaz
for b in ["v13_public_0p77", "v15_full_bge", "v16_current"]:
    if b not in top_names:
        top_names.append(b)

for name in top_names:
    out = ROOT / "submissions" / f"FINAL_CANDIDATE_v17p2_{clean_name(name)}.csv"
    pd.DataFrame({"id": ids, "prediction": variants[name]}).to_csv(out, index=False)
    print("saved:", out)

print("saved:", OUT_EVAL)
print("saved:", OUT_AGG)
print("saved:", OUT_FULL)
