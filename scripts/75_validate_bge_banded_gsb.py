from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(".")
VAL = ROOT / "data/processed/v15_bge_reranker_validation_scores.parquet"
OUT = ROOT / "reports/manual_review/v15_bge_banded_gsb_validation.csv"

LOW = 0.45
HIGH = 0.80

VARIANTS = [
    ("bge_random_best_w020_t016", 0.20, 0.16),
    ("bge_manual_best_w030_t034", 0.30, 0.34),
    ("bge_w020_t018", 0.20, 0.18),
    ("bge_w020_t020", 0.20, 0.20),
    ("bge_w030_t026", 0.30, 0.26),
    ("bge_w030_t030", 0.30, 0.30),
    ("bge_w040_t032", 0.40, 0.32),
    ("bge_w040_t034", 0.40, 0.34),
    ("bge_w050_t040", 0.50, 0.40),
]

SAFE_BRANDS = [
    "nike","adidas","puma","reebok","vans","skechers","new balance","converse",
    "apple","samsung","xiaomi","huawei","lenovo","asus","acer","hp","logitech",
    "sony","philips","dyson","beko","arcelik","bosch","siemens","tefal","fakir",
    "vestel","karaca","korkmaz","arzum","ikea","english home","madame coco",
    "cerave","la roche posay","bioderma","avene","vichy","neutrogena","nivea",
    "maybelline","loreal","l oreal","golden rose","flormar","clinique",
    "defacto","koton","lc waikiki","zara","bershka","stradivarius","mango",
    "citizen","casio","seiko","daniel klein","calvin klein","tommy hilfiger",
    "hot wheels",
]

ALIASES = {
    "apple":["iphone","ipad","macbook","airpods","ios"],
    "samsung":["galaxy"],
    "xiaomi":["redmi","poco"],
    "nike":["jordan","air max"],
    "loreal":["l oreal"],
    "l oreal":["loreal"],
    "la roche posay":["roche posay"],
    "hot wheels":["hotwheels"],
}

COMPAT_WORDS = [
    "uyumlu","kilif","kılıf","sarj","şarj","kablo","adapter","adaptor",
    "adaptör","ekran koruyucu","koruyucu cam","yedek parca","yedek parça",
    "kapak","kayis","kayış"
]

INTRINSIC_WORDS = [
    "ayakkabi","ayakkabı","bot","terlik","sneaker","tisort","t shirt",
    "sweatshirt","ceket","mont","gomlek","gömlek","pantolon","elbise",
    "canta","çanta","saat","parfum","parfüm","krem","serum","sampuan",
    "şampuan","ruj","maskara","fondoten","oyuncak","bebek","dolap",
    "masa","sandalye","hali","halı","nevresim","forma"
]

def norm(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_series(s):
    tr = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    return (
        s.fillna("").astype(str).str.lower().str.translate(tr)
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

def word_contains(series, words):
    words = [re.escape(norm(w)) for w in words if norm(w)]
    pat = r"(?:^| )(?:%s)(?: |$)" % "|".join(words)
    return series.str.contains(pat, regex=True, na=False)

def any_contains(series, words):
    words = [re.escape(norm(w)) for w in words if norm(w)]
    return series.str.contains("|".join(words), regex=True, na=False)

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return {
        "macro_f1": f1_score(y, p, average="macro", labels=[0,1]),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y, p, pos_label=0, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "pred_pos_ratio": float(p.mean()),
        "true_pos_ratio": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

SAFE_BRANDS = sorted(set(norm(x) for x in SAFE_BRANDS), key=len, reverse=True)
ALIASES = {norm(k): [norm(v) for v in vals] for k, vals in ALIASES.items()}

df = pd.read_parquet(VAL)
df = df[df["assistant_label"].isin([0,1,"0","1"])].copy()
df["assistant_label"] = df["assistant_label"].astype(int)

q = norm_series(df["query"])
item_gender_text = norm_series(
    df["gender"].fillna("").astype(str) + " " +
    df["title"].fillna("").astype(str) + " " +
    df["category"].fillna("").astype(str)
)
item_text = norm_series(
    df["title"].fillna("").astype(str) + " " +
    df["category"].fillna("").astype(str) + " " +
    df["attributes"].fillna("").astype(str) + " " +
    df["brand"].fillna("").astype(str)
)
item_brand = norm_series(df["brand"].fillna("").astype(str))

q_male = word_contains(q, ["erkek","bay","men","male"])
q_female = word_contains(q, ["kadin","bayan","kiz","women","woman","female"])
item_male = word_contains(item_gender_text, ["erkek","bay","men","male"])
item_female = word_contains(item_gender_text, ["kadin","bayan","kiz","women","woman","female"])

gender_mismatch = ((q_male & ~q_female & item_female & ~item_male) |
                   (q_female & ~q_male & item_male & ~item_female))

safe_brand_mismatch = pd.Series(False, index=df.index)
for brand in SAFE_BRANDS:
    q_has = word_contains(q, [brand])
    if not q_has.any():
        continue
    item_brand_match = item_brand.eq(brand) | item_brand.str.contains(re.escape(brand), regex=True, na=False)
    text_match = word_contains(item_text, [brand] + ALIASES.get(brand, []))
    safe_brand_mismatch = safe_brand_mismatch | (q_has & ~item_brand_match & ~text_match)

has_compat = any_contains(q + " " + item_text, COMPAT_WORDS)
has_intrinsic = any_contains(q + " " + item_text, INTRINSIC_WORDS)
final_filter = (gender_mismatch | (safe_brand_mismatch & has_intrinsic & ~has_compat)).to_numpy()

rows = []

for label_set, part_idx in df.groupby("label_set").groups.items():
    part = df.loc[part_idx]
    y = part["assistant_label"].to_numpy()
    v5 = part["proba_avg"].to_numpy()
    bge = part["bge_sigmoid"].to_numpy()
    filt = final_filter[df.index.get_indexer(part.index)]

    for name, w, th in VARIANTS:
        pred = np.zeros(len(part), dtype=np.int8)
        pred[v5 > HIGH] = 1

        mid = (v5 >= LOW) & (v5 <= HIGH)
        blend = w * v5 + (1.0 - w) * bge
        pred[mid] = (blend[mid] >= th).astype(np.int8)

        pred[filt & (pred == 1)] = 0

        row = {"label_set": label_set, "variant": name, "w_v5": w, "threshold": th, "n": len(part)}
        row.update(metrics(y, pred))
        rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

print(res.sort_values(["label_set", "macro_f1"], ascending=[True, False]).to_string(index=False))

print("\nAGGREGATE")
agg = (
    res.groupby(["variant","w_v5","threshold"])
    .agg(mean_macro=("macro_f1","mean"), min_macro=("macro_f1","min"), mean_precision=("precision","mean"), mean_recall=("recall","mean"), sets=("label_set","nunique"))
    .reset_index()
    .sort_values(["min_macro","mean_macro"], ascending=False)
)
print(agg.to_string(index=False))
print("saved:", OUT)
