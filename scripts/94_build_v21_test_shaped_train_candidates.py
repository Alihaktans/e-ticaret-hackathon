from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import re
import time
import numpy as np
import pandas as pd

ROOT = Path(".")

TRAIN_PAIRS = ROOT / "data/raw/training_pairs.csv"
TERMS = ROOT / "data/raw/terms.csv"
ITEMS = ROOT / "data/raw/items.csv"

OUT = ROOT / "data/processed/v21_train_candidates.parquet"
REPORT = ROOT / "reports/manual_review/v21_train_candidates_report.csv"
SOURCE_REPORT = ROOT / "reports/manual_review/v21_train_candidates_source_report.csv"

SEED = 2026
rng = np.random.default_rng(SEED)

MAX_POSTINGS_PER_TOKEN = 6000
LEX_TOP_PER_TERM = 90
SAME_ROOT_SAMPLE_PER_TERM = 45
RANDOM_SAMPLE_PER_TERM = 20
MAX_NEG_PER_TERM = 130
MAX_POS_PER_TERM = 999999

STOPWORDS = {
    "ve", "ile", "icin", "için", "bir", "bu", "de", "da", "the", "and", "or",
    "in", "on", "of", "to", "for", "a", "an"
}

TR_MAP = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def toks(x):
    out = []
    for t in norm(x).split():
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out

def root_category(x):
    s = "" if pd.isna(x) else str(x)
    for sep in [">", "/", "|"]:
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()

def fold_of_term(term_id):
    h = hashlib.md5(str(term_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 5

def add_candidate(term_id, item_idx, label, source, fold, term_list, item_list, label_list, source_list, fold_list):
    term_list.append(term_id)
    item_list.append(item_ids[item_idx])
    label_list.append(label)
    source_list.append(source)
    fold_list.append(fold)

t0 = time.time()

print("loading raw data...")
train = pd.read_csv(TRAIN_PAIRS)
terms = pd.read_csv(TERMS)
items = pd.read_csv(ITEMS, low_memory=False)

train["term_id"] = train["term_id"].astype(str)
train["item_id"] = train["item_id"].astype(str)
terms["term_id"] = terms["term_id"].astype(str)
items["item_id"] = items["item_id"].astype(str)

if "label" not in train.columns:
    train["label"] = 1

train = train[train["label"].astype(int) == 1].copy()

print("train positives:", len(train))
print("unique train terms:", train["term_id"].nunique())

print("normalizing terms...")
terms["query_norm"] = terms["query"].map(norm)
terms["query_tokens"] = terms["query"].map(lambda x: list(dict.fromkeys(toks(x))))

train_terms = terms[terms["term_id"].isin(set(train["term_id"]))].copy()
query_tokens_by_term = dict(zip(train_terms["term_id"], train_terms["query_tokens"]))

query_vocab = set()
for ts in train_terms["query_tokens"]:
    query_vocab.update(ts)

print("query vocab:", len(query_vocab))

print("normalizing items...")
for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    if c not in items.columns:
        items[c] = ""

items["root_category"] = items["category"].map(root_category)
items["title_norm"] = items["title"].map(norm)
items["category_norm"] = items["category"].map(norm)
items["brand_norm"] = items["brand"].map(norm)

item_ids = items["item_id"].to_numpy()
item_roots = items["root_category"].fillna("").astype(str).to_numpy()
n_items = len(items)

item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

print("items:", n_items)

print("building root index...")
root_to_idxs = {}
for root, g in items.groupby("root_category", sort=False):
    root_to_idxs[str(root)] = g.index.to_numpy(dtype=np.int32)

print("roots:", len(root_to_idxs))

print("building token inverted index...")
token_to_items = defaultdict(list)

for i, r in enumerate(items.itertuples(index=False)):
    title_norm = getattr(r, "title_norm")
    category_norm = getattr(r, "category_norm")
    brand_norm = getattr(r, "brand_norm")

    item_tokens = set(title_norm.split()) | set(category_norm.split()) | set(brand_norm.split())
    item_tokens = item_tokens & query_vocab

    for t in item_tokens:
        token_to_items[t].append(i)

    if (i + 1) % 200000 == 0:
        print(f"indexed items {i+1}/{n_items}")

print("raw token postings:", len(token_to_items))

print("capping token postings...")
token_to_arr = {}

for t, lst in token_to_items.items():
    arr = np.asarray(lst, dtype=np.int32)

    if len(arr) > MAX_POSTINGS_PER_TOKEN:
        arr = rng.choice(arr, size=MAX_POSTINGS_PER_TOKEN, replace=False).astype(np.int32)

    token_to_arr[t] = arr

token_to_items = None

posting_sizes = [len(v) for v in token_to_arr.values()]
print("tokens kept:", len(token_to_arr))
print("posting mean:", float(np.mean(posting_sizes)) if posting_sizes else 0)
print("posting p95:", float(np.quantile(posting_sizes, 0.95)) if posting_sizes else 0)
print("posting max:", int(np.max(posting_sizes)) if posting_sizes else 0)

print("building positives by term...")
pos_idx_by_term = defaultdict(list)
missing_pos_items = 0

for r in train.itertuples(index=False):
    iid = r.item_id
    idx = item_id_to_idx.get(iid)
    if idx is None:
        missing_pos_items += 1
        continue
    pos_idx_by_term[r.term_id].append(idx)

print("terms with positives:", len(pos_idx_by_term))
print("missing positive items:", missing_pos_items)

term_list = []
item_list = []
label_list = []
source_list = []
fold_list = []

source_counts = Counter()
term_counter = 0

all_item_indices = np.arange(n_items, dtype=np.int32)

print("generating v21 candidates...")

for term_id, pos_idxs_raw in pos_idx_by_term.items():
    term_counter += 1

    q_tokens = query_tokens_by_term.get(term_id, [])
    fold = fold_of_term(term_id)

    # positive dedupe
    pos_idxs = list(dict.fromkeys(pos_idxs_raw))
    pos_set = set(pos_idxs)

    if len(pos_idxs) > MAX_POS_PER_TERM:
        pos_idxs = list(rng.choice(np.asarray(pos_idxs, dtype=np.int32), size=MAX_POS_PER_TERM, replace=False))
        pos_set = set(pos_idxs)

    # add positives
    for pi in pos_idxs:
        add_candidate(term_id, pi, 1, "positive", fold, term_list, item_list, label_list, source_list, fold_list)
        source_counts["positive"] += 1

    seen = set(pos_idxs)
    neg_candidates = []

    # lexical candidates from query tokens
    score = Counter()
    for qt in q_tokens:
        arr = token_to_arr.get(qt)
        if arr is None:
            continue
        for idx in arr:
            if int(idx) not in pos_set:
                score[int(idx)] += 1

    if score:
        ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
        for idx, sc in ranked[:LEX_TOP_PER_TERM]:
            if idx not in seen:
                neg_candidates.append((idx, "lex_top"))
                seen.add(idx)

    # same root candidates based on positive products
    roots = list(dict.fromkeys([item_roots[i] for i in pos_idxs if str(item_roots[i]) != ""]))
    root_added = 0

    for root in roots[:4]:
        pool = root_to_idxs.get(str(root))
        if pool is None or len(pool) == 0:
            continue

        take_n = min(max(8, SAME_ROOT_SAMPLE_PER_TERM // max(1, len(roots[:4]))), len(pool))
        sampled = rng.choice(pool, size=take_n, replace=False)

        for idx in sampled:
            idx = int(idx)
            if idx not in seen:
                neg_candidates.append((idx, "same_root"))
                seen.add(idx)
                root_added += 1

    # query-token random lexical extra
    if score and len(neg_candidates) < MAX_NEG_PER_TERM:
        ranked_tail = [idx for idx, sc in sorted(score.items(), key=lambda x: (-x[1], x[0]))[LEX_TOP_PER_TERM:LEX_TOP_PER_TERM+300]]
        if ranked_tail:
            take_n = min(25, len(ranked_tail))
            sampled = rng.choice(np.asarray(ranked_tail, dtype=np.int32), size=take_n, replace=False)
            for idx in sampled:
                idx = int(idx)
                if idx not in seen:
                    neg_candidates.append((idx, "lex_tail"))
                    seen.add(idx)

    # random easy negatives
    if len(neg_candidates) < MAX_NEG_PER_TERM:
        sampled = rng.choice(all_item_indices, size=RANDOM_SAMPLE_PER_TERM * 3, replace=False)
        for idx in sampled:
            idx = int(idx)
            if idx not in seen:
                neg_candidates.append((idx, "random_easy"))
                seen.add(idx)
            if len([1 for _, s in neg_candidates if s == "random_easy"]) >= RANDOM_SAMPLE_PER_TERM:
                break

    # cap negatives per term, but keep source diversity
    if len(neg_candidates) > MAX_NEG_PER_TERM:
        lex = [x for x in neg_candidates if x[1] == "lex_top"]
        root = [x for x in neg_candidates if x[1] == "same_root"]
        tail = [x for x in neg_candidates if x[1] == "lex_tail"]
        rnd = [x for x in neg_candidates if x[1] == "random_easy"]

        selected = []
        selected.extend(lex[:70])
        selected.extend(root[:35])
        selected.extend(tail[:15])
        selected.extend(rnd[:10])

        if len(selected) < MAX_NEG_PER_TERM:
            remaining = [x for x in neg_candidates if x not in selected]
            selected.extend(remaining[:MAX_NEG_PER_TERM - len(selected)])

        neg_candidates = selected[:MAX_NEG_PER_TERM]

    for ni, src in neg_candidates:
        add_candidate(term_id, ni, 0, src, fold, term_list, item_list, label_list, source_list, fold_list)
        source_counts[src] += 1

    if term_counter % 1000 == 0:
        elapsed = time.time() - t0
        print(
            f"terms {term_counter}/{len(pos_idx_by_term)} | "
            f"rows {len(label_list):,} | elapsed {elapsed/60:.1f} min"
        )

print("building dataframe...")
df = pd.DataFrame({
    "term_id": term_list,
    "item_id": item_list,
    "label": np.asarray(label_list, dtype=np.int8),
    "candidate_source": source_list,
    "fold": np.asarray(fold_list, dtype=np.int8),
})

print("deduping...")
before = len(df)
df = df.drop_duplicates(["term_id", "item_id"], keep="first").reset_index(drop=True)
after = len(df)
print("dedup removed:", before - after)

# Positive collision guard: if any pair is known positive, force label=1
known_pos = set(zip(train["term_id"].astype(str), train["item_id"].astype(str)))
pair_tuples = list(zip(df["term_id"].astype(str), df["item_id"].astype(str)))
is_known_pos = np.fromiter((p in known_pos for p in pair_tuples), dtype=bool, count=len(df))
df.loc[is_known_pos, "label"] = 1
df.loc[is_known_pos, "candidate_source"] = "positive"

df["label"] = df["label"].astype(np.int8)
df["fold"] = df["fold"].astype(np.int8)

OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(OUT, index=False)

report = pd.DataFrame([{
    "rows": len(df),
    "positives": int((df["label"] == 1).sum()),
    "negatives": int((df["label"] == 0).sum()),
    "pos_ratio": float(df["label"].mean()),
    "unique_terms": int(df["term_id"].nunique()),
    "unique_items": int(df["item_id"].nunique()),
    "folds": ",".join(map(str, sorted(df["fold"].unique().tolist()))),
    "runtime_min": round((time.time() - t0) / 60, 3),
}])

report.to_csv(REPORT, index=False)

source_report = (
    df.groupby(["candidate_source", "label"])
    .size()
    .reset_index(name="rows")
    .sort_values(["label", "rows"], ascending=[False, False])
)

source_report.to_csv(SOURCE_REPORT, index=False)

print("\nREPORT")
print(report.to_string(index=False))

print("\nSOURCE REPORT")
print(source_report.to_string(index=False))

print("saved:", OUT)
print("report:", REPORT)
print("source report:", SOURCE_REPORT)
