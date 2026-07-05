"""Independent local-LLM audit of stratified V71 consensus swaps."""
from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(".")
FEATURES = ROOT / "data/processed/v70_trendyol_embedding/v70_compact_rank_features.parquet"
BASE = ROOT / "submissions/FINAL_MAIN_v22_vote_full_risky_big_v2.csv"
OUT_JSONL = ROOT / "reports/manual_review/v71_qwen25_consensus_audit.jsonl"
OUT_CSV = ROOT / "reports/manual_review/v71_qwen25_consensus_audit.csv"
OUT_SUMMARY = ROOT / "reports/experiments/v71_qwen25_consensus_audit_summary.json"
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def proposals(frame: pd.DataFrame, base: np.ndarray) -> list[tuple]:
    trend = frame["trendyol_qpct"].to_numpy(np.float32)
    lex = frame["lex_qpct"].to_numpy(np.float32)
    blend = .5 * (trend + lex)
    out = []
    for term, idx in frame.groupby("term_id", sort=False).indices.items():
        idx = np.asarray(idx); pos = idx[base[idx] == 1]; neg = idx[base[idx] == 0]
        add = neg[np.argsort(-blend[neg])]; drop = pos[np.argsort(blend[pos])]
        for ai, di in zip(add, drop):
            dt, dl = float(trend[ai]-trend[di]), float(lex[ai]-lex[di])
            if dt > 0 and dl > 0: out.append((min(dt,dl), .5*(dt+dl), int(ai), int(di), str(term)))
    out.sort(reverse=True)
    return out


def stratified_sample(items: list[tuple]) -> list[tuple]:
    rng = np.random.default_rng(20260702)
    bands = [(0, 5_000, 15, "top_5k"), (5_000, 15_000, 20, "5k_15k"),
             (15_000, 25_000, 25, "15k_25k"), (25_000, 50_000, 15, "25k_50k")]
    picked = []
    for lo, hi, n, band in bands:
        ix = rng.choice(np.arange(lo, min(hi, len(items))), size=n, replace=False)
        for i in ix: picked.append((int(i)+1, band, *items[int(i)]))
    return sorted(picked)


def load_item_text(ids: set[str]) -> pd.DataFrame:
    parts = []
    for c in pd.read_csv(ROOT / "data/raw/items.csv", chunksize=50_000, low_memory=False):
        x = c[c["item_id"].astype(str).isin(ids)]
        if len(x): parts.append(x)
    return pd.concat(parts).drop_duplicates("item_id").set_index("item_id")


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match: return {"parse_error": True}
    try: return json.loads(match.group())
    except Exception: return {"parse_error": True}


def main() -> None:
    f = pd.read_parquet(FEATURES)
    b = pd.read_csv(BASE)
    if not np.array_equal(f["id"].astype(str), b["id"].astype(str)): raise RuntimeError("alignment failure")
    props = proposals(f, b["prediction"].to_numpy(np.int8))
    picked = stratified_sample(props)
    rows = []
    for rank, band, weak, mean, ai, di, term in picked:
        rows.append({"proposal_rank": rank, "band": band, "weak_margin": weak, "mean_margin": mean,
                     "term_id": term, "id_add": f.iloc[ai].id, "item_id_add": f.iloc[ai].item_id,
                     "trend_add": f.iloc[ai].trendyol_qpct, "lex_add": f.iloc[ai].lex_qpct,
                     "id_drop": f.iloc[di].id, "item_id_drop": f.iloc[di].item_id,
                     "trend_drop": f.iloc[di].trendyol_qpct, "lex_drop": f.iloc[di].lex_qpct})
    review = pd.DataFrame(rows)
    terms = pd.read_csv(ROOT / "data/raw/terms.csv").drop_duplicates("term_id").set_index("term_id")["query"]
    item_ids = set(review.item_id_add.astype(str)) | set(review.item_id_drop.astype(str))
    products = load_item_text(item_ids)
    review["query"] = review.term_id.map(terms)
    for side in ["add", "drop"]:
        review[f"title_{side}"] = review[f"item_id_{side}"].map(products["title"])
        review[f"category_{side}"] = review[f"item_id_{side}"].map(products["category"])
        review[f"brand_{side}"] = review[f"item_id_{side}"].map(products["brand"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    system = ("Sen katı bir Türkçe e-ticaret arama alaka denetçisisin. Marka, model, sayı, beden, cinsiyet, yaş, "
              "ürün türü ve aksesuar/ana ürün ayrımına dikkat et. A ve B'yi birbirinden bağımsız puanla. "
              "Yalnızca tek satır geçerli JSON döndür.")
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    parsed = []; started = time.time()
    with OUT_JSONL.open("w", encoding="utf8") as stream:
        for i, r in review.iterrows():
            prompt = f"""Sorgu: {r['query']}
A başlık: {r['title_add']}
A kategori: {r['category_add']}
A marka: {r['brand_add']}
B başlık: {r['title_drop']}
B kategori: {r['category_drop']}
B marka: {r['brand_drop']}

Her ürünün sorguya alakasını 0=alakasız, 1=zayıf/kısmi, 2=alakalı, 3=tam eşleşme olarak puanla. Sonra hangisinin daha alakalı olduğunu seç. Şema: {{"a_score":0,"b_score":0,"preferred":"A|B|TIE","a_hard_conflict":false,"b_hard_conflict":false,"reason":"kısa gerekçe"}}"""
            chat = tokenizer.apply_chat_template([{"role":"system","content":system},{"role":"user","content":prompt}],
                                                 tokenize=False, add_generation_prompt=True)
            enc = tokenizer(chat, return_tensors="pt", truncation=True, max_length=768).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=110, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            obj = parse_json(answer); parsed.append(obj)
            rec = {**r.to_dict(), "answer": answer, "parsed": obj}
            stream.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); stream.flush()
            print(f"{i+1}/{len(review)} rank={r['proposal_rank']} {obj}", flush=True)
    for key in ["a_score", "b_score", "preferred", "a_hard_conflict", "b_hard_conflict", "reason", "parse_error"]:
        review["qwen_" + key] = [x.get(key) for x in parsed]
    review.to_csv(OUT_CSV, index=False)
    summary = []
    for band, x in review.groupby("band", sort=False):
        valid = x.qwen_parse_error.ne(True)
        z = x[valid]
        summary.append({"band": band, "n": len(x), "valid": int(valid.sum()),
                        "preferred_a_rate": float(z.qwen_preferred.eq("A").mean()),
                        "a_score_gt_b_rate": float((pd.to_numeric(z.qwen_a_score)>pd.to_numeric(z.qwen_b_score)).mean()),
                        "a_hard_conflict_rate": float(z.qwen_a_hard_conflict.eq(True).mean())})
    result = {"model": MODEL, "rows": len(review), "elapsed_minutes": (time.time()-started)/60,
              "bands": summary, "warning": "Qwen is a weak independent auditor, not ground truth."}
    OUT_SUMMARY.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
