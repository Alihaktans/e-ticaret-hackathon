from pathlib import Path
import os
import re
import json
import time
import hashlib
import unicodedata
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(".")
MODEL = "Qwen/Qwen2.5-3B-Instruct"

SAMPLE = ROOT / "data/raw/sample_submission.csv"
V60_CAP250 = ROOT / "submissions/final_candidates_v60/v60_public080_ultra_cap250.csv"
V65_POOL = ROOT / "reports/manual_review/review_v65_general_aggressive_pool.csv"

OUT_DIR = ROOT / "reports/manual_review"
SUB_DIR = ROOT / "submissions/final_candidates_v66"

OUT_JSONL = OUT_DIR / "v66_qwen25_intent_judgements.jsonl"
OUT_REVIEW = OUT_DIR / "review_v66_qwen25_intent_guarded_pool.csv"
OUT_SUMMARY = OUT_DIR / "v66_qwen25_intent_guarded_candidate_summary.csv"
OUT_SAVED = OUT_DIR / "v66_qwen25_intent_guarded_saved_candidates.csv"

TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

# Keep these generic. They are not product-specific allow/deny rules; they detect common query-intent failures.
MAIN_VS_ACCESSORY_WORDS = {
    "kilif", "kılıf", "stand", "tutucu", "aksesuar", "kordon", "kayis", "kayış",
    "yedek", "parca", "parça", "susu", "süsü", "kapak", "minder", "kulpu",
    "toz torbasi", "toz torbası",
}

QUERY_ACCESSORY_HINTS = MAIN_VS_ACCESSORY_WORDS | {
    "mousepad", "anahtarlik", "anahtarlık", "halkasi", "halkası", "paspas",
    "bıçakları", "bicaklari", "askisi", "askısı", "canta", "çanta", "çantası",
}

IMPORTANT_SPEC_CONTEXT = {
    "gb", "tb", "jant", "cm", "mm", "sinif", "sınıf", "hz", "mah", "w30",
    "iphone", "ipad", "matepad", "ford", "focus", "ps5", "ps", "playstation",
}


def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).translate(TR_MAP).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def find_col(df, names, contains=None):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    if contains:
        for pat in contains:
            for c in df.columns:
                if pat.lower() in c.lower():
                    return c
    return None


def short_hash(x):
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8]


def to_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "evet"}


def safe_num(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype("float64")


def pct01(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index, dtype="float64")
    s = s.fillna(s.median())
    if float(s.std()) < 1e-12:
        return pd.Series(0.5, index=s.index, dtype="float64")
    return s.rank(method="average", pct=True).astype("float64")


def get_text(row, col):
    if col and col in row.index and not pd.isna(row[col]):
        return str(row[col])
    return ""


def extract_json(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {
            "intent_winner": "UNCERTAIN",
            "a_exact_intent": False,
            "b_exact_intent": False,
            "a_keyword_bait": True,
            "b_keyword_bait": False,
            "a_constraint_violation": True,
            "b_constraint_violation": False,
            "confidence": 0.0,
            "intent_reason": "no_json:" + text[:180],
        }

    raw = m.group(0).replace("“", "\"").replace("”", "\"")
    try:
        obj = json.loads(raw)
    except Exception as e:
        # Try one mild repair.
        raw2 = raw.replace("'", "\"")
        try:
            obj = json.loads(raw2)
        except Exception:
            return {
                "intent_winner": "UNCERTAIN",
                "a_exact_intent": False,
                "b_exact_intent": False,
                "a_keyword_bait": True,
                "b_keyword_bait": False,
                "a_constraint_violation": True,
                "b_constraint_violation": False,
                "confidence": 0.0,
                "intent_reason": "json_parse_error:" + str(e) + ":" + raw[:180],
            }

    winner = str(obj.get("winner", obj.get("intent_winner", "UNCERTAIN"))).upper().strip()
    if winner not in {"A", "B", "TIE", "UNCERTAIN"}:
        if winner.startswith("A"):
            winner = "A"
        elif winner.startswith("B"):
            winner = "B"
        elif "TIE" in winner:
            winner = "TIE"
        else:
            winner = "UNCERTAIN"

    def b(key, default=False):
        val = obj.get(key, default)
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in {"true", "1", "yes", "evet"}

    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    if conf > 1.0:
        conf = conf / 100.0
    conf = max(0.0, min(1.0, conf))

    return {
        "intent_winner": winner,
        "a_exact_intent": b("a_exact_intent", False),
        "b_exact_intent": b("b_exact_intent", False),
        "a_keyword_bait": b("a_keyword_bait", False),
        "b_keyword_bait": b("b_keyword_bait", False),
        "a_constraint_violation": b("a_constraint_violation", False),
        "b_constraint_violation": b("b_constraint_violation", False),
        "confidence": conf,
        "intent_reason": str(obj.get("reason", obj.get("intent_reason", "")))[:700],
    }


def build_prompt(query, add_title, add_cat, drop_title, drop_cat):
    return f"""You are a very strict Turkish e-commerce search relevance judge.

Search query:
{query}

Product A = candidate to ADD:
title: {add_title}
category: {add_cat}

Product B = current positive item to DROP:
title: {drop_title}
category: {drop_cat}

Your job is NOT keyword matching. Your job is exact shopping-intent matching.

Strict interpretation rules:
1. Decide the exact product type requested by the query.
2. A product is NOT relevant just because the query word appears as decoration, print, theme, compatibility text, or a random keyword.
   Examples:
   - "köpek çantası" can mean a dog carrier/pet bag; a normal shoulder bag with a dog print is keyword bait unless query clearly asks for a printed bag.
   - "hafıza kartı / sd kart" is not the same as a USB flash drive.
   - "telefon / tablet / ps5" main product is not a case, holder, stand, wall mount, or accessory unless the query asks for kılıf/stand/aksesuar.
3. Brand/model/number/size/gender/age/class/vehicle/device constraints are hard constraints.
4. If both products match the exact intent, choose A only if A is clearly more specific or clearly better.
5. If unsure, choose UNCERTAIN. Be conservative.

Return exactly one JSON object:
{{
  "winner":"A|B|TIE|UNCERTAIN",
  "a_exact_intent":true/false,
  "b_exact_intent":true/false,
  "a_keyword_bait":true/false,
  "b_keyword_bait":true/false,
  "a_constraint_violation":true/false,
  "b_constraint_violation":true/false,
  "confidence":0.0-1.0,
  "reason":"short Turkish explanation"
}}
"""


def qwen_generate(tok, model, prompt):
    messages = [
        {"role": "system", "content": "You are a precise JSON-only Turkish e-commerce intent judge. Output only JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = "System: Output only JSON.\nUser:\n" + prompt + "\nAssistant:\n"

    device = next(model.parameters()).device
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=210,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def generic_prefilter_penalty(query, add_title, add_cat):
    q = norm(query)
    a = norm(add_title + " " + add_cat)
    penalty = 0.0
    reasons = []

    # Main-vs-accessory generic guard.
    q_accessory = any(norm(w) in q for w in QUERY_ACCESSORY_HINTS)
    a_accessory = any(norm(w) in a for w in MAIN_VS_ACCESSORY_WORDS)
    if a_accessory and not q_accessory:
        penalty += 0.50
        reasons.append("generic_main_vs_accessory")

    # Important numeric/spec preservation.
    q_nums = set(re.findall(r"\b\d+[a-z]*\b", q))
    a_nums = set(re.findall(r"\b\d+[a-z]*\b", a))
    if q_nums and any(ctx in q for ctx in IMPORTANT_SPEC_CONTEXT) and not (q_nums & a_nums):
        penalty += 0.35
        reasons.append("generic_important_spec_missing")

    # Content token coverage.
    stop = {"ve", "ile", "icin", "için", "bir", "adet", "set", "model", "uyumlu", "erkek", "kadin", "kadın", "cocuk", "çocuk", "bebek", "siyah", "beyaz"}
    q_tokens = [t for t in q.split() if len(t) >= 3 and t not in stop and not t.isdigit()]
    if q_tokens:
        hits = sum(1 for t in q_tokens if t in a)
        cov = hits / len(q_tokens)
        if cov < 0.34:
            penalty += 0.25
            reasons.append("generic_low_query_coverage")

    return penalty, "|".join(reasons)


def load_existing_jsonl(path):
    done = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done[str(obj["id_add"]) + "||" + str(obj["id_drop"])] = obj
            except Exception:
                pass
    return done


def choose_rows_for_qwen(pool):
    # Judge enough rows to be aggressive, but not all low-quality rows.
    cond = pd.Series(False, index=pool.index)
    for c in ["v65_mode_ultra", "v65_mode_safe", "v65_mode_aggressive", "v65_mode_model_boost"]:
        if c in pool.columns:
            cond |= pool[c].map(to_bool)

    if "v65_swap_utility" in pool.columns:
        util = safe_num(pool, "v65_swap_utility", 0.0)
        cond |= util >= util.quantile(0.45)

    if "qwen_accept_agree" in pool.columns:
        cond |= pool["qwen_accept_agree"].map(to_bool)

    # Do not send completely broken rows unless they were high utility.
    if "base_add_is0" in pool.columns:
        cond &= pool["base_add_is0"].map(to_bool)
    if "base_drop_is1" in pool.columns:
        cond &= pool["base_drop_is1"].map(to_bool)

    rows = pool[cond].copy()
    rows = rows.sort_values(["v65_swap_utility", "v65_order"], ascending=[False, True])
    rows = rows.drop_duplicates(["id_add_str", "id_drop_str"], keep="first")
    return rows


def reconstruct_base(sample, v60, pool):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample["id"])
    v60_pred = v60["prediction"].astype(np.int8).to_numpy()
    base = v60_pred.copy()

    # Undo detected V60 swaps.
    detected = 0
    for _, r in pool.iterrows():
        aid = str(r["id_add_str"])
        did = str(r["id_drop_str"])
        if aid not in id_to_idx.index or did not in id_to_idx.index:
            continue
        ai = int(id_to_idx.loc[aid])
        di = int(id_to_idx.loc[did])
        if v60_pred[ai] == 1 and v60_pred[di] == 0:
            base[ai] = 0
            base[di] = 1
            detected += 1

    print("detected V60 swaps", detected, "base-v60 diff", int((base != v60_pred).sum()))
    return base, v60_pred, id_to_idx


def unique_rows(df, id_col, score_col):
    if len(df) == 0:
        return df
    return df.sort_values([score_col, "v65_order"], ascending=[False, True]).drop_duplicates(id_col, keep="first").reset_index(drop=True)


def apply_variant(sample, base, id_to_idx, swaps=None, plus=None, minus=None):
    pred = base.copy()
    used_add = set()
    used_drop = set()

    if swaps is not None and len(swaps):
        for _, r in swaps.iterrows():
            aid = str(r["id_add_str"])
            did = str(r["id_drop_str"])
            if aid in used_add or did in used_drop:
                continue
            pred[int(id_to_idx.loc[aid])] = 1
            pred[int(id_to_idx.loc[did])] = 0
            used_add.add(aid)
            used_drop.add(did)

    if plus is not None and len(plus):
        for _, r in plus.iterrows():
            aid = str(r["id_add_str"])
            if aid in used_add:
                continue
            pred[int(id_to_idx.loc[aid])] = 1
            used_add.add(aid)

    if minus is not None and len(minus):
        for _, r in minus.iterrows():
            did = str(r["id_drop_str"])
            if did in used_drop:
                continue
            pred[int(id_to_idx.loc[did])] = 0
            used_drop.add(did)

    return pred


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    if not V65_POOL.exists():
        raise FileNotFoundError(V65_POOL)

    sample = pd.read_csv(SAMPLE, usecols=["id"])
    sample["id"] = sample["id"].astype(str)

    v60 = pd.read_csv(V60_CAP250)
    v60["id"] = v60["id"].astype(str)
    if not v60["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("V60 cap250 order mismatch")

    pool = pd.read_csv(V65_POOL)
    # Required columns from V65.
    for c in ["id_add_str", "id_drop_str"]:
        if c not in pool.columns:
            raise RuntimeError(f"{c} missing from V65 pool. Run script 168 first.")

    query_col = "query_x" if "query_x" in pool.columns else "query"
    title_add_col = "title_add_x" if "title_add_x" in pool.columns else "title_add"
    title_drop_col = "title_drop_x" if "title_drop_x" in pool.columns else "title_drop"
    cat_add_col = "category_add" if "category_add" in pool.columns else None
    cat_drop_col = "category_drop" if "category_drop" in pool.columns else None

    base, v60_pred, id_to_idx = reconstruct_base(sample, v60, pool)
    judge_rows = choose_rows_for_qwen(pool)
    print("judge rows", len(judge_rows))

    done = load_existing_jsonl(OUT_JSONL)
    print("already judged", len(done))

    print("loading", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("device", next(model.parameters()).device)

    start = time.time()
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        for j, (_, row) in enumerate(judge_rows.iterrows(), start=1):
            aid = str(row["id_add_str"])
            did = str(row["id_drop_str"])
            key = aid + "||" + did
            if key in done:
                continue

            query = get_text(row, query_col)
            add_title = get_text(row, title_add_col)
            add_cat = get_text(row, cat_add_col)
            drop_title = get_text(row, title_drop_col)
            drop_cat = get_text(row, cat_drop_col)

            raw = qwen_generate(tok, model, build_prompt(query, add_title, add_cat, drop_title, drop_cat))
            parsed = extract_json(raw)
            gp, gr = generic_prefilter_penalty(query, add_title, add_cat)

            obj = {
                "id_add": aid,
                "id_drop": did,
                "query": query,
                "title_add": add_title,
                "title_drop": drop_title,
                "qwen_intent_raw": raw[:1500],
                "generic_prefilter_penalty": gp,
                "generic_prefilter_reasons": gr,
                "elapsed_min": (time.time() - start) / 60.0,
                **parsed,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()

            print(
                j, "/", len(judge_rows),
                query[:36],
                obj["intent_winner"],
                "Aexact", obj["a_exact_intent"],
                "Bexact", obj["b_exact_intent"],
                "bait", obj["a_keyword_bait"],
                "viol", obj["a_constraint_violation"],
                "conf", round(obj["confidence"], 2),
                "gp", gp,
            )

    # Load all judgements and merge.
    jud = []
    with OUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                jud.append(json.loads(line))
    jud = pd.DataFrame(jud).drop_duplicates(["id_add", "id_drop"], keep="last")
    jud["id_add_str"] = jud["id_add"].astype(str)
    jud["id_drop_str"] = jud["id_drop"].astype(str)

    # Avoid merge suffix collisions with old V64/V65 judgement columns.
    # The pool already has query_x/query_y plus old confidence/winner columns.
    old_judge_cols = [
        "query", "title_add", "title_drop",
        "row_index", "qwen_raw", "winner", "a_relevant", "b_relevant",
        "confidence", "reason", "hard_guard_reasons", "elapsed_min",
        "qwen_accept_ultra", "qwen_accept_safe", "qwen_accept_agree",
        "v64_rank_score",
    ]
    pool_merge = pool.drop(columns=[c for c in old_judge_cols if c in pool.columns], errors="ignore")

    jud_merge = jud.drop(columns=["id_add", "id_drop"], errors="ignore")

    merged = pool_merge.merge(
        jud_merge,
        on=["id_add_str", "id_drop_str"],
        how="left",
        suffixes=("", "_intent"),
    )

    # Intent utility = second-pass strict Qwen + V65 model utility.
    conf = safe_num(merged, "confidence", 0.0).clip(0, 1)
    winner = merged["intent_winner"].fillna("UNCERTAIN").astype(str).str.upper()
    a_exact = merged["a_exact_intent"].map(to_bool)
    b_exact = merged["b_exact_intent"].map(to_bool)
    a_bait = merged["a_keyword_bait"].map(to_bool)
    a_viol = merged["a_constraint_violation"].map(to_bool)
    gp = safe_num(merged, "generic_prefilter_penalty", 0.0)

    merged["v66_intent_clean"] = (
        (winner == "A")
        & a_exact
        & (~a_bait)
        & (~a_viol)
        & (conf >= 0.60)
        & (gp <= 0.30)
    )
    merged["v66_intent_ultra"] = (
        merged["v66_intent_clean"]
        & (~b_exact)
        & (conf >= 0.70)
        & (gp <= 0.10)
    )
    merged["v66_intent_safe"] = (
        merged["v66_intent_clean"]
        & (conf >= 0.65)
        & (gp <= 0.18)
    )
    merged["v66_intent_aggressive"] = (
        merged["v66_intent_clean"]
        & (safe_num(merged, "v65_model_gain_score", 0.5) >= 0.50)
    )

    merged["v66_score"] = (
        0.46 * (winner == "A").astype(float)
        + 0.28 * a_exact.astype(float)
        - 0.35 * a_bait.astype(float)
        - 0.40 * a_viol.astype(float)
        - 0.18 * b_exact.astype(float)
        + 0.20 * conf
        + 0.20 * safe_num(merged, "v65_model_gain_score", 0.5)
        + 0.15 * safe_num(merged, "v65_swap_utility", 0.0).rank(pct=True)
        - 0.38 * gp
    )

    # Eligibility from base.
    merged["base_add_is0"] = merged["id_add_str"].map(lambda x: int(base[int(id_to_idx.loc[x])]) == 0 if x in id_to_idx.index else False)
    merged["base_drop_is1"] = merged["id_drop_str"].map(lambda x: int(base[int(id_to_idx.loc[x])]) == 1 if x in id_to_idx.index else False)
    merged.to_csv(OUT_REVIEW, index=False)

    saved = []
    picked_frames = []

    def save(name, pred, meta, frames):
        file = SUB_DIR / f"FINAL_CANDIDATE_{name}_{short_hash(name)}.csv"
        pd.DataFrame({"id": sample["id"], "prediction": pred.astype(np.int8)}).to_csv(file, index=False)
        row = {
            "variant": name,
            "file": str(file),
            "ones": int(pred.sum()),
            "pos_ratio": float(pred.mean()),
            "diff_vs_reconstructed_base": int((pred != base).sum()),
            "diff_vs_v60_cap250": int((pred != v60_pred).sum()),
            **meta,
        }
        saved.append(row)
        for action, df in frames:
            if df is not None and len(df):
                x = df.copy()
                x["variant"] = name
                x["action"] = action
                picked_frames.append(x)
        print("saved", file, row)

    modes = [
        ("ultra", "v66_intent_ultra"),
        ("safe", "v66_intent_safe"),
        ("aggressive", "v66_intent_aggressive"),
    ]
    caps = [10, 25, 40, 50, 64, 75, 100, 150]

    for mode, flag in modes:
        sp = merged[merged[flag] & merged["base_add_is0"] & merged["base_drop_is1"]].copy()
        sp = unique_rows(sp, "id_add_str", "v66_score")
        sp = sp.drop_duplicates("id_drop_str", keep="first")
        print(mode, "swap pool", len(sp))

        for cap in caps:
            if len(sp) == 0:
                continue
            take = sp.head(min(cap, len(sp)))
            pred = apply_variant(sample, base, id_to_idx, swaps=take)
            save(
                f"v66_intent_{mode}_swap_cap{cap}",
                pred,
                {
                    "family": "swap",
                    "mode": mode,
                    "used_swaps": int(len(take)),
                    "used_plus": 0,
                    "used_minus": 0,
                    "pool_size": int(len(sp)),
                    "mean_v66_score": float(take["v66_score"].mean()),
                    "mean_confidence": float(safe_num(take, "confidence", 0).mean()),
                    "keyword_bait_rate": float(take["a_keyword_bait"].map(to_bool).mean()),
                    "constraint_violation_rate": float(take["a_constraint_violation"].map(to_bool).mean()),
                    "b_exact_rate": float(take["b_exact_intent"].map(to_bool).mean()),
                    "generic_penalty_mean": float(safe_num(take, "generic_prefilter_penalty", 0).mean()),
                },
                [("swap", take)],
            )

    # Quota shift, but only from safe/ultra exact intent.
    plus_pool = merged[
        merged["v66_intent_safe"]
        & merged["base_add_is0"]
    ].copy()
    plus_pool = unique_rows(plus_pool, "id_add_str", "v66_score")

    minus_pool = merged[
        (merged["base_drop_is1"])
        & (merged["b_exact_intent"].map(to_bool) == False)
        & (safe_num(merged, "confidence", 0) >= 0.65)
        & (merged["intent_winner"].fillna("").astype(str).str.upper().isin(["A", "UNCERTAIN"]))
    ].copy()
    minus_pool["v66_minus_score"] = (
        0.45 * (safe_num(minus_pool, "confidence", 0))
        + 0.35 * (1.0 - safe_num(minus_pool, "v65_drop_quality", 0.5))
        + 0.20 * safe_num(minus_pool, "v65_model_gain_score", 0.5)
    )
    minus_pool = unique_rows(minus_pool, "id_drop_str", "v66_minus_score")

    for cap in [10, 25, 40, 50]:
        tp = plus_pool.head(min(cap, len(plus_pool)))
        if len(tp):
            pred = apply_variant(sample, base, id_to_idx, plus=tp)
            save(
                f"v66_intent_plus_cap{cap}",
                pred,
                {
                    "family": "plus",
                    "mode": "plus",
                    "used_swaps": 0,
                    "used_plus": int(len(tp)),
                    "used_minus": 0,
                    "pool_size": int(len(plus_pool)),
                    "mean_v66_score": float(tp["v66_score"].mean()),
                    "mean_confidence": float(safe_num(tp, "confidence", 0).mean()),
                    "keyword_bait_rate": float(tp["a_keyword_bait"].map(to_bool).mean()),
                    "constraint_violation_rate": float(tp["a_constraint_violation"].map(to_bool).mean()),
                    "b_exact_rate": float(tp["b_exact_intent"].map(to_bool).mean()),
                    "generic_penalty_mean": float(safe_num(tp, "generic_prefilter_penalty", 0).mean()),
                },
                [("plus", tp)],
            )

        tm = minus_pool.head(min(cap, len(minus_pool)))
        if len(tm):
            pred = apply_variant(sample, base, id_to_idx, minus=tm)
            save(
                f"v66_intent_minus_cap{cap}",
                pred,
                {
                    "family": "minus",
                    "mode": "minus",
                    "used_swaps": 0,
                    "used_plus": 0,
                    "used_minus": int(len(tm)),
                    "pool_size": int(len(minus_pool)),
                    "mean_v66_score": float(tm["v66_score"].mean()),
                    "mean_confidence": float(safe_num(tm, "confidence", 0).mean()),
                    "keyword_bait_rate": float(tm["a_keyword_bait"].map(to_bool).mean()),
                    "constraint_violation_rate": float(tm["a_constraint_violation"].map(to_bool).mean()),
                    "b_exact_rate": float(tm["b_exact_intent"].map(to_bool).mean()),
                    "generic_penalty_mean": float(safe_num(tm, "generic_prefilter_penalty", 0).mean()),
                },
                [("minus", tm)],
            )

    # Hybrids: strong swap plus tiny quota shift.
    safe_sp = merged[merged["v66_intent_safe"] & merged["base_add_is0"] & merged["base_drop_is1"]].copy()
    safe_sp = unique_rows(safe_sp, "id_add_str", "v66_score").drop_duplicates("id_drop_str", keep="first")

    for swap_cap, plus_cap, minus_cap in [
        (25, 10, 0),
        (40, 10, 0),
        (50, 10, 0),
        (25, 0, 10),
        (40, 0, 10),
        (50, 0, 10),
        (40, 10, 10),
        (50, 10, 10),
        (50, 25, 0),
        (50, 0, 25),
    ]:
        sw = safe_sp.head(min(swap_cap, len(safe_sp)))
        used_add = set(sw["id_add_str"])
        used_drop = set(sw["id_drop_str"])
        pp = plus_pool[~plus_pool["id_add_str"].isin(used_add)].head(min(plus_cap, len(plus_pool)))
        mm = minus_pool[~minus_pool["id_drop_str"].isin(used_drop)].head(min(minus_cap, len(minus_pool)))
        if len(sw) + len(pp) + len(mm) == 0:
            continue
        pred = apply_variant(sample, base, id_to_idx, swaps=sw, plus=pp, minus=mm)
        cat = pd.concat([x for x in [sw, pp, mm] if len(x)], ignore_index=True)
        save(
            f"v66_intent_hybrid_swap{swap_cap}_plus{plus_cap}_minus{minus_cap}",
            pred,
            {
                "family": "hybrid",
                "mode": "safe",
                "used_swaps": int(len(sw)),
                "used_plus": int(len(pp)),
                "used_minus": int(len(mm)),
                "pool_size": int(len(safe_sp)),
                "mean_v66_score": float(cat["v66_score"].mean()),
                "mean_confidence": float(safe_num(cat, "confidence", 0).mean()),
                "keyword_bait_rate": float(cat["a_keyword_bait"].map(to_bool).mean()),
                "constraint_violation_rate": float(cat["a_constraint_violation"].map(to_bool).mean()),
                "b_exact_rate": float(cat["b_exact_intent"].map(to_bool).mean()),
                "generic_penalty_mean": float(safe_num(cat, "generic_prefilter_penalty", 0).mean()),
            },
            [("swap", sw), ("plus", pp), ("minus", mm)],
        )

    summary = pd.DataFrame(saved)
    if len(summary):
        summary["v66_diagnostic_score"] = (
            0.028 * np.minimum(1, np.log1p(summary["diff_vs_reconstructed_base"]) / np.log1p(300))
            + 0.030 * summary["mean_v66_score"].fillna(0)
            + 0.020 * summary["mean_confidence"].fillna(0)
            - 0.030 * summary["generic_penalty_mean"].fillna(0)
            - 0.045 * summary["keyword_bait_rate"].fillna(0)
            - 0.045 * summary["constraint_violation_rate"].fillna(0)
            - np.maximum(0, summary["diff_vs_reconstructed_base"] - 350) / 12000
        )
        summary = summary.sort_values(["v66_diagnostic_score", "generic_penalty_mean", "diff_vs_reconstructed_base"], ascending=[False, True, True])
    summary.to_csv(OUT_SUMMARY, index=False)
    summary.to_csv(OUT_SAVED, index=False)

    if picked_frames:
        picked = pd.concat(picked_frames, ignore_index=True)
        picked.to_csv(OUT_DIR / "review_v66_qwen25_intent_guarded_picked.csv", index=False)

    print("\nTOP SUMMARY")
    cols = [
        "variant", "file", "family", "mode", "used_swaps", "used_plus", "used_minus",
        "ones", "pos_ratio", "diff_vs_reconstructed_base", "diff_vs_v60_cap250",
        "mean_v66_score", "mean_confidence", "keyword_bait_rate", "constraint_violation_rate",
        "b_exact_rate", "generic_penalty_mean", "v66_diagnostic_score",
    ]
    if len(summary):
        print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))

    print("\noutputs:")
    print(OUT_JSONL)
    print(OUT_REVIEW)
    print(OUT_SUMMARY)
    print(OUT_SAVED)
    print(SUB_DIR)


if __name__ == "__main__":
    main()
