"""V233 — Doğrulanmış V221'i hard-conflict korumalı olarak genişlet.

V230'un global Platt/cardinality yaklaşımı leaderboard'da 0.809 aldığı için bu
script Platt skorlarını kullanmaz. Başlangıç noktası leaderboard'da 0.832 alan
V221'dir. V221'in özgün Trendyol embedding sıralamasında 1001-3000 arasındaki
add adayları tekrar alınır; kesin teknik uyuşmazlıklar çıkarılır ve yeni removal
yapılmaz.

Kullanım:
    python scripts/233_build_guarded_asymmetric_v221_expansion.py build
"""

# %% [markdown]
# 0. Tasarım
# ----------
# - Anchor: FINAL_BEST_VERIFIED_0p832.csv
# - Yeni remove: yok
# - Kaynak: V221 V104/V219/V220 disagreement + Trendyol embedding
# - Genişleme: özgün add rank 1001..3000
# - Veto: model/grade/subject/bez no/formula stage/ölçü/cinsiyet/tek-çift
# - Marka veya genel alternatif tek başına veto değildir; yarı alakalı olabilir.

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# %% [markdown]
# 1. Ayarlar


@dataclass(frozen=True)
class Config:
    anchor: Path = Path("submissions/FINAL_BEST_VERIFIED_0p832.csv")
    v104: Path = Path("submissions/reference/FINAL_CANDIDATE_v104_full_balanced.csv")
    scored_disagreement: Path = Path("data/processed/v221_embedding_guarded_disagreement.parquet")
    terms: Path = Path("data/raw/terms.csv")
    items: Path = Path("data/raw/items.csv")
    output: Path = Path(
        "submissions/final_candidates_v233/FINAL_CANDIDATE_v233_guarded_asymmetric_add_3000.csv"
    )
    audit: Path = Path("reports/experiments/v233_guarded_asymmetric_add_audit.csv")
    report: Path = Path("reports/experiments/v233_guarded_asymmetric_add.json")
    verified_add_rank: int = int(os.environ.get("V233_VERIFIED_ADD_RANK", "1000"))
    expansion_add_rank: int = int(os.environ.get("V233_EXPANSION_ADD_RANK", "3000"))
    item_chunk: int = int(os.environ.get("V233_ITEM_CHUNK", "100000"))


CFG = Config()


# %% [markdown]
# 2. Metin normalizasyonu ve V221 önceliği


def normalize(value: object) -> str:
    text = str(value or "").lower().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def percentile(values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.rank(method="average", pct=True, ascending=ascending).astype(np.float32)


def rebuild_v221_add_ranking(rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    add = rows[rows["direction"].eq("add")].copy()
    add["priority"] = (
        0.55 * percentile(add["trendyol_embedding"])
        + 0.25 * percentile(add["v220_score"])
        + 0.20 * percentile(add["v219_score"])
    )
    cutoff = float(add["trendyol_embedding"].quantile(0.75))
    add = add[add["trendyol_embedding"].ge(cutoff)].copy()
    add = add.sort_values(["priority", "id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    add["v221_add_rank"] = np.arange(1, len(add) + 1, dtype=np.int32)
    return add, cutoff


# %% [markdown]
# 3. Kesin uyuşmazlık çıkarıcıları
# ---------------------------------
# Kurallar yalnızca iki tarafta da açık bilgi bulunduğunda veto verir. Bilgi
# eksikliği veto değildir. Böylece marka alternatifi ve genel yarı-alaka korunur.


GRADE_RE = re.compile(r"(?<!\d)(1[0-2]|[1-9])\s*\.?\s*(?:sinif|sınıf)")
# ``normalize`` punctuationı boşluğa çevirdiği için 215/60 R17 aynı zamanda
# ``215 60 r17`` biçiminde yakalanır.
TIRE_RE = re.compile(r"(?<!\d)(\d{3})\s+(\d{2})\s*(?:r\s*)?(\d{2})(?!\d)")
JANT_RE = re.compile(r"(?<!\d)(1[0-9]|2[0-9])\s*(?:inc|inch|jant)(?!\w)")
PERSON_RE = re.compile(r"\b(tek|cift)\s+kisilik\b")
SUBJECTS = {
    "matematik", "turkce", "ingilizce", "fen", "fizik", "kimya", "biyoloji",
    "sosyal", "cografya", "tarih", "paragraf", "din", "inkilap",
}

DEVICE_PATTERNS = {
    "iphone": re.compile(r"\biphone\s*(\d{1,2})\s*(pro\s*max|promax|pro|plus|mini|e)?\b"),
    "ipad": re.compile(r"\bipad\s*(pro|air|mini)?\s*(\d{1,2}(?:\s*\.\s*\d)?)?\s*(?:nesil)?\b"),
    "samsung_tab": re.compile(r"\b(?:samsung\s+)?(?:galaxy\s+)?tab\s*([as]\s*\d{1,2})\s*(fe|plus|lite)?\b"),
    "samsung_phone": re.compile(r"\b(?:samsung\s+)?(?:galaxy\s+)?([asmz]\s*\d{1,3})\s*(fe|plus|ultra)?\b"),
    "redmi": re.compile(r"\b(?:xiaomi\s+)?redmi\s+(note\s+)?(\d{1,2})\s*(pro\s*plus|pro|plus|ultra|t)?\b"),
    "poco": re.compile(r"\bpoco\s*([xfc]\s*\d{1,2})\s*(pro|plus|ultra)?\b"),
    "oppo": re.compile(r"\boppo\s*([ar]\s*\d{1,3})\s*(pro|plus|5g|4g)?\b"),
    "vivo": re.compile(r"\bvivo\s*([yv]\s*\d{1,3})\s*(pro|plus|lite|5g|4g)?\b"),
    "honor_pad": re.compile(r"\bhonor\s+pad\s*([a-z]?\s*\d{1,2})\s*(pro|plus|lite)?\b"),
}


def canonical_match(match: re.Match) -> str:
    return " ".join(part.replace(" ", "") for part in match.groups() if part).strip()


def device_models(text: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for family, pattern in DEVICE_PATTERNS.items():
        values = {canonical_match(match) for match in pattern.finditer(text)}
        if values:
            out[family] = values
    return out


def explicit_device_conflict(query: str, item: str) -> str | None:
    q_models = device_models(query)
    i_models = device_models(item)
    for family in sorted(set(q_models) & set(i_models)):
        if q_models[family].isdisjoint(i_models[family]):
            return f"device_model_mismatch:{family}"
    return None


def extract_grades(text: str) -> set[int]:
    return {int(value) for value in GRADE_RE.findall(text)}


def extract_subjects(text: str) -> set[str]:
    tokens = set(text.split())
    return tokens & SUBJECTS


def diaper_numbers(text: str) -> set[int]:
    if not re.search(r"\b(?:bebek\s+bezi|kulot\s+bez|bez)\b", text):
        return set()
    values = set()
    for pattern in [
        r"\b(?:bebek\s+bezi|kulot\s+bez)\s*(?:no|numara)?\s*([1-8])(?!\d|\s*['’]?\s*li\b)",
        r"\b([1-8])\s*(?:numara|no|beden)\b",
        r"\b(?:numara|no|beden)\s*([1-8])\b",
    ]:
        values.update(int(v) for v in re.findall(pattern, text))
    return values


def formula_stages(text: str) -> set[int]:
    if not re.search(r"\b(?:aptamil|bebelac|hipp|devam\s+sutu|bebek\s+sutu)\b", text):
        return set()
    values = set()
    for pattern in [
        r"\b(?:aptamil|bebelac|hipp)\D{0,12}([1-4])(?!\d)",
        r"\b(?:no|numara)\s*([1-4])\b",
        r"\b([1-4])\s*(?:no|numara)\b",
    ]:
        values.update(int(v) for v in re.findall(pattern, text))
    return values


def genders(text: str) -> set[str]:
    out = set()
    if re.search(r"\b(?:kadin|bayan|kiz)\b", text):
        out.add("female")
    if re.search(r"\b(?:erkek|bay)\b", text):
        out.add("male")
    if "unisex" in text:
        out.add("unisex")
    return out


def disjoint_nonempty(left: set, right: set) -> bool:
    return bool(left and right and left.isdisjoint(right))


def hard_conflict_reasons(query_raw: object, title_raw: object, category_raw: object, attributes_raw: object) -> list[str]:
    query = normalize(query_raw)
    item = normalize(f"{title_raw} {category_raw} {attributes_raw}")
    reasons: list[str] = []

    model_reason = explicit_device_conflict(query, item)
    if model_reason:
        reasons.append(model_reason)

    q_grade, i_grade = extract_grades(query), extract_grades(item)
    if disjoint_nonempty(q_grade, i_grade):
        reasons.append("class_grade_mismatch")
    q_subject, i_subject = extract_subjects(query), extract_subjects(item)
    if q_grade and i_grade and not q_grade.isdisjoint(i_grade) and disjoint_nonempty(q_subject, i_subject):
        reasons.append("education_subject_mismatch")

    if disjoint_nonempty(diaper_numbers(query), diaper_numbers(item)):
        reasons.append("diaper_number_mismatch")
    if disjoint_nonempty(formula_stages(query), formula_stages(item)):
        reasons.append("formula_stage_mismatch")

    q_tire, i_tire = set(TIRE_RE.findall(query)), set(TIRE_RE.findall(item))
    if disjoint_nonempty(q_tire, i_tire):
        reasons.append("tire_size_mismatch")
    if ("jant" in query or "bisiklet" in query) and disjoint_nonempty(
        set(JANT_RE.findall(query)), set(JANT_RE.findall(item))
    ):
        reasons.append("wheel_size_mismatch")

    q_person, i_person = set(PERSON_RE.findall(query)), set(PERSON_RE.findall(item))
    if disjoint_nonempty(q_person, i_person):
        reasons.append("person_count_mismatch")

    q_gender, i_gender = genders(query), genders(item)
    q_explicit = q_gender - {"unisex"}
    i_explicit = i_gender - {"unisex"}
    if "unisex" not in i_gender and disjoint_nonempty(q_explicit, i_explicit):
        reasons.append("opposite_gender_mismatch")
    return sorted(set(reasons))


# %% [markdown]
# 4. Aday metadatası


def load_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    terms = pd.read_csv(CFG.terms, usecols=["term_id", "query"], dtype=str).drop_duplicates("term_id")
    needed_items = set(candidates["item_id"].astype(str))
    item_frames = []
    columns = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    for chunk in pd.read_csv(CFG.items, usecols=columns, dtype=str, chunksize=CFG.item_chunk, low_memory=False):
        hit = chunk[chunk["item_id"].isin(needed_items)]
        if not hit.empty:
            item_frames.append(hit)
    if not item_frames:
        raise ValueError("No V233 item metadata found")
    items = pd.concat(item_frames, ignore_index=True).drop_duplicates("item_id")
    out = candidates.merge(terms, on="term_id", how="left", validate="many_to_one")
    out = out.merge(items, on="item_id", how="left", validate="many_to_one")
    if out["query"].isna().any() or out["title"].isna().any():
        raise ValueError("Missing query or item metadata in V233 pool")
    return out


# %% [markdown]
# 5. Guard uygulanması ve submission


def build() -> dict:
    required = [CFG.anchor, CFG.v104, CFG.scored_disagreement, CFG.terms, CFG.items]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V233 inputs:\n- " + "\n- ".join(missing))
    if CFG.expansion_add_rank <= CFG.verified_add_rank:
        raise ValueError("V233_EXPANSION_ADD_RANK must exceed V233_VERIFIED_ADD_RANK")

    rows = pd.read_parquet(CFG.scored_disagreement)
    needed = {"id", "term_id", "item_id", "direction", "v219_score", "v220_score", "trendyol_embedding"}
    if not needed <= set(rows):
        raise ValueError(f"Scored disagreement missing {sorted(needed - set(rows))}")
    for column in ["id", "term_id", "item_id"]:
        rows[column] = rows[column].astype(str)
    ranked_add, semantic_cutoff = rebuild_v221_add_ranking(rows)
    tail = ranked_add[
        ranked_add["v221_add_rank"].gt(CFG.verified_add_rank)
        & ranked_add["v221_add_rank"].le(CFG.expansion_add_rank)
    ].copy()
    if tail.empty:
        raise ValueError("V233 expansion tail is empty")
    enriched = load_metadata(tail)
    enriched["guard_reasons"] = enriched.apply(
        lambda row: "|".join(hard_conflict_reasons(row["query"], row["title"], row["category"], row["attributes"])),
        axis=1,
    )
    enriched["guarded"] = enriched["guard_reasons"].ne("")
    eligible = enriched[~enriched["guarded"]].copy()

    anchor = pd.read_csv(CFG.anchor, dtype={"id": str})
    v104 = pd.read_csv(CFG.v104, dtype={"id": str})
    if not anchor["id"].equals(v104["id"]):
        raise ValueError("V221 anchor and V104 ID order mismatch")
    old = anchor["prediction"].to_numpy(np.int8)
    teacher = v104["prediction"].to_numpy(np.int8)
    add_ids = set(eligible["id"])
    add_mask = anchor["id"].isin(add_ids).to_numpy() & (old == 0)
    prediction = old.copy()
    prediction[add_mask] = 1
    if ((old == 1) & (prediction == 0)).any():
        raise RuntimeError("V233 invariant violated: a positive was removed")

    CFG.output.parent.mkdir(parents=True, exist_ok=True)
    CFG.audit.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": anchor["id"], "prediction": prediction}).to_csv(CFG.output, index=False)
    audit_columns = [
        "id", "term_id", "item_id", "v221_add_rank", "priority", "trendyol_embedding",
        "v219_score", "v220_score", "guarded", "guard_reasons", "query", "title",
        "category", "brand", "gender", "age_group", "attributes",
    ]
    enriched[audit_columns].to_csv(CFG.audit, index=False, encoding="utf-8-sig")

    guard_counts = (
        enriched.loc[enriched["guarded"], "guard_reasons"]
        .str.split("|").explode().value_counts().to_dict()
    )
    report = {
        "run": "v233_guarded_asymmetric_v221_expansion",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "anchor": str(CFG.anchor),
        "anchor_changes_vs_v104": int((old != teacher).sum()),
        "v221_semantic_cutoff_q75": semantic_cutoff,
        "ranked_add_rows": len(ranked_add),
        "expansion_tail_rows": len(tail),
        "guarded_rows": int(enriched["guarded"].sum()),
        "eligible_new_add_rows": int(add_mask.sum()),
        "guard_reason_counts": {str(k): int(v) for k, v in guard_counts.items()},
        "new_remove_rows": int(((old == 1) & (prediction == 0)).sum()),
        "changes_vs_v221": int((old != prediction).sum()),
        "changes_vs_v104": int((teacher != prediction).sum()),
        "anchor_positive_ratio": float(old.mean()),
        "output_positive_ratio": float(prediction.mean()),
        "positive_ratio_delta": float(prediction.mean() - old.mean()),
        "submission": str(CFG.output),
        "audit": str(CFG.audit),
        "policy": "V221 anchor; add ranks 1001..3000; hard conflicts vetoed; no new removals",
        "warning": "V221's +0.001 public improvement validates only its first 1000 equal-direction swaps; tail expansion remains a challenger.",
    }
    CFG.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# %% [markdown]
# 6. Komut satırı


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    args = parser.parse_args()
    if args.command == "build":
        build()


if __name__ == "__main__":
    main()
