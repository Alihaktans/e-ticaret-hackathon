"""Build V113 by turning the audited shortlist into stable expansion rules."""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd


ROOT = Path(".")

BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
CURRENT = ROOT / "submissions/final_candidates_v108/FINAL_CANDIDATE_v108_bridge_top.csv"
AUDIT = ROOT / "reports/manual_review/v112_human_audit_shortlist_assistant_filled.csv"

SCRIPT_210 = ROOT / "scripts/210_make_v106_generalized_filterpack.py"
SCRIPT_212 = ROOT / "scripts/212_make_v108_stable_route_filterpack.py"

OOF_CACHE = ROOT / "data/processed/v113_audit_rulepack_oof_eval.parquet"
TEST_CACHE = ROOT / "data/processed/v113_audit_rulepack_test_eval.parquet"

OUT_DIR = ROOT / "submissions/final_candidates_v113"
OUT_FILE = OUT_DIR / "FINAL_CANDIDATE_v113_audit_rulepack.csv"
REPORT = ROOT / "reports/experiments/v113_audit_rulepack.json"

LOCKED_FAMILIES = [
    {
        "family": "repair_general_nohead",
        "decision": "lock",
        "reason": "No head anchor means repair candidates drift semantically and should stay closed.",
    },
    {
        "family": "repair_same_head_swap",
        "decision": "lock",
        "reason": "Same head overlap alone is not enough to certify a repair.",
    },
    {
        "family": "repair_cross_root",
        "decision": "lock",
        "reason": "Cross-root repair remains unstable and should not be generalized.",
    },
    {
        "family": "repair_device_domain",
        "decision": "lock",
        "reason": "Device-domain repair crosses main/accessory families too easily.",
    },
]

RULE_SPECS = [
    {
        "name": "audit_phone_main_accessory",
        "description": "Main phone query matched to an explicit accessory item.",
        "audit_selector": lambda df: (
            df["human_label"].eq("expand_rule_family")
            & df["reason_text"].fillna("").str.contains("query_main_item_accessory")
            & df["query_device_family"].eq("phone")
        ),
        "guardrails": [
            "Keep the rule phone-only; do not reopen console/controller cases such as PS5 accessory intents.",
            "Keep a score cap at 0.40 so ambiguous high-score rows like 'iphone power' stay outside this expansion.",
        ],
        "thresholds": {"score_max": 0.40},
    },
    {
        "name": "audit_tablet_compatibility",
        "description": "Tablet accessory query matched to a different model family.",
        "audit_selector": lambda df: (
            df["human_label"].eq("expand_rule_family")
            & df["reason_text"].fillna("").str.contains("compatibility_model_mismatch")
            & df["query_device_family"].eq("tablet")
        ),
        "guardrails": [
            "Keep the rule tablet-only; phone compatibility already has its own family.",
            "Require accessory intent and a low score cap to avoid broad taxonomy-only removals.",
        ],
        "thresholds": {"score_max": 0.18},
    },
    {
        "name": "audit_gender_opposite",
        "description": "Explicit male/female mismatch with a safe alternative candidate available.",
        "audit_selector": lambda df: (
            df["human_label"].eq("expand_rule_family")
            & df["reason_text"].fillna("").str.contains("gender_female_vs_male|gender_male_vs_female")
        ),
        "guardrails": [
            "Male/female -> unisex stays allowed.",
            "This expansion only certifies explicit opposite-gender cases from the audit.",
        ],
        "thresholds": {"alt_gap_gender_veto_min": 0.02},
    },
    {
        "name": "audit_bedding_person_count",
        "description": "Double/single mismatch on bedding heads where package count is decisive.",
        "audit_selector": lambda df: (
            df["human_label"].eq("expand_rule_family")
            & df["reason_text"].fillna("").str.contains("person_count_mismatch")
            & df["query_heads"].fillna("").str.contains("yorgan|pike|carsaf")
        ),
        "guardrails": [
            "Keep the rule limited to bedding heads yorgan/pike/carsaf.",
            "Messy sheet-size rows stay outside unless they also satisfy the low score cap.",
        ],
        "thresholds": {"score_max": 0.40},
    },
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attach_query_heads(df: pd.DataFrame, mod209) -> pd.DataFrame:
    out = df.copy()
    brand_tokens = mod209.build_brand_token_set()
    query_info = mod209.build_query_info(set(out["term_id"].astype(str)), brand_tokens)
    out["query_heads"] = out["term_id"].map(
        lambda term_id: "|".join(sorted(query_info.get(str(term_id), {}).get("head_labels", [])))
    )
    return out


def build_or_load_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if OOF_CACHE.exists() and TEST_CACHE.exists():
        base_eval = pd.read_parquet(OOF_CACHE)
        test_eval = pd.read_parquet(TEST_CACHE)
        for df in (base_eval, test_eval):
            for col in ("id", "term_id", "item_id"):
                df[col] = df[col].astype(str)
            df["reason_text"] = df["reason_text"].fillna("")
            df["query_heads"] = df["query_heads"].fillna("")
        return base_eval, test_eval

    mod210 = load_module(SCRIPT_210, "mod210")
    mod212 = load_module(SCRIPT_212, "mod212")
    mod209 = mod210.load_filterpack_module()

    oof_new, test_new, _ = mod212.build_or_load_source_matched_scores(mod210, mod209)
    base_eval = attach_query_heads(mod212.add_alt_features(mod212.build_base_eval(mod209, mod210, oof_new)), mod209)
    test_eval = attach_query_heads(mod212.add_alt_features(mod212.build_test_eval(mod209, test_new)), mod209)

    OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    base_eval.to_parquet(OOF_CACHE, index=False)
    test_eval.to_parquet(TEST_CACHE, index=False)
    return base_eval, test_eval


def reason_contains(df: pd.DataFrame, pattern: str) -> np.ndarray:
    return df["reason_text"].fillna("").str.contains(pattern, regex=True).to_numpy()


def head_contains(df: pd.DataFrame, pattern: str) -> np.ndarray:
    return df["query_heads"].fillna("").str.contains(pattern, regex=True).to_numpy()


def build_rule_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "audit_phone_main_accessory":
        return (
            reason_contains(df, "query_main_item_accessory")
            & df["query_intent"].eq("device_main").to_numpy()
            & df["query_device_family"].eq("phone").to_numpy()
            & df["score"].le(0.40).to_numpy()
        )
    if name == "audit_tablet_compatibility":
        return (
            reason_contains(df, "compatibility_model_mismatch")
            & df["query_intent"].eq("device_accessory").to_numpy()
            & df["query_device_family"].eq("tablet").to_numpy()
            & df["score"].le(0.18).to_numpy()
        )
    if name == "audit_gender_opposite":
        return (
            df["gender_veto"].eq(1).to_numpy()
            & reason_contains(df, "gender_female_vs_male|gender_male_vs_female")
            & df["alt_gap_gender_veto"].ge(0.02).to_numpy()
        )
    if name == "audit_bedding_person_count":
        return (
            reason_contains(df, "person_count_mismatch")
            & head_contains(df, "yorgan|pike|carsaf")
            & df["score"].le(0.40).to_numpy()
        )
    raise KeyError(name)


def fold_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    rows = int(mask.sum())
    errs = int(df.loc[mask, "label"].sum())
    by_fold = []
    if rows:
        for fold, grp in df.loc[mask].groupby("fold5", sort=True):
            n = int(len(grp))
            e = int(grp["label"].sum())
            by_fold.append(
                {
                    "fold": int(fold),
                    "rows": n,
                    "positive_errors": e,
                    "negative_precision": float((n - e) / n),
                }
            )
    return {
        "rows": rows,
        "positive_errors": errs,
        "negative_precision": float((rows - errs) / rows) if rows else float("nan"),
        "min_fold_precision": min((row["negative_precision"] for row in by_fold), default=float("nan")),
        "by_fold": by_fold,
    }


def load_candidate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    df["prediction"] = df["prediction"].astype(np.int8)
    return df


def audit_support_rows(audit_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    support = audit_df.loc[spec["audit_selector"](audit_df), ["term_id", "query", "id", "reason_text", "human_label"]].copy()
    support = support.drop_duplicates().reset_index(drop=True)
    return support


def save_candidate(current: pd.DataFrame, removed_ids: set[str]) -> dict:
    base = load_candidate(BASE)
    pred = current.copy()
    row_mask = pred["id"].isin(removed_ids) & pred["prediction"].eq(1)
    pred.loc[row_mask, "prediction"] = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred[["id", "prediction"]].to_csv(OUT_FILE, index=False)
    new_pred = pred["prediction"].to_numpy(np.int8)
    base_pred = base["prediction"].to_numpy(np.int8)
    return {
        "file": str(OUT_FILE),
        "sha256": hashlib.sha256(OUT_FILE.read_bytes()).hexdigest(),
        "positives": int(new_pred.sum()),
        "ratio": float(new_pred.mean()),
        "removed_vs_current": int(len(removed_ids)),
        "removed_vs_base": int((base_pred == 1).sum() - new_pred.sum()),
    }


def main():
    audit_df = pd.read_csv(AUDIT)
    for col in ("id", "term_id"):
        if col in audit_df.columns:
            audit_df[col] = audit_df[col].astype(str)
    audit_df["reason_text"] = audit_df["reason_text"].fillna("")
    audit_df["query_heads"] = audit_df["query_heads"].fillna("")

    base_eval, test_eval = build_or_load_eval_frames()
    current = load_candidate(CURRENT)
    current_keep_ids = set(current.loc[current["prediction"].eq(1), "id"])

    rule_reports = []
    union_oof = np.zeros(len(base_eval), dtype=bool)
    union_test = np.zeros(len(test_eval), dtype=bool)

    for spec in RULE_SPECS:
        mask_oof = build_rule_mask(base_eval, spec["name"])
        mask_test = build_rule_mask(test_eval, spec["name"])
        union_oof |= mask_oof
        union_test |= mask_test

        support = audit_support_rows(audit_df, spec)
        test_ids = set(test_eval.loc[mask_test, "id"].astype(str))
        incremental_ids = test_ids & current_keep_ids

        rule_reports.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "thresholds": spec["thresholds"],
                "guardrails": spec["guardrails"],
                "audit_support_rows": int(len(support)),
                "audit_support_examples": support.head(8).to_dict(orient="records"),
                "oof_metrics": fold_metrics(base_eval, mask_oof),
                "test_positive_hits": int(mask_test.sum()),
                "incremental_vs_current": int(len(incremental_ids)),
                "already_removed_by_current": int(mask_test.sum() - len(incremental_ids)),
            }
        )

    removed_ids = set(test_eval.loc[union_test, "id"].astype(str)) & current_keep_ids
    candidate_summary = save_candidate(current, removed_ids)

    report = {
        "base_file": str(BASE),
        "current_file": str(CURRENT),
        "audit_file": str(AUDIT),
        "locked_families": LOCKED_FAMILIES,
        "rule_reports": rule_reports,
        "union_oof_metrics": fold_metrics(base_eval, union_oof),
        "union_test_positive_hits": int(union_test.sum()),
        "union_incremental_vs_current": int(len(removed_ids)),
        "candidate": candidate_summary,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
