"""Build V107 from only objectively stable, OOF-certified subfilters."""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd


ROOT = Path(".")
BASE = ROOT / "submissions/final_candidates_v95/FINAL_CANDIDATE_v95_constraint_raw004115.csv"
OUT_DIR = ROOT / "submissions/final_candidates_v107"
REPORT = ROOT / "reports/experiments/v107_objective_stability_pack.json"
HANDOFF = ROOT / "reports/HANDOFF_V107_OBJECTIVE_STABILITY_2026-07-05.md"

SCRIPT_210 = ROOT / "scripts/210_make_v106_generalized_filterpack.py"
FEATURE_REPORT = ROOT / "reports/experiments/v83_independent_pairlocal_classifier.json"

CERTIFIED_CONFIGS = {
    "objective_certified_top": {
        "gender_fashion_max": 0.22,
        "gender_general_max": 0.10,
        "main_phone_max": 0.10,
        "main_vacuum_max": 0.15,
        "main_tablet_max": 0.30,
        "person_count_max": 0.18,
        "include_accessory_vacuum": True,
    },
    "objective_certified_precision": {
        "gender_fashion_max": 0.22,
        "gender_general_max": 0.10,
        "main_phone_max": 0.10,
        "main_vacuum_max": 0.12,
        "main_tablet_max": 0.30,
        "person_count_max": 0.18,
        "include_accessory_vacuum": True,
    },
}


def load_v106_module():
    spec = importlib.util.spec_from_file_location("v106mod", SCRIPT_210)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def certified_masks(df: pd.DataFrame, cfg: dict[str, float]) -> dict[str, pd.Series]:
    txt = df["reason_text"].fillna("")
    masks = {
        "family_strict_all": df["family_veto_strict"].eq(1),
        "family_balanced_all": df["family_veto_balanced"].eq(1),
        "class_grade_all": txt.str.contains("class_grade_mismatch"),
        "tire_size_all": txt.str.contains("tire_size_mismatch"),
        "diaper_no_all": txt.str.contains("diaper_no_mismatch"),
        "formula_stage_all": txt.str.contains("formula_stage_mismatch"),
        "jant_certified": txt.str.contains("jant_mismatch") & df["score"].le(0.18),
        "person_count_certified": txt.str.contains("person_count_mismatch") & df["score"].le(cfg["person_count_max"]),
        "gender_fashion_certified": (
            df["gender_veto"].eq(1)
            & df["query_intent"].eq("gendered_fashion")
            & df["score"].le(cfg["gender_fashion_max"])
        ),
        "gender_general_certified": (
            df["gender_veto"].eq(1)
            & df["query_intent"].eq("general")
            & df["score"].le(cfg["gender_general_max"])
        ),
        "main_phone_certified": (
            df["main_accessory_veto"].eq(1)
            & df["query_intent"].eq("device_main")
            & df["query_device_family"].eq("phone")
            & df["score"].le(cfg["main_phone_max"])
        ),
        "main_vacuum_certified": (
            df["main_accessory_veto"].eq(1)
            & df["query_intent"].eq("device_main")
            & df["query_device_family"].eq("vacuum")
            & df["score"].le(cfg["main_vacuum_max"])
        ),
        "main_tablet_certified": (
            df["main_accessory_veto"].eq(1)
            & df["query_intent"].eq("device_main")
            & df["query_device_family"].eq("tablet")
            & df["score"].le(cfg["main_tablet_max"])
        ),
    }
    if cfg["include_accessory_vacuum"]:
        masks["accessory_vacuum_certified"] = (
            df["main_accessory_veto"].eq(1)
            & df["query_intent"].eq("device_accessory")
            & df["query_device_family"].eq("vacuum")
            & df["score"].le(0.30)
        )
    else:
        masks["accessory_vacuum_certified"] = pd.Series(False, index=df.index)
    return masks


def union_mask(masks: dict[str, pd.Series]) -> np.ndarray:
    out = np.zeros(len(next(iter(masks.values()))), dtype=bool)
    for mask in masks.values():
        out |= mask.to_numpy()
    return out


def fold_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    rows = int(mask.sum())
    errs = int(df.loc[mask, "label"].sum())
    prec = float((rows - errs) / rows) if rows else float("nan")
    by_fold = []
    min_fold = float("nan")
    if rows:
        vals = []
        for fold, grp in df[mask].groupby("fold5", sort=True):
            n = int(len(grp))
            e = int(grp["label"].sum())
            p = float((n - e) / n)
            vals.append(p)
            by_fold.append({"fold": int(fold), "rows": n, "positive_errors": e, "negative_precision": p})
        min_fold = float(min(vals)) if vals else float("nan")
    return {
        "rows": rows,
        "positive_errors": errs,
        "negative_precision": prec,
        "min_fold_precision": min_fold,
        "by_fold": by_fold,
    }


def variant_components(df: pd.DataFrame, cfg: dict[str, float]) -> tuple[np.ndarray, dict[str, int]]:
    masks = certified_masks(df, cfg)
    union = union_mask(masks)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    return union, counts


def save_variant(base_df: pd.DataFrame, ids: pd.Series, mask_ids: set[str], name: str) -> dict:
    pred = base_df.copy()
    pred["prediction"] = pred["prediction"].astype(np.int8)
    row_mask = pred["id"].isin(mask_ids) & pred["prediction"].eq(1)
    pred.loc[row_mask, "prediction"] = 0
    path = OUT_DIR / f"FINAL_CANDIDATE_v107_{name}.csv"
    pred[["id", "prediction"]].to_csv(path, index=False)
    base_pred = base_df["prediction"].to_numpy(np.int8)
    new_pred = pred["prediction"].to_numpy(np.int8)
    return {
        "variant": name,
        "removed_vs_base": int((base_pred == 1).sum() - new_pred.sum()),
        "positives": int(new_pred.sum()),
        "ratio": float(new_pred.mean()),
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    mod210 = load_v106_module()
    mod209 = mod210.load_filterpack_module()
    feature_cols = json.loads(FEATURE_REPORT.read_text(encoding="utf8"))["features"]
    train_df, test_df, oof_df, test_score_df = mod210.build_or_load_scores(feature_cols)
    base_eval = mod210.build_base_eval(mod209, oof_df)
    test_eval = mod210.build_test_eval(mod209, test_score_df, {"gender_score_max": 0.15, "color_score_max": 0.08, "brand_score_max": 0.04, "brand_wov_pct_max": 0.4})

    base = pd.read_csv(BASE)
    base["id"] = base["id"].astype(str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    candidates = []
    for name, cfg in CERTIFIED_CONFIGS.items():
        oof_mask, oof_counts = variant_components(base_eval, cfg)
        test_mask, test_counts = variant_components(test_eval, cfg)
        metrics = fold_metrics(base_eval, oof_mask)
        reports.append(
            {
                "variant": name,
                "config": cfg,
                "oof_metrics": metrics,
                "oof_component_counts": oof_counts,
                "test_component_counts": test_counts,
                "test_positive_hits": int(test_mask.sum()),
            }
        )
        mask_ids = set(test_eval.loc[test_mask, "id"])
        candidates.append(save_variant(base, base["id"], mask_ids, name))

    family_variants = {
        "family_strict_only": test_eval["family_veto_strict"].eq(1).to_numpy(),
        "family_balanced_only": test_eval["family_veto_balanced"].eq(1).to_numpy(),
    }
    for name, mask in family_variants.items():
        source_mask = (
            base_eval["family_veto_strict"].eq(1).to_numpy()
            if name == "family_strict_only"
            else base_eval["family_veto_balanced"].eq(1).to_numpy()
        )
        metrics = fold_metrics(base_eval, source_mask)
        reports.append(
            {
                "variant": name,
                "config": None,
                "oof_metrics": metrics,
                "oof_component_counts": {name: int(metrics["rows"])},
                "test_component_counts": {name: int(mask.sum())},
                "test_positive_hits": int(mask.sum()),
            }
        )
        mask_ids = set(test_eval.loc[mask, "id"])
        candidates.append(save_variant(base, base["id"], mask_ids, name))

    report = {
        "base_file": str(BASE),
        "base_positives": int(base["prediction"].sum()),
        "source_generalization_audit": str(mod210.REPORT),
        "variants": reports,
        "candidates": candidates,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf8")

    lines = [
        "# V107 Objective Stability Handoff",
        "",
        "Date: 2026-07-05",
        "Branch: `hybrid-reranker-v1`",
        "",
        "## Purpose",
        "",
        "V107 keeps only the rule slices that survived the V106 multi-fold OOF audit.",
        "",
        "## Main Variants",
        "",
    ]
    for row in reports:
        m = row["oof_metrics"]
        lines.extend(
            [
                f"- `{row['variant']}`",
                f"  - OOF rows: `{m['rows']}`",
                f"  - OOF positive errors: `{m['positive_errors']}`",
                f"  - OOF negative precision: `{m['negative_precision']:.4f}`",
                f"  - OOF min fold precision: `{m['min_fold_precision']:.4f}`",
                f"  - test hits: `{row['test_positive_hits']}`",
            ]
        )
    HANDOFF.write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
