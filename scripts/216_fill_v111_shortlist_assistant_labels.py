"""Fill the v111 human-audit shortlist with assistant labels and notes."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(".")
SRC = ROOT / "reports/manual_review/v111_human_audit_shortlist.csv"
OUT = ROOT / "reports/manual_review/v112_human_audit_shortlist_assistant_filled.csv"
SUMMARY = ROOT / "reports/experiments/v112_human_audit_shortlist_assistant_filled_summary.json"

DEFAULTS = {
    "repair_general_nohead": (
        "lock_repair_family",
        "Query has no reliable head anchor; accepted repairs show semantic bait and should stay closed.",
    ),
    "repair_same_head_swap": (
        "lock_repair_family",
        "Same head alone is not enough; these swaps stay within the same surface family but still drift semantically.",
    ),
    "repair_cross_root": (
        "lock_repair_family",
        "Cross-root repair is too unstable; accepted examples often jump to a different root category.",
    ),
    "repair_device_domain": (
        "lock_repair_family",
        "Device-domain repair is unsafe; accepted adds jump across device families or accessory/main boundaries.",
    ),
}

V104_MAP = {
    "TST_0bff56cd63b6a9": ("expand_rule_family", "Main phone query matched to a case; this is a valid removal and the stricter main/accessory veto can be expanded."),
    "TST_12cb0992908e1a": ("expand_rule_family", "Tablet-case query points to the wrong model family; compatibility filtering should expand here."),
    "TST_b5989e8affed22": ("expand_rule_family", "Double-person query matched to single-person yorgan; person-count veto is valid here."),
    "TST_8bc9b8a7a30a2a": ("expand_rule_family", "Male sock query matched to a female item; gender veto looks valid."),
    "TST_0e49311b039061": ("expand_rule_family", "Female trouser query matched to male trouser; gender veto should expand."),
    "TST_79fd6e70ffdfaf": ("lock_old_broad_rule", "PS5 controller is itself an accessory intent; old main/accessory logic overreached here."),
    "TST_e22667298f0f66": ("expand_rule_family", "Male jewelry query matched to a clearly feminine design; gender veto looks useful."),
    "TST_3d774d66ce24f4": ("expand_rule_family", "Phone query matched to a case; keep expanding the certified phone main/accessory veto."),
    "TST_20a1a868415b20": ("expand_rule_family", "Phone query matched to a case; removal is correct."),
    "TST_20160475b6aac6": ("expand_rule_family", "Phone query matched to a case; removal is correct."),
    "TST_6ef55d2c50d1d5": ("expand_rule_family", "Girls' shoe query matched to a boys' boot; gender veto is valid."),
    "TST_01eb81f7250854": ("expand_rule_family", "Phone query matched to a case for a different model family; removal is correct."),
    "TST_05a7f4a8d557b1": ("expand_rule_family", "PS4 query matched to a vacuum-bag product; device-domain/main-accessory filtering should expand."),
    "TST_834ad6d2f4d989": ("expand_rule_family", "Phone query matched to a case; removal is correct."),
    "TST_7306933c92c52d": ("expand_rule_family", "Phone query matched to a case; removal is correct."),
    "TST_2ba43d030547f5": ("expand_rule_family", "3D printer query matched to toner; main/accessory or root-level filtering is valid."),
    "TST_97938e8dbb10ac": ("expand_rule_family", "Double-person pike query matched to a single-person product; person-count veto is valid."),
    "TST_3e70e993491239": ("recheck_old_rule", "The query is ambiguous enough that a phone-case removal may still be right, but this row should be rechecked before broadening."),
    "TST_4722043717f3d2": ("expand_rule_family", "Double-person sheet query matched to a single-person set; person-count veto is valid."),
    "TST_def1e155c0dc4d": ("expand_rule_family", "Phone query matched to a case; removal is correct."),
    "TST_a8f681ad682c50": ("expand_rule_family", "Phone query matched to a lens protector; valid main/accessory removal."),
    "TST_762dd26a320bd7": ("recheck_old_rule", "This sheet row looks structurally mismatched, but the size text is messy; keep it for manual recheck."),
    "TST_50822fc746c8f5": ("expand_rule_family", "iPad-case query matched to a Samsung tablet case; compatibility filtering should expand."),
    "TST_8ce92d76cc2888": ("expand_rule_family", "Girls' pajama-bottom query matched to male pajama bottoms; gender veto is valid."),
}

V108_MAP = {
    "TST_cf981fa415af2c": ("keep_veto", "Wheel size mismatch is explicit and should stay vetoed."),
    "TST_c58b26ff0fd140": ("recheck_veto", "The product is a sweatshirt but color intent is unclear; keep as a recheck rather than auto-expand."),
    "TST_174c3fa341cfd1": ("keep_veto", "Main phone query matched to a case; veto is correct."),
    "TST_47c3d16952514b": ("keep_veto", "Main phone query matched to a case for a different model; veto is correct."),
    "TST_1ef46cfc7b403e": ("recheck_veto", "Title says charger but category is smartwatch main item; likely taxonomy noise, so recheck."),
    "TST_53fcd821de3093": ("keep_veto", "Main phone query matched to a case; veto is correct."),
    "TST_d80ff36be02f9f": ("keep_veto", "Balloon query matched to a wall sticker; veto looks correct."),
    "TST_00ce00c94eda7a": ("keep_veto", "Goalkeeper glove query matched to baby winter gloves; veto looks correct."),
    "TST_c6e7aa3b1987b2": ("keep_veto", "Main phone query matched to a case; veto is correct."),
    "TST_99ebf0f2ff8fbe": ("keep_veto", "PlayStation query matched to a PS5 sticker accessory; veto is correct."),
    "TST_88d1958d847eb2": ("keep_veto", "Wheel size mismatch is explicit and should stay vetoed."),
    "TST_0fdff3bf96a14a": ("keep_veto", "Black shawl query matched to a coat; veto looks correct."),
    "TST_4edbb0ebc00fc1": ("keep_veto", "Main phone query matched to a case; veto is correct."),
    "TST_88e70b13ddeab1": ("keep_veto", "Wheel size mismatch is explicit and should stay vetoed."),
    "TST_e11bd64730f104": ("keep_veto", "Wheel size mismatch is explicit and should stay vetoed."),
    "TST_3163d1847476c4": ("likely_false_veto", "Child suit query matched to a child suit; this looks relevant and the veto is likely too aggressive."),
    "TST_7be236a65fe993": ("recheck_veto", "Vanity-chair query matched to a dining-style chair; could be off-intent, but it is close enough to recheck."),
    "TST_5bc159a0ef216d": ("keep_veto", "Long plush cardigan query matched to a baby set; veto looks correct."),
    "TST_f0a6d9ab616f9b": ("keep_veto", "Baby backpack query matched to a toy set; veto looks correct."),
    "TST_825ac40f15bdba": ("keep_veto", "Female sneaker query matched to a male sneaker; gender veto is correct."),
    "TST_795585b90a622e": ("keep_veto", "Desk-chair query matched to a scale model; veto looks correct."),
    "TST_949dcb81f94c05": ("likely_false_veto", "Broad female-shoe query matched to an actual female shoe; this veto likely removed a relevant item."),
    "TST_2527ec25c7746a": ("recheck_veto", "Basketball-shoe query matched to a running shoe from the same brand; wrong subtype is plausible but not fully deterministic."),
    "TST_9d5aa81fa374c3": ("keep_veto", "Hair-oil query matched to shampoo; product-type mismatch is strong enough to keep vetoed."),
}

HEAD_SAFE_MAP = {
    "TST_6ab49eb42bbb90": ("not_safe_repair", "The add item is a baby boot, so this is still not a safe refill for a women's sneaker-bot query."),
    "TST_6fed9567be6bf5": ("possible_safe_repair", "The add item at least matches ceket and unisex gender better than the dropped women's mont; keep only as a tiny future whitelist candidate."),
    "TST_a202d6bd39f7e6": ("possible_safe_repair", "The add item matches pantolon and male gender while the dropped row is a women's coat; this is the cleanest repair candidate in the shortlist."),
    "TST_88bb46f6223046": ("not_safe_repair", "The add item is a girls' ceket rather than an adult women's triko ceket; still not safe."),
    "TST_167a32250b5022": ("not_safe_repair", "The add item matches mont and male gender but loses the suede intent, so it is still too weak for auto-refill."),
    "TST_99273ffa56dacd": ("not_safe_repair", "The add item is a girls' mont while the query wants a women's yelek/mont concept; not safe enough."),
}


def fill_row(row: pd.Series) -> tuple[str, str]:
    bucket = row["bucket"]

    if bucket == "v104_only_vs_v107":
        return V104_MAP.get(
            row["id"],
            ("recheck_old_rule", "This old v104-only removal needs a manual recheck before turning it into a broader family decision."),
        )

    if bucket == "v108_only_removal":
        return V108_MAP.get(
            row["id"],
            ("recheck_veto", "New v108-only removal without an explicit deterministic reason should be rechecked before expansion."),
        )

    if bucket == "repair_head_safe":
        return HEAD_SAFE_MAP.get(
            row["add_id"],
            ("review_cautiously", "This repair row is more promising than the other repair families but still needs caution."),
        )

    return DEFAULTS.get(
        bucket,
        ("review_cautiously", "No default label was defined for this bucket; keep it for manual review."),
    )


def main():
    df = pd.read_csv(SRC)
    labels = df.apply(fill_row, axis=1, result_type="expand")
    labels.columns = ["human_label", "notes"]
    df["human_label"] = labels["human_label"]
    df["notes"] = labels["notes"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    summary = {
        "source_file": str(SRC),
        "output_file": str(OUT),
        "rows": int(len(df)),
        "label_counts": df["human_label"].value_counts(dropna=False).to_dict(),
        "bucket_label_counts": (
            df.groupby(["bucket", "human_label"]).size().reset_index(name="rows").to_dict(orient="records")
        ),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
