from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import polars as pl

from trendyol.config import (
    RAW_DIR,
    EDA_REPORTS_DIR,
    REQUIRED_RAW_FILES,
    EXPECTED_COLUMNS,
)


def ensure_dirs() -> None:
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def scan(name: str) -> pl.LazyFrame:
    return pl.scan_csv(RAW_DIR / REQUIRED_RAW_FILES[name])


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def validate_files() -> dict:
    section("1. File existence check")

    result = {}

    for key, filename in REQUIRED_RAW_FILES.items():
        path = RAW_DIR / filename
        exists = path.exists()

        result[key] = {
            "filename": filename,
            "path": str(path),
            "exists": exists,
            "size_mb": file_size_mb(path) if exists else None,
        }

        status = "OK" if exists else "MISSING"
        print(f"{status:8} {filename:35} {result[key]['size_mb']} MB")

    missing = [v["filename"] for v in result.values() if not v["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")

    return result


def validate_columns() -> dict:
    section("2. Schema / column check")

    result = {}

    for key, expected in EXPECTED_COLUMNS.items():
        path = RAW_DIR / REQUIRED_RAW_FILES[key]
        head = pl.read_csv(path, n_rows=5)
        actual = head.columns

        ok = actual == expected

        result[key] = {
            "expected_columns": expected,
            "actual_columns": actual,
            "ok": ok,
        }

        print(f"\n{key}")
        print(f"Expected: {expected}")
        print(f"Actual  : {actual}")
        print(f"Status  : {'OK' if ok else 'FAIL'}")

        if not ok:
            raise ValueError(f"Column mismatch in {key}: expected {expected}, got {actual}")

    return result


def row_counts() -> dict:
    section("3. Row counts")

    result = {}

    for key in REQUIRED_RAW_FILES:
        n = scan(key).select(pl.len().alias("n")).collect()["n"][0]
        result[key] = int(n)
        print(f"{key:20} {n:,}")

    return result


def null_report() -> dict:
    section("4. Null report")

    result = {}

    for key in REQUIRED_RAW_FILES:
        print(f"\n{key}")

        lf = scan(key)
        cols = lf.collect_schema().names()

        exprs = [
            pl.col(c).is_null().sum().alias(c)
            for c in cols
        ]

        counts = lf.select(exprs).collect()
        n_rows = scan(key).select(pl.len().alias("n")).collect()["n"][0]

        file_result = {}

        for col in cols:
            null_count = int(counts[col][0])
            null_ratio = null_count / n_rows if n_rows else 0.0

            file_result[col] = {
                "null_count": null_count,
                "null_ratio": round(null_ratio, 8),
            }

            print(f"{col:20} null={null_count:,} ratio={null_ratio:.6f}")

        result[key] = file_result

    return result


def duplicate_report() -> dict:
    section("5. Duplicate / uniqueness checks")

    result = {}

    checks = {
        "items.item_id": ("items", "item_id"),
        "terms.term_id": ("terms", "term_id"),
        "training_pairs.id": ("training_pairs", "id"),
        "submission_pairs.id": ("submission_pairs", "id"),
        "sample_submission.id": ("sample_submission", "id"),
    }

    for label, (table, col) in checks.items():
        lf = scan(table)

        stats = lf.select(
            pl.len().alias("n_rows"),
            pl.col(col).n_unique().alias("n_unique"),
        ).collect()

        n_rows = int(stats["n_rows"][0])
        n_unique = int(stats["n_unique"][0])
        duplicates = n_rows - n_unique

        result[label] = {
            "n_rows": n_rows,
            "n_unique": n_unique,
            "duplicates": duplicates,
            "ok": duplicates == 0,
        }

        print(f"{label:30} rows={n_rows:,} unique={n_unique:,} duplicates={duplicates:,}")

        if duplicates != 0:
            raise ValueError(f"Duplicate ids detected in {label}")

    return result


def label_report() -> dict:
    section("6. Label distribution")

    dist = (
        scan("training_pairs")
        .group_by("label")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("ratio")
        )
        .sort("label")
        .collect()
    )

    print(dist)

    result = {
        str(row["label"]): {
            "count": int(row["count"]),
            "ratio": float(row["ratio"]),
        }
        for row in dist.to_dicts()
    }

    labels = set(dist["label"].to_list())
    if labels != {1}:
        print("WARNING: Training data is not positive-only.")
    else:
        print("OK: Training labels are positive-only.")

    return result


def submission_format_report() -> dict:
    section("7. Submission format check")

    pairs_ids = pl.read_csv(RAW_DIR / REQUIRED_RAW_FILES["submission_pairs"], columns=["id"])
    sample_ids = pl.read_csv(RAW_DIR / REQUIRED_RAW_FILES["sample_submission"], columns=["id"])

    same_len = pairs_ids.height == sample_ids.height
    same_order = bool((pairs_ids["id"] == sample_ids["id"]).all())

    sample_pred_unique = (
        scan("sample_submission")
        .select(pl.col("prediction").unique().sort())
        .collect()["prediction"]
        .to_list()
    )

    result = {
        "same_length": same_len,
        "same_id_order": same_order,
        "sample_prediction_unique_values": sample_pred_unique,
    }

    print(f"Same length    : {same_len}")
    print(f"Same ID order  : {same_order}")
    print(f"Sample preds   : {sample_pred_unique}")

    if not same_len or not same_order:
        raise ValueError("sample_submission and submission_pairs are not aligned.")

    return result


def metadata_coverage_report() -> dict:
    section("8. Metadata coverage")

    train = scan("training_pairs").select(["term_id", "item_id"])
    test = scan("submission_pairs").select(["term_id", "item_id"])
    terms = scan("terms").select("term_id")
    items = scan("items").select("item_id")

    train_missing_terms = (
        train.select("term_id")
        .unique()
        .join(terms, on="term_id", how="anti")
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )

    test_missing_terms = (
        test.select("term_id")
        .unique()
        .join(terms, on="term_id", how="anti")
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )

    train_missing_items = (
        train.select("item_id")
        .unique()
        .join(items, on="item_id", how="anti")
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )

    test_missing_items = (
        test.select("item_id")
        .unique()
        .join(items, on="item_id", how="anti")
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )

    result = {
        "train_missing_terms": int(train_missing_terms),
        "test_missing_terms": int(test_missing_terms),
        "train_missing_items": int(train_missing_items),
        "test_missing_items": int(test_missing_items),
    }

    for k, v in result.items():
        print(f"{k:25} {v:,}")

    if any(v != 0 for v in result.values()):
        raise ValueError("Metadata coverage problem detected.")

    return result


def overlap_report() -> dict:
    section("9. Train/test overlap report")

    train = scan("training_pairs").select(["term_id", "item_id"])
    test = scan("submission_pairs").select(["term_id", "item_id"])

    train_terms = train.select("term_id").unique()
    test_terms = test.select("term_id").unique()

    train_items = train.select("item_id").unique()
    test_items = test.select("item_id").unique()

    term_overlap = (
        train_terms
        .join(test_terms, on="term_id", how="inner")
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )

    item_overlap = (
        train_items
        .join(test_items, on="item_id", how="inner")
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )

    exact_pair_overlap = (
        train.unique()
        .join(test.unique(), on=["term_id", "item_id"], how="inner")
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )

    result = {
        "term_overlap": int(term_overlap),
        "item_overlap": int(item_overlap),
        "exact_pair_overlap": int(exact_pair_overlap),
    }

    for k, v in result.items():
        print(f"{k:20} {v:,}")

    return result


def candidate_distribution_report() -> dict:
    section("10. Test candidate distribution per query")

    q = (
        scan("submission_pairs")
        .group_by("term_id")
        .agg(pl.len().alias("candidate_count"))
        .collect()
    )

    stats = q.select(
        pl.len().alias("n_terms"),
        pl.col("candidate_count").min().alias("min"),
        pl.col("candidate_count").quantile(0.05).alias("p05"),
        pl.col("candidate_count").quantile(0.10).alias("p10"),
        pl.col("candidate_count").median().alias("median"),
        pl.col("candidate_count").mean().alias("mean"),
        pl.col("candidate_count").quantile(0.90).alias("p90"),
        pl.col("candidate_count").quantile(0.95).alias("p95"),
        pl.col("candidate_count").max().alias("max"),
    )

    print(stats)

    row = stats.to_dicts()[0]
    result = {
        k: float(v) if isinstance(v, float) else int(v)
        for k, v in row.items()
    }

    return result


def write_markdown_report(report: dict) -> None:
    path = EDA_REPORTS_DIR / "audit_report.md"

    lines = []
    lines.append("# Trendyol Kaggle 2026 Data Audit Report")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Train rows: `{report['row_counts']['training_pairs']:,}`")
    lines.append(f"- Test rows: `{report['row_counts']['submission_pairs']:,}`")
    lines.append(f"- Items: `{report['row_counts']['items']:,}`")
    lines.append(f"- Terms: `{report['row_counts']['terms']:,}`")
    lines.append(f"- Train/test term overlap: `{report['overlap_report']['term_overlap']:,}`")
    lines.append(f"- Train/test item overlap: `{report['overlap_report']['item_overlap']:,}`")
    lines.append(f"- Exact train/test pair overlap: `{report['overlap_report']['exact_pair_overlap']:,}`")
    lines.append("")
    lines.append("## Key conclusion")
    lines.append("")
    lines.append(
        "Training data is positive-only. Therefore, model training requires explicit negative sampling "
        "and validation must be group-based by `term_id` rather than random row split."
    )
    lines.append("")
    lines.append("## Candidate distribution")
    lines.append("")
    for k, v in report["candidate_distribution_report"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report saved: {path}")


def main() -> None:
    ensure_dirs()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_report": validate_files(),
        "column_report": validate_columns(),
        "row_counts": row_counts(),
        "null_report": null_report(),
        "duplicate_report": duplicate_report(),
        "label_report": label_report(),
        "submission_format_report": submission_format_report(),
        "metadata_coverage_report": metadata_coverage_report(),
        "overlap_report": overlap_report(),
        "candidate_distribution_report": candidate_distribution_report(),
    }

    json_path = EDA_REPORTS_DIR / "audit_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_markdown_report(report)

    section("DONE")
    print(f"JSON report saved: {json_path}")
    print("Data audit completed successfully.")


if __name__ == "__main__":
    main()
