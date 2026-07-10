from __future__ import annotations

import argparse

from trendyol.data_io import audit_raw_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trendyol relevance project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="check required raw files and their schemas")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "audit":
        results = audit_raw_data()
        for name, row_count in results.items():
            print(f"[ok] {name}: {row_count:,} rows")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
