from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


LOG_PATH = Path("reports/submission_log.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--public-score", default="")
    parser.add_argument("--private-score", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    sub_path = Path(args.file)

    if not sub_path.exists():
        raise FileNotFoundError(sub_path)

    sub = pd.read_csv(sub_path)

    if list(sub.columns) != ["id", "prediction"]:
        raise ValueError("Submission columns must be exactly: id,prediction")

    if set(sub["prediction"].unique()) - {0, 1}:
        raise ValueError("Prediction column must contain only 0/1")

    pos_ratio = float(sub["prediction"].mean())

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": str(sub_path),
        "model_family": args.model_family,
        "pos_ratio": pos_ratio,
        "public_score": args.public_score,
        "private_score": args.private_score,
        "notes": args.notes,
    }

    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)

    print("Logged submission:")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"\nSaved: {LOG_PATH}")


if __name__ == "__main__":
    main()
