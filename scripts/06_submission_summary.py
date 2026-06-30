from __future__ import annotations

from pathlib import Path
import pandas as pd

SUB_DIR = Path("submissions")

def main():
    rows = []

    for path in sorted(SUB_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path)
            counts = df["prediction"].value_counts().to_dict()
            ratio = df["prediction"].mean()

            rows.append(
                {
                    "file": path.name,
                    "rows": len(df),
                    "zeros": int(counts.get(0, 0)),
                    "ones": int(counts.get(1, 0)),
                    "pos_ratio": ratio,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": path.name,
                    "rows": "ERROR",
                    "zeros": "",
                    "ones": "",
                    "pos_ratio": str(exc),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["pos_ratio", "file"], ascending=[True, True])

    print(out.to_string(index=False))

    out_path = Path("reports/experiments/submission_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
