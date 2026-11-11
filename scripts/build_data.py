"""Build the processed product catalog from the raw Kaggle CSV.

Usage:
    python scripts/build_data.py                       # auto-finds the CSV in data/raw/
    python scripts/build_data.py --raw path/to/file.csv --out data/processed
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceshop.data import ingest  # noqa: E402


def find_raw(raw_dir: Path) -> Path:
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV in {raw_dir}. Download the Kaggle 'Amazon Product Dataset 2020' "
                 "CSV and put it there.")
    return csvs[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    raw = args.raw or find_raw(ROOT / "data" / "raw")
    rep = ingest.run(raw, args.out)
    print(rep.to_json())
    for w in rep.warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
