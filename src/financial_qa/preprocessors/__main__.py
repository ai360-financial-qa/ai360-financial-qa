"""CLI entry point.

Usage (from project root)::

    python -m financial_qa.preprocessors                                   # data/parsed -> data/preprocessed
    python -m financial_qa.preprocessors --src data/parsed --dst clean_md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from financial_qa.preprocessors.regulatory_report import RegulatoryReportPreprocessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess parsed regulatory reports")
    p.add_argument("--src", default="data/parsed", help="source directory with *.md files")
    p.add_argument("--dst", default="data/preprocessed", help="destination directory")
    p.add_argument("--pattern", default="*.md", help="filename glob to match")
    p.add_argument("--keep-images", action="store_true")
    p.add_argument("--keep-details", action="store_true")
    p.add_argument("--keep-dot-leaders", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pre = RegulatoryReportPreprocessor(
        strip_images=not args.keep_images,
        strip_details=not args.keep_details,
        collapse_dot_leaders=not args.keep_dot_leaders,
    )
    outputs = pre.preprocess_dir(args.src, args.dst, pattern=args.pattern)
    print(f"Preprocessed {len(outputs)} file(s) into {Path(args.dst).resolve()}")


if __name__ == "__main__":
    main()
