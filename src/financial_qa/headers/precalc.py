"""
Build the header-based hierarchical index for all *.md files in data/parsed/.

Usage (from project root):
    python -m financial_qa.headers.precalc
    python -m financial_qa.headers.precalc --name my_experiment
    python -m financial_qa.headers.precalc --backend gigachat --model GigaChat-2-Pro
    python -m financial_qa.headers.precalc --model google/gemini-2.5-flash --max-concurrent 8
"""

import argparse

from financial_qa.headers.headers_index import HeadersIndex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precalculate header-based hierarchical index"
    )
    p.add_argument("--data-dir", default="data/parsed")
    p.add_argument("--store-dir", default="indexes/headers")
    p.add_argument("--name", default="default", help="experiment database name")
    p.add_argument(
        "--backend",
        default="openrouter",
        choices=["openrouter", "gigachat", "ohmycode"],
        help="LLM backend (default: openrouter)",
    )
    p.add_argument("--apikey")
    p.add_argument(
        "--model",
        default=None,
        help="model name (default: google/gemini-2.0-flash-lite-001 for openrouter, GigaChat-2-Pro for gigachat)",
    )
    p.add_argument("--scope", default="GIGACHAT_API_PERS", help="GigaChat OAuth scope")
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="max concurrent LLM calls per file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    index = HeadersIndex(
        data_dir=args.data_dir,
        store_dir=args.store_dir,
        name=args.name,
        backend=args.backend,
        model=args.model,
        scope=args.scope,
        api_key=args.apikey,
        max_concurrent=args.max_concurrent,
    )

    print(f"Backend: {args.backend}, model: {index._model}")
    print("Starting header index precalculation...")
    index.precalc()
    print("\nDone. Indexed files:")
    print(index.get_structure())


if __name__ == "__main__":
    main()
