"""
Run precalculation: chunk and embed all *.md files in data/parsed/.
Usage (from project root):
    python rag/precalc.py
    python rag/precalc.py --chunk-size 800 --overlap 150
    python rag/precalc.py --model intfloat/multilingual-e5-base --top-k 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_qa.chunkers import SlidingWindowChunker
from financial_qa.embedders import HFEmbedder
from financial_qa.preprocessors import RegulatoryReportPreprocessor
from financial_qa.rag.rag import RAG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precalculate RAG embeddings")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--data-dir", default="data/parsed")
    p.add_argument("--store-dir", default="indexes")
    p.add_argument("--name", default="default", help="experiment database name")
    p.add_argument(
        "--preprocess",
        action="store_true",
        help="Run RegulatoryReportPreprocessor on --data-dir before indexing",
    )
    p.add_argument(
        "--preprocessed-dir",
        default="data/preprocessed",
        help="Where to write cleaned markdown when --preprocess is set",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    if args.preprocess:
        print(f"Preprocessing {data_dir} -> {args.preprocessed_dir}...")
        outputs = RegulatoryReportPreprocessor().preprocess_dir(
            data_dir, args.preprocessed_dir
        )
        print(f"Preprocessed {len(outputs)} file(s).")
        data_dir = args.preprocessed_dir

    chunker = SlidingWindowChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    embedder = HFEmbedder(model_name=args.model)
    rag = RAG(
        chunker=chunker,
        embedder=embedder,
        data_dir=data_dir,
        store_dir=args.store_dir,
        name=args.name,
        top_k=args.top_k,
    )

    print("Starting precalculation...")
    rag.precalc()
    print("\nDone. Indexed files:")
    print(rag.get_structure())


if __name__ == "__main__":
    main()
