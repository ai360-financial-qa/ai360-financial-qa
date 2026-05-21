from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import auto as tqdm

from financial_qa.base import BaseChunker, Chunk


class TableAwareRecursiveChunker(BaseChunker):
    """Recursive chunker that uses section headers and blank lines as primary separators to keep tables intact."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, filepath: str) -> List[Chunk]:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        chunk_texts = self.splitter.split_text(text)
        return [
            Chunk(text=t.strip(), doc=filepath, pos=i)
            for i, t in tqdm.tqdm(enumerate(chunk_texts), desc="Splitting text", unit="chunk") if t.strip()
        ]
