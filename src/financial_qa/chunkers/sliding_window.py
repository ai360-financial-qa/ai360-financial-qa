from pathlib import Path
from typing import List

from financial_qa.base import BaseChunker, Chunk


class SlidingWindowChunker(BaseChunker):
    """Fixed-size character sliding window with overlap, snapping breaks to whitespace."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, filepath: str) -> List[Chunk]:
        text = Path(filepath).read_text(encoding="utf-8")
        step = self.chunk_size - self.overlap
        chunks: List[Chunk] = []
        pos = 0
        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            if end < len(text):
                snap = text.rfind(" ", pos, end)
                if snap > pos:
                    end = snap
            chunk_text = text[pos:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, doc=filepath, pos=pos))
            pos += step
        return chunks