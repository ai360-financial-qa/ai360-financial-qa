from financial_qa.base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Slides a fixed‑length window over the text with a constant overlap between chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, filepath: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        idx = 0
        start = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        while start < len(text):
            chunk_text = text[start:start + self.chunk_size].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, doc=filepath, pos=idx))
                idx += 1
            start += step
        return chunks
