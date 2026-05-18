from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_qa.base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """Recursively splits text on the most meaningful separators (paragraph, newline, sentence, space)."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, filepath: str = "") -> list[Chunk]:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.chunk_text(text, filepath)

    def chunk_text(self, text: str, filepath: str = "") -> list[Chunk]:
        chunk_texts = self.splitter.split_text(text)
        return [
            Chunk(text=t.strip(), doc=filepath, pos=i)
            for i, t in enumerate(chunk_texts) if t.strip()
        ]
