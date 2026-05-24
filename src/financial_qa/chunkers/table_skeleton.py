from typing import List

from financial_qa.base import BaseChunker, Chunk
from financial_qa.chunkers.table_split import TableSplitChunker
from financial_qa.chunkers.table_summary import TableChunk


class TableSkeletonChunker(BaseChunker):
    """
    Produces one TableChunk per table where:
      - text      = structure skeleton (column headers + row labels, no values)
                    used for embedding — preserves exact terminology for retrieval
      - full_text = complete table in Markdown — returned to the agent

    No LLM calls. Text segments are handled by the fallback text_chunker unchanged.
    """

    def __init__(self, text_chunker: BaseChunker):
        self.text_chunker = text_chunker

    @staticmethod
    def _parse_segment(seg: dict) -> tuple[str, str]:
        """Return (skeleton_for_embedding, full_markdown)."""
        if seg["fmt"] == "html":
            header, sep, rows = TableSplitChunker._parse_html_table(seg["content"])
        else:
            header, sep, rows = TableSplitChunker._parse_md_table(seg["content"])

        full_parts = []
        if header:
            full_parts.append(header)
        if sep:
            full_parts.append(sep)
        full_parts.extend(rows)
        full_md = "\n".join(full_parts)

        # Skeleton: column headers + first cell (row label) of each data row.
        skel_parts = []
        if header:
            skel_parts.append(header)
        if sep:
            skel_parts.append(sep)
        for row in rows:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if cells and cells[0]:
                skel_parts.append(f"| {cells[0]} | ... |")
        skeleton = "\n".join(skel_parts)

        return skeleton, full_md

    def chunk(self, filepath: str) -> List[Chunk]:
        with open(filepath, "r", encoding="utf-8") as f:
            return self.chunk_text(f.read(), filepath)

    def chunk_text(self, text: str, filepath: str = "") -> List[Chunk]:
        segments = TableSplitChunker._split_into_segments(text)
        chunks: list[Chunk] = []

        for seg in segments:
            if seg["type"] == "text":
                chunks.extend(self.text_chunker.chunk_text(seg["content"], filepath))
            else:
                skeleton, full_md = self._parse_segment(seg)
                if full_md.strip():
                    chunks.append(TableChunk(
                        text=skeleton if skeleton.strip() else full_md,
                        doc=filepath,
                        pos=0,
                        full_text=full_md,
                    ))

        for i, c in enumerate(chunks):
            c.pos = i
        return chunks
