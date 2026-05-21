import re
from html.parser import HTMLParser
from typing import List

from financial_qa.base import BaseChunker, Chunk

_HTML_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_FINANCIAL_VALUE_RE = re.compile(r"\(\d|\d{5,}|\d{1,3}[\s,]\d{3}")
_MAX_HEADER_CELL_LEN = 100


class _HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_colspan = 1
        self._row_is_header = False
        self.rows: list[tuple[list[str], bool]] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._depth += 1
        if self._depth > 1:
            return
        if tag == "tr":
            self._row = []
            self._row_is_header = False
        elif tag in ("td", "th"):
            self._cell = []
            self._row_is_header = tag == "th" or self._row_is_header
            attrs_dict = dict(attrs)
            try:
                self._cell_colspan = max(1, int(attrs_dict.get("colspan", 1)))
            except (ValueError, TypeError):
                self._cell_colspan = 1

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table":
            self._depth -= 1
            return
        if self._depth > 1:
            return
        if tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append((self._row, self._row_is_header))
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            if self._row is not None:
                for _ in range(self._cell_colspan):
                    self._row.append(text)
            self._cell = None
            self._cell_colspan = 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _row_to_md(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


class TableChunker(BaseChunker):
    """
    Detects both Markdown and HTML tables, isolates them as standalone chunks,
    and processes surrounding text with a fallback chunker.
    """

    def __init__(self, fallback_chunker: BaseChunker, max_table_rows_per_chunk: int = 20):
        self.fallback = fallback_chunker
        self.max_rows = max_table_rows_per_chunk

    # ------------------------------------------------------------------
    # HTML helpers

    @staticmethod
    def _parse_html_table(html: str) -> dict:
        """Parse an HTML <table> block into the same dict shape as _find_md_tables."""
        parser = _HTMLTableParser()
        parser.feed(html)
        if not parser.rows:
            return {"full_table": html, "header": None, "rows": [], "fmt": "html"}

        header_row_cells: list[list[str]] = []
        data_start = 0
        for i, (cells, is_th) in enumerate(parser.rows):
            is_short = all(len(c) <= _MAX_HEADER_CELL_LEN for c in cells)
            if is_th or (cells and is_short and not any(_FINANCIAL_VALUE_RE.search(c) for c in cells)):
                header_row_cells.append(cells)
                data_start = i + 1
            else:
                break

        if not header_row_cells:
            header_row_cells = [parser.rows[0][0]]
            data_start = 1

        header_md = "\n".join(_row_to_md(cells) for cells in header_row_cells) or None
        data_rows = [_row_to_md(cells) for cells, _ in parser.rows[data_start:] if cells]
        return {"full_table": html, "header": header_md, "rows": data_rows, "fmt": "html"}

    # ------------------------------------------------------------------
    # Markdown helpers (original logic kept intact)

    @staticmethod
    def _find_md_tables(text: str) -> list[dict]:
        lines = text.splitlines()
        tables = []
        in_table = False
        current_rows: list[str] = []
        current_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                if not in_table:
                    in_table = True
                    current_start = i
                    current_rows = [stripped]
                else:
                    current_rows.append(stripped)
            else:
                if in_table:
                    in_table = False
                    header = current_rows[0] if current_rows else None
                    if len(current_rows) > 1 and all(c in "-|: " for c in current_rows[1]):
                        body_rows = current_rows[2:]
                    else:
                        body_rows = current_rows[:]
                    tables.append({
                        "full_table": "\n".join(current_rows),
                        "rows": body_rows,
                        "header": header,
                        "start_idx": current_start,
                        "end_idx": i - 1,
                        "fmt": "markdown",
                    })
            if in_table and i == len(lines) - 1:
                header = current_rows[0] if current_rows else None
                if len(current_rows) > 1 and all(c in "-|: " for c in current_rows[1]):
                    body_rows = current_rows[2:]
                else:
                    body_rows = current_rows[:]
                tables.append({
                    "full_table": "\n".join(current_rows),
                    "rows": body_rows,
                    "header": header,
                    "start_idx": current_start,
                    "end_idx": i,
                    "fmt": "markdown",
                })
        return tables

    def _split_large_table(self, header: str | None, rows: list) -> list[str]:
        chunks = []
        for i in range(0, len(rows), self.max_rows):
            batch = rows[i : i + self.max_rows]
            chunk_text = (header + "\n" + "\n".join(batch)) if header else "\n".join(batch)
            chunks.append(chunk_text)
        return chunks

    # ------------------------------------------------------------------
    # Public interface

    def chunk(self, filepath: str) -> List[Chunk]:
        with open(filepath, "r", encoding="utf-8") as f:
            return self._split_and_assemble(f.read(), filepath)

    def _split_and_assemble(self, text: str, filepath: str) -> List[Chunk]:
        # Collect HTML tables first (they span lines; regex is more reliable).
        html_matches = list(_HTML_TABLE_RE.finditer(text))

        if html_matches:
            return self._assemble_html(text, html_matches, filepath)

        # Fall back to the original Markdown path.
        return self._assemble_md(text, filepath)

    def _assemble_html(self, text: str, matches, filepath: str) -> List[Chunk]:
        """Interleave fallback-chunked plain text with parsed HTML table chunks."""
        all_chunks: List[Chunk] = []
        chunk_idx = 0
        prev = 0

        for m in matches:
            before = text[prev : m.start()].strip()
            if before:
                for ch in self.fallback.chunk_text(before, filepath):
                    ch.pos = chunk_idx
                    all_chunks.append(ch)
                    chunk_idx += 1

            tbl = self._parse_html_table(m.group())
            header = tbl["header"]
            rows = tbl["rows"]
            if rows and self.max_rows > 0 and len(rows) > self.max_rows:
                table_chunks = self._split_large_table(header, rows)
            elif rows:
                table_chunks = [(header + "\n" + "\n".join(rows)) if header else "\n".join(rows)]
            else:
                table_chunks = [m.group()]
            for tc in table_chunks:
                all_chunks.append(Chunk(text=tc, doc=filepath, pos=chunk_idx))
                chunk_idx += 1

            prev = m.end()

        after = text[prev:].strip()
        if after:
            for ch in self.fallback.chunk_text(after, filepath):
                ch.pos = chunk_idx
                all_chunks.append(ch)
                chunk_idx += 1

        return all_chunks

    def _assemble_md(self, text: str, filepath: str) -> List[Chunk]:
        """Original Markdown table assembly logic."""
        tables = self._find_md_tables(text)
        if not tables:
            return self.fallback.chunk_text(text, filepath)

        lines = text.splitlines()
        not_table_texts: list[str] = []
        last_line_idx = 0
        for tbl in tables:
            before = "\n".join(lines[last_line_idx : tbl["start_idx"]]).strip()
            if before:
                not_table_texts.append(before)
            last_line_idx = tbl["end_idx"] + 1
        after = "\n".join(lines[last_line_idx:]).strip()
        if after:
            not_table_texts.append(after)

        all_chunks: List[Chunk] = []
        chunk_idx = 0
        not_table_idx = 0

        for tbl in tables:
            if not_table_idx < len(not_table_texts):
                for ch in self.fallback.chunk_text(not_table_texts[not_table_idx], filepath):
                    ch.pos = chunk_idx
                    all_chunks.append(ch)
                    chunk_idx += 1
                not_table_idx += 1

            header = tbl.get("header")
            rows = tbl.get("rows", [])
            if len(rows) > self.max_rows and self.max_rows > 0:
                table_chunks = self._split_large_table(header, rows)
            else:
                table_chunks = [tbl["full_table"]]
            for tc in table_chunks:
                all_chunks.append(Chunk(text=tc, doc=filepath, pos=chunk_idx))
                chunk_idx += 1

        while not_table_idx < len(not_table_texts):
            for ch in self.fallback.chunk_text(not_table_texts[not_table_idx], filepath):
                ch.pos = chunk_idx
                all_chunks.append(ch)
                chunk_idx += 1
            not_table_idx += 1

        return all_chunks
