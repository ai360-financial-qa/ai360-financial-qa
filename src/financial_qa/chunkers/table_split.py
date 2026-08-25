import re
from html.parser import HTMLParser

from financial_qa.base import BaseChunker, Chunk

_HTML_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)

# Matches financial values: parenthesised numbers, 5+ digit numbers,
# or space/comma-grouped numbers like "123 781" or "1,234".
# Excludes bare 4-digit years like "2022".
_FINANCIAL_VALUE_RE = re.compile(r"\(\d|\d{5,}|\d{1,3}[\s,]\d{3}")
_MAX_HEADER_CELL_LEN = 100  # cells longer than this belong to data rows, not header rows


class _HTMLTableParser(HTMLParser):
    """
    Extracts rows from a single HTML table as lists of cell text strings.
    Expands colspan so header columns align with data columns.
    """

    def __init__(self):
        super().__init__()
        self._depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_colspan = 1
        self._row_is_header = False
        self.rows: list[tuple[list[str], bool]] = []  # (cells, is_header_row)

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
                # Repeat cell text for each column it spans so header aligns with data.
                for _ in range(self._cell_colspan):
                    self._row.append(text)
            self._cell = None
            self._cell_colspan = 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _row_to_md(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


class TableSplitChunker(BaseChunker):
    """
    Splits non-table text with a provided text chunker and splits tables
    (both Markdown | and HTML <table>) into row-batches, prepending the
    header + separator line to each batch.

    The fallback text_chunker must expose chunk_text(text, filepath) -> list[Chunk].
    SemanticChunker satisfies this out of the box.
    """

    def __init__(self, text_chunker: BaseChunker, max_rows_per_chunk: int = 20, max_chunk_chars: int = 3000):
        self.text_chunker = text_chunker
        self.max_rows = max_rows_per_chunk
        self.max_chunk_chars = max_chunk_chars

    @staticmethod
    def _split_into_segments(text: str) -> list[dict]:
        """
        Split text into alternating {'type': 'text'|'table', 'fmt': 'html'|'markdown', 'content': str}.

        HTML <table> blocks are matched by regex first; the remaining text fragments
        are then scanned line-by-line for Markdown | tables.
        """
        segments: list[dict] = []

        # Locate all HTML table blocks and split the document around them.
        html_spans: list[tuple[int, int, str]] = [
            (m.start(), m.end(), m.group()) for m in _HTML_TABLE_RE.finditer(text)
        ]

        boundaries: list[tuple[int, int, str | None]] = []  # (start, end, html_content|None)
        prev = 0
        for start, end, html in html_spans:
            boundaries.append((prev, start, None))
            boundaries.append((start, end, html))
            prev = end
        boundaries.append((prev, len(text), None))

        for start, end, html in boundaries:
            if html is not None:
                segments.append({"type": "table", "fmt": "html", "content": html})
            else:
                fragment = text[start:end]
                if fragment.strip():
                    segments.extend(TableSplitChunker._md_segments(fragment))

        return segments

    @staticmethod
    def _md_segments(text: str) -> list[dict]:
        """Line-by-line detection of Markdown | tables within a plain-text fragment."""
        segments: list[dict] = []
        current_type: str | None = None
        current_lines: list[str] = []

        def flush():
            if current_lines:
                content = "".join(current_lines).strip()
                if content:
                    segments.append({"type": current_type, "fmt": "markdown", "content": content})

        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            is_table = stripped.startswith("|") and "|" in stripped[1:]
            seg_type = "table" if is_table else "text"
            if seg_type != current_type:
                flush()
                current_lines = []
                current_type = seg_type
            current_lines.append(line)

        flush()
        return segments

    @staticmethod
    def _parse_html_table(html: str) -> tuple[str | None, str | None, list[str]]:
        """Parse an HTML table into (header_md_row, separator_row, data_md_rows)."""
        parser = _HTMLTableParser()
        parser.feed(html)
        if not parser.rows:
            return None, None, []

        # Collect ALL leading non-data rows as the logical header.
        # A row is a "data row" once it contains financial values.
        # This captures multi-row headers like [year-spans, segment-names, section-labels].
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
            # All rows have financial values — treat first row as header anyway.
            header_row_cells = [parser.rows[0][0]]
            data_start = 1

        header_md = "\n".join(_row_to_md(cells) for cells in header_row_cells) or None
        ref_cells = header_row_cells[-1]
        separator = ("| " + " | ".join("---" for _ in ref_cells) + " |") if ref_cells else None
        data_rows = [_row_to_md(cells) for cells, _ in parser.rows[data_start:] if cells]
        return header_md, separator, data_rows

    @staticmethod
    def _parse_md_table(table_text: str) -> tuple[str | None, str | None, list[str]]:
        """Return (header_line, separator_line, data_rows) for a Markdown | table."""
        lines = [line for line in table_text.splitlines() if line.strip()]
        if not lines:
            return None, None, []
        header = lines[0]
        separator = None
        data_start = 1
        if len(lines) > 1 and all(c in "-|: " for c in lines[1].strip()):
            separator = lines[1]
            data_start = 2
        return header, separator, lines[data_start:]

    def _split_table(self, seg: dict, filepath: str) -> list[Chunk]:
        if seg["fmt"] == "html":
            header, separator, rows = self._parse_html_table(seg["content"])
        else:
            header, separator, rows = self._parse_md_table(seg["content"])

        prefix_parts = []
        if header:
            prefix_parts.append(header)
        if separator:
            prefix_parts.append(separator)
        prefix = "\n".join(prefix_parts) + "\n" if prefix_parts else ""

        if not rows:
            fallback = header or seg["content"]
            return [Chunk(text=fallback, doc=filepath, pos=0)]

        chunks: list[Chunk] = []
        for i in range(0, len(rows), self.max_rows):
            batch = rows[i : i + self.max_rows]
            chunk_text = prefix + "\n".join(batch)
            if self.max_chunk_chars > 0 and len(chunk_text) > self.max_chunk_chars:
                # Batch is too large to embed; fall back to plain text chunking of the row content.
                chunks.extend(self.text_chunker.chunk_text("\n".join(batch), filepath))
            else:
                chunks.append(Chunk(text=chunk_text, doc=filepath, pos=0))
        return chunks

    def chunk(self, filepath: str) -> list[Chunk]:
        with open(filepath, "r", encoding="utf-8") as f:
            return self.chunk_text(f.read(), filepath)

    def chunk_text(self, text: str, filepath: str = "") -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for seg in self._split_into_segments(text):
            if seg["type"] == "text":
                all_chunks.extend(self.text_chunker.chunk_text(seg["content"], filepath))
            else:
                all_chunks.extend(self._split_table(seg, filepath))
        for i, ch in enumerate(all_chunks):
            ch.pos = i
        return all_chunks
