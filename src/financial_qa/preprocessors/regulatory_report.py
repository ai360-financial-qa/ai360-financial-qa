"""Preprocessing for regulatory / financial bank reports parsed to Markdown.

The markdown files under ``data/parsed/`` are produced by PDF -> markdown
converters (docling / pymupdf). They contain a number of artifacts that hurt
both retrieval quality and chunk boundary detection:

* ``![](images/<hash>.jpg)`` image references and stray ``<img>`` tags;
* ``<details><summary>...</summary>...</details>`` auto-generated image
  captions inserted by docling;
* Dotted leaders in the table of contents (``..... 89``) and inside body
  text;
* Soft-hyphenated line breaks (``финан-\nсовая``);
* Repeated whitespace / unicode NBSPs from OCR, empty markdown headings,
  trailing spaces, runs of blank lines.

``<table>...</table>`` blocks are preserved verbatim because the
``TableAwareRecursiveChunker`` and ``TableRowToTextChunker`` rely on them
to detect table boundaries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from tqdm import auto as tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from financial_qa.base import BasePreprocessor


# Sentinel used to temporarily replace ``<table>`` blocks while cleaning the
# surrounding prose. ``\x00`` cannot appear in ordinary markdown text.
_TABLE_PLACEHOLDER = "\x00TBL{idx}\x00"


class RegulatoryReportPreprocessor(BasePreprocessor):
    """Clean parsed markdown reports before chunking.

    Parameters
    ----------
    strip_images:
        Remove ``![](...)`` and ``<img ...>`` references.
    strip_details:
        Remove ``<details>...</details>`` auto-generated image captions.
    collapse_dot_leaders:
        Replace runs of dots (TOC leaders) with a single space.
    keep_tables_verbatim:
        Skip whitespace normalisation inside ``<table>`` blocks so that
        table-aware chunkers can still detect their boundaries.
    """

    _IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
    _IMAGE_HTML_RE = re.compile(r"<img[^>]*/?>", re.IGNORECASE)
    _DETAILS_RE = re.compile(r"<details>.*?</details>", re.DOTALL | re.IGNORECASE)
    _TABLE_BLOCK_RE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
    # Dot leaders, e.g. "Отчет о движении денежных средств. ..13" -> " 13"
    _DOT_LEADER_RE = re.compile(r"[.\u2026][.\u2026 ]{1,}(\d+)")
    _HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
    _MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _EMPTY_HEADING_RE = re.compile(r"^#+\s*$", re.MULTILINE)
    _TRAILING_SPACE_RE = re.compile(r"[ \t]+(\n|$)")
    _NBSP_CHARS = {"\u00a0": " ", "\u202f": " ", "\u2007": " "}

    def __init__(
        self,
        strip_images: bool = True,
        strip_details: bool = True,
        collapse_dot_leaders: bool = True,
        keep_tables_verbatim: bool = True,
    ) -> None:
        self.strip_images = strip_images
        self.strip_details = strip_details
        self.collapse_dot_leaders = collapse_dot_leaders
        self.keep_tables_verbatim = keep_tables_verbatim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """Return a cleaned version of ``text``."""
        tables: List[str] = []
        if self.keep_tables_verbatim:
            text = self._stash_tables(text, tables)

        text = self._clean(text)

        if self.keep_tables_verbatim:
            text = self._restore_tables(text, tables)
        return text

    def preprocess_file(self, src: str | Path, dst: str | Path) -> Path:
        """Read ``src`` markdown, clean it, write the result to ``dst``."""
        src_path, dst_path = Path(src), Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = self.preprocess(src_path.read_text(encoding="utf-8"))
        dst_path.write_text(cleaned, encoding="utf-8")
        return dst_path

    def preprocess_dir(
        self,
        src_dir: str | Path,
        dst_dir: str | Path,
        pattern: str = "*.md",
    ) -> List[Path]:
        """Preprocess every file matching ``pattern`` under ``src_dir``.

        Output mirrors the source directory layout under ``dst_dir``.
        """
        src = Path(src_dir)
        dst = Path(dst_dir)
        files = sorted(src.rglob(pattern))
        outputs: List[Path] = []
        for path in tqdm.tqdm(files, desc="Preprocessing reports", unit="file"):
            rel = path.relative_to(src)
            outputs.append(self.preprocess_file(path, dst / rel))
        return outputs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stash_tables(self, text: str, sink: List[str]) -> str:
        def _stash(match: re.Match) -> str:
            sink.append(match.group(0))
            return _TABLE_PLACEHOLDER.format(idx=len(sink) - 1)

        return self._TABLE_BLOCK_RE.sub(_stash, text)

    @staticmethod
    def _restore_tables(text: str, blocks: List[str]) -> str:
        for idx, block in enumerate(blocks):
            text = text.replace(_TABLE_PLACEHOLDER.format(idx=idx), block)
        return text

    def _clean(self, text: str) -> str:
        for src_ch, dst_ch in self._NBSP_CHARS.items():
            text = text.replace(src_ch, dst_ch)

        if self.strip_details:
            text = self._DETAILS_RE.sub("", text)
        if self.strip_images:
            text = self._IMAGE_MD_RE.sub("", text)
            text = self._IMAGE_HTML_RE.sub("", text)

        text = self._HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

        if self.collapse_dot_leaders:
            text = self._DOT_LEADER_RE.sub(r" \1", text)

        text = self._EMPTY_HEADING_RE.sub("", text)
        text = self._TRAILING_SPACE_RE.sub(r"\1", text)
        text = self._MULTI_SPACE_RE.sub(" ", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip() + "\n"
