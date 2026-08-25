"""Split markdown text into level-1 segments by single-# headers."""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Level1Segment:
    header: str  # "# Title" or "" for pre-header content
    text: str    # content after the header line (stripped)

    @property
    def full_text(self) -> str:
        if self.header:
            return f"{self.header}\n\n{self.text}".strip()
        return self.text


def split_by_headers(text: str) -> List[Level1Segment]:
    """Split markdown by H1 headers (single #). Pre-header content is kept as segment 0."""
    lines = text.split("\n")
    segments: List[Level1Segment] = []
    current_header = ""
    current_lines: List[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content or current_header:
            segments.append(Level1Segment(header=current_header, text=content))

    for line in lines:
        if re.match(r"^# ", line):
            flush()
            current_header = line.rstrip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return segments
