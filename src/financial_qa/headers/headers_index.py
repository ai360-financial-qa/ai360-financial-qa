"""Hierarchical header-based index for markdown financial reports.

Index structure per file (stored as JSON):
{
  "file": "alfa/2024/annual_parsed.md",
  "level2_segments": [
    {
      "id": 0,
      "theme": "Аудиторское заключение",
      "summary": "...",
      "level1_segments": [
        {"id": 0, "header": "# Title", "summary": "...", "text": "..."},
        ...
      ]
    },
    ...
  ]
}

Level-1 segment ids are local (0-based within the level-2 group).
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from tqdm import auto as tqdm

from financial_qa.headers.gigachat_llm import GigaChatLLM
from financial_qa.headers.openrouter_llm import OpenRouterLLM
from financial_qa.headers.splitter import Level1Segment, split_by_headers


_SUMMARIZE_MAX_CHARS = 6000
_GROUPING_MAX_TOKENS = 2048
_SUMMARY_MAX_TOKENS = 160  # ~60 Russian words


def _parse_groups(raw: str, n: int) -> tuple[Optional[List[Dict]], Optional[str]]:
    """Parse LLM grouping response into a validated group list.

    Expected JSON: {"themes": ["...", ...], "ends": [e1, e2, ..., en]}
    "ends" are strictly increasing exclusive end indices.
    The last value should equal n; if it doesn't we clamp it and warn.
    Reconstructed groups: [0, e1), [e1, e2), ..., [..., n).
    Returns (groups, warning_or_none).
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                return None, "no JSON object found"
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as e:
                return None, f"JSON parse error: {e}"

        themes = data.get("themes", [])
        ends = data.get("ends", [])

        if not isinstance(themes, list) or not isinstance(ends, list):
            return None, f"themes/ends not lists; keys={list(data.keys())}"
        if not ends:
            return None, "ends is empty"

        # Tolerate length mismatch by taking the shorter side
        warning = None
        k = min(len(themes), len(ends))
        if len(themes) != len(ends):
            warning = f"len mismatch: {len(themes)} themes, {len(ends)} ends — truncated to {k}"
            themes, ends = themes[:k], list(ends[:k])
        else:
            ends = list(ends)

        if not (1 <= k <= 30):
            return None, f"group count {k} out of [1, 30]"

        if ends[-1] != n:
            clamp_msg = f"last end {ends[-1]} clamped to {n}"
            warning = f"{warning}; {clamp_msg}" if warning else clamp_msg
            ends[-1] = n

        prev = 0
        for e in ends:
            if not isinstance(e, int) or e <= prev or e > n:
                return None, f"ends not strictly increasing in (0,{n}]: {ends}"
            prev = e

        groups = []
        start = 0
        for theme, end in zip(themes, ends):
            groups.append({"start": start, "end": end, "theme": theme})
            start = end
        return groups, warning
    except Exception as e:
        return None, f"exception: {e}"


def _fallback_groups(n: int) -> List[Dict]:
    """Divide n segments into consecutive groups of ~3, capped at 30 groups."""
    target = max(1, min(30, n // 3 if n >= 3 else n))
    size = max(1, (n + target - 1) // target)
    groups = []
    start = 0
    gid = 0
    while start < n:
        end = min(start + size, n)
        groups.append({"start": start, "end": end, "theme": f"Group {gid + 1}"})
        start = end
        gid += 1
    return groups


class HeadersIndex:
    """Build and query a hierarchical header-based index for parsed markdown files."""

    def __init__(
        self,
        data_dir: str | Path = "data/parsed",
        store_dir: str | Path = "indexes/headers",
        name: str = "default",
        backend: str = "openrouter",
        model: Optional[str] = None,
        # OpenRouter options
        api_key: Optional[str] = None,
        # GigaChat options
        credentials: Optional[str] = None,
        scope: str = "GIGACHAT_API_PERS",
        max_concurrent: int = 5,
    ):
        if backend not in ("openrouter", "gigachat", "ohmycode"):
            raise ValueError(
                f"backend must be 'openrouter', 'ohmycode' or 'gigachat', got {backend!r}"
            )

        # Relative paths resolve against the working directory, matching the
        # rest of the package; absolute paths are used as given.
        self.data_dir = Path(data_dir)
        self.store_dir = Path(store_dir) / name
        self.name = name
        self.max_concurrent = max_concurrent
        self._backend = backend
        self._model = model or (
            "google/gemini-2.0-flash-lite-001"
            if backend == "openrouter"
            else "GigaChat-2-Pro"
        )
        self._api_key = api_key
        self._credentials = credentials
        self._scope = scope
        self._llm: Optional[OpenRouterLLM | GigaChatLLM] = None  # created on first use

    def _store_path(self, file_key: str) -> Path:
        safe = file_key.replace("/", "__").replace("\\", "__")
        return self.store_dir / (safe + ".json")

    def _file_key(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.data_dir.resolve()))

    @property
    def _get_llm(self) -> OpenRouterLLM | GigaChatLLM:
        if self._llm is None:
            if self._backend == "openrouter":
                self._llm = OpenRouterLLM(model=self._model, api_key=self._api_key)
            elif self._backend == "ohmycode":
                self._llm = OpenRouterLLM(
                    model=self._model,
                    api_key=self._api_key,
                    base_url="https://api.ohmycode.ai/v1/chat/completions",
                )
            else:
                self._llm = GigaChatLLM(
                    model=self._model,
                    credentials=self._credentials,
                    scope=self._scope,
                )
        return self._llm

    async def _summarize_segment(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        seg: Level1Segment,
    ) -> str:
        truncated = seg.full_text[:_SUMMARIZE_MAX_CHARS]
        prompt = (
            "Кратко изложи содержание этого раздела банковского финансового отчёта "
            "в примерно 60 словах на русском языке. "
            "Обязательно упомяни: конкретные названия статей, показателей или инструментов "
            "(например: «прибыль на акцию», «торговые финансовые активы», «резерв под ОКУ»), "
            "ключевые суммы и единицы измерения (млрд/млн/тыс. руб.), даты или периоды.\n\n"
            f"{truncated}\n\nКраткое содержание:"
        )
        async with sem:
            return await self._get_llm.complete(
                prompt, session, max_tokens=_SUMMARY_MAX_TOKENS
            )

    async def _build_groups(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        segments: List[Level1Segment],
        summaries: List[str],
    ) -> List[Dict]:
        n = len(segments)
        if n == 0:
            return []

        sections_text = "\n".join(
            f"{i}: [{seg.header or '(file beginning)'}] {summ}"
            for i, (seg, summ) in enumerate(zip(segments, summaries))
        )

        prompt = (
            f"Ниже приведены краткие содержания {n} последовательных разделов "
            f"(с индексами от 0 до {n - 1}) банковского финансового отчёта.\n"
            "Сгруппируй их в тематические группы, сохраняя порядок следования.\n\n"
            "СТРОГИЕ ПРАВИЛА:\n"
            f"1. Массивы 'themes' и 'ends' ДОЛЖНЫ иметь ОДИНАКОВУЮ длину — ровно столько элементов, сколько групп.\n"
            "2. Каждая группа должна содержать не более 20 разделов.\n"
            f"3. Последний элемент 'ends' ОБЯЗАТЕЛЬНО равен {n}.\n"
            "4. 'ends' — строго возрастающий список эксклюзивных правых границ.\n"
            "5. Темы — на русском языке, 3–6 слов.\n\n"
            "Ответь ТОЛЬКО валидным JSON без каких-либо пояснений:\n"
            '{"themes": ["тема 1", "тема 2", ...], "ends": [e1, e2, ..., '
            f"{n}]}}\n\n"
            f"Разделы:\n{sections_text}"
        )

        async with sem:
            raw = await self._get_llm.complete(
                prompt, session, max_tokens=_GROUPING_MAX_TOKENS
            )

        groups, msg = _parse_groups(raw, n)
        if groups is None:
            print(
                f"    Warning: grouping failed ({msg}), using fallback. Raw: {raw[:600]!r}"
            )
            groups = _fallback_groups(n)
        elif msg:
            print(f"    Note: {msg}")
        return groups

    async def _summarize_group(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        theme: str,
        level1_summaries: List[str],
    ) -> str:
        sections = "\n".join(f"- {s}" for s in level1_summaries)
        prompt = (
            "Кратко изложи содержание этой группы разделов банковского финансового отчёта "
            "в примерно 60 словах на русском языке. "
            "Упомяни ключевые финансовые статьи, показатели и суммы.\n"
            f"Тема группы: {theme}\n\nКраткие содержания разделов:\n{sections}\n\nКраткое содержание группы:"
        )
        async with sem:
            return await self._get_llm.complete(
                prompt, session, max_tokens=_SUMMARY_MAX_TOKENS
            )

    async def _precalc_file(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        path: Path,
    ) -> None:
        text = path.read_text(encoding="utf-8")
        segments = split_by_headers(text)
        if not segments:
            return

        summaries: List[str] = list(
            await asyncio.gather(
                *[self._summarize_segment(session, sem, seg) for seg in segments]
            )
        )

        groups = await self._build_groups(session, sem, segments, summaries)

        group_summaries: List[str] = list(
            await asyncio.gather(
                *[
                    self._summarize_group(
                        session,
                        sem,
                        g["theme"],
                        [summaries[i] for i in range(g["start"], g["end"])],
                    )
                    for g in groups
                ]
            )
        )

        index: Dict[str, Any] = {
            "file": self._file_key(path),
            "level2_segments": [
                {
                    "id": gid,
                    "theme": groups[gid]["theme"],
                    "summary": group_summaries[gid],
                    "level1_segments": [
                        {
                            "id": local_idx,
                            "header": segments[global_idx].header,
                            "summary": summaries[global_idx],
                            "text": segments[global_idx].text,
                        }
                        for local_idx, global_idx in enumerate(
                            range(groups[gid]["start"], groups[gid]["end"])
                        )
                    ],
                }
                for gid in range(len(groups))
            ],
        }

        store_path = self._store_path(self._file_key(path))
        store_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def aprecalc(self) -> None:
        """Process every *.md in data_dir: split by headers, summarize, group, and save."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        md_files = sorted(self.data_dir.rglob("*.md"))

        sem = asyncio.Semaphore(self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=180)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for path in tqdm.tqdm(md_files, desc="Building header index", unit="file"):
                file_key = self._file_key(path)
                if self._store_path(file_key).exists():
                    print(f"  skip (exists): {file_key}")
                    continue
                print(f"  indexing: {file_key}")
                try:
                    await self._precalc_file(session, sem, path)
                except Exception as e:
                    print(f"  error on {file_key}: {e}")

    def precalc(self) -> None:
        asyncio.run(self.aprecalc())

    def get_structure(self) -> str:
        lines = [f"data/parsed/  (headers index: {self.name})"]
        for path in sorted(self.data_dir.rglob("*.md")):
            key = self._file_key(path)
            indexed = " [indexed]" if self._store_path(key).exists() else ""
            lines.append(f"  {key}{indexed}")
        return "\n".join(lines)

    def load_index(self, filename: str) -> Dict:
        store_path = self._store_path(filename)
        if not store_path.exists():
            raise FileNotFoundError(
                f"No header index for '{filename}'. Run precalc() first.\n"
                f"Available:\n{self.get_structure()}"
            )
        return json.loads(store_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Retrieval API (used by the agent loop in Part 2)

    def get_level2_summaries(self, filename: str) -> List[Dict]:
        """Return all level-2 groups: [{id, theme, summary}, ...]."""
        index = self.load_index(filename)
        return [
            {"id": seg["id"], "theme": seg["theme"], "summary": seg["summary"]}
            for seg in index["level2_segments"]
        ]

    def get_level1_summaries(self, filename: str, level2_id: int) -> List[Dict]:
        """Return level-1 segments in a group: [{id, header, summary}, ...]."""
        index = self.load_index(filename)
        groups = index["level2_segments"]
        if not (0 <= level2_id < len(groups)):
            raise IndexError(f"level2_id={level2_id} out of range [0, {len(groups)})")
        return [
            {"id": seg["id"], "header": seg["header"], "summary": seg["summary"]}
            for seg in groups[level2_id]["level1_segments"]
        ]

    def get_segment_text(self, filename: str, level2_id: int, level1_id: int) -> str:
        """Return full text (header + content) of a level-1 segment."""
        index = self.load_index(filename)
        groups = index["level2_segments"]
        if not (0 <= level2_id < len(groups)):
            raise IndexError(f"level2_id={level2_id} out of range")
        l1_segs = groups[level2_id]["level1_segments"]
        if not (0 <= level1_id < len(l1_segs)):
            raise IndexError(f"level1_id={level1_id} out of range [0, {len(l1_segs)})")
        seg = l1_segs[level1_id]
        if seg["header"]:
            return f"{seg['header']}\n\n{seg['text']}".strip()
        return seg["text"]

    @staticmethod
    def list_databases(store_dir: str | Path = "indexes/headers") -> List[str]:
        root = Path(store_dir)
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())
