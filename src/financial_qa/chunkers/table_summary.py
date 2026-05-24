import asyncio
from dataclasses import dataclass
from threading import Thread
from typing import List, Optional

from openai import AsyncOpenAI
from tqdm import auto as tqdm

from financial_qa.base import BaseChunker, Chunk
from financial_qa.chunkers.table_split import TableSplitChunker


@dataclass
class TableChunk(Chunk):
    """Chunk whose embedding text is an LLM-generated structural summary,
    but whose full_text (the complete table markdown) is returned to the agent on retrieval."""
    full_text: str = ""


class TableSummaryChunker(BaseChunker):
    """
    Replaces each table with a single TableChunk:
      - text      = LLM structural summary (no values) — used for embedding
      - full_text = complete table in Markdown          — returned to the agent

    Text segments are handled by the fallback text_chunker unchanged.
    """

    _DEFAULT_PROMPT = (
        "Ты индексируешь таблицу из финансового отчёта для семантического поиска. "
        "Опиши таблицу в 2–3 предложениях на русском языке, не упоминая никаких числовых значений. Укажи:\n"
        "- Какие финансовые показатели или статьи перечислены в строках (назови их)\n"
        "- Какие периоды времени или категории представлены в столбцах\n"
        "- К какому разделу или виду финансовой отчётности относится таблица\n\n"
        "Структура таблицы (только заголовки столбцов и метки строк):\n"
        "{table_structure}\n\n"
        "Описание:"
    )

    def __init__(
        self,
        text_chunker: BaseChunker,
        api_key: str = "",
        model: str = "google/gemini-2.0-flash-lite-001",
        temperature: float = 0.0,
        max_tokens: int = 200,
        max_retries: int = 3,
        max_concurrent: int = 8,
        max_table_chars: int = 4000,
        prompt_template: Optional[str] = None,
    ):
        self.text_chunker = text_chunker
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.max_table_chars = max_table_chars
        self.prompt_template = prompt_template or self._DEFAULT_PROMPT
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @staticmethod
    def _parse_segment(seg: dict) -> tuple[str, str]:
        """Return (structure_for_summary, full_markdown) for a table segment."""
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

        # Structure for summary: column headers + first cell (row label) of each data row.
        struct_parts = []
        if header:
            struct_parts.append(header)
        if sep:
            struct_parts.append(sep)
        for row in rows:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if cells:
                struct_parts.append(f"| {cells[0]} | ... |")
        structure = "\n".join(struct_parts)

        return structure, full_md

    async def _call_llm(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if hasattr(response, "choices") and response.choices:
                    return response.choices[0].message.content.strip()
                return str(response).strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"Table summary LLM failed: {e}")
                    return ""
        return ""

    async def _summarize_all(self, structures: list[str]) -> list[str]:
        sem = asyncio.Semaphore(self.max_concurrent)

        async def bounded(structure: str, pbar) -> str:
            async with sem:
                truncated = structure[: self.max_table_chars]
                prompt = self.prompt_template.format(table_structure=truncated)
                result = await self._call_llm(prompt)
                pbar.update(1)
                return result

        with tqdm.tqdm(total=len(structures), desc="Summarizing tables", unit="table", leave=False) as pbar:
            return list(await asyncio.gather(*[bounded(s, pbar) for s in structures]))

    @staticmethod
    def _run_in_thread(coro):
        output: dict = {}

        def target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                output["result"] = loop.run_until_complete(coro)
            except Exception as e:
                output["error"] = e
            finally:
                loop.close()

        t = Thread(target=target)
        t.start()
        t.join()
        if "error" in output:
            raise output["error"]
        return output["result"]

    def chunk(self, filepath: str) -> List[Chunk]:
        with open(filepath, "r", encoding="utf-8") as f:
            return self.chunk_text(f.read(), filepath)

    def chunk_text(self, text: str, filepath: str = "") -> List[Chunk]:
        segments = TableSplitChunker._split_into_segments(text)

        # Build a flat pending list; table slots start as None (filled after LLM batch).
        pending: list[Chunk | None] = []
        table_positions: list[int] = []
        table_structures: list[str] = []
        table_fulls: list[str] = []

        for seg in segments:
            if seg["type"] == "text":
                for c in self.text_chunker.chunk_text(seg["content"], filepath):
                    pending.append(c)
            else:
                structure, full_md = self._parse_segment(seg)
                if full_md.strip():
                    table_positions.append(len(pending))
                    table_structures.append(structure)
                    table_fulls.append(full_md)
                    pending.append(None)

        if table_structures:
            summaries = self._run_in_thread(self._summarize_all(table_structures))
            for pos, full_md, summary in zip(table_positions, table_fulls, summaries):
                pending[pos] = TableChunk(
                    text=summary if summary else full_md,
                    doc=filepath,
                    pos=0,
                    full_text=full_md,
                )

        chunks = [c for c in pending if c is not None]
        for i, c in enumerate(chunks):
            c.pos = i
        return chunks
