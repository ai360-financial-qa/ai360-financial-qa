import time
import uuid
import urllib3
import requests
from openai import OpenAI
from financial_qa.base import BaseChunker, Chunk
from financial_qa.chunkers.table import TableChunker

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


class TableRowToTextChunker(BaseChunker):
    """
    Wraps a chunker (preferably TableChunker) and, for every chunk that looks
    like a Markdown table, generates a natural-language description of each
    data row. The descriptions are appended as synthetic chunks.
    """

    def __init__(
            self,
            base_chunker: BaseChunker = TableChunker,
            api_key: str = "",
            model: str = 'google/gemini-2.0-flash-lite-001',
            max_retries: int = 3,
            rows_per_summary: int = 1,
            prompt_template: str = None,
            gigachat_credentials: str = "",
            gigachat_scope: str = "GIGACHAT_API_PERS",
    ):
        self.base_chunker = base_chunker
        self.model = model
        self.max_retries = max_retries
        self.rows_per_summary = rows_per_summary
        self.gigachat_credentials = gigachat_credentials
        self.gigachat_scope = gigachat_scope
        self._gigachat_token: str = ""
        self._gigachat_token_expires_at: float = 0.0

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        ) if not gigachat_credentials else None

        self.prompt_template = prompt_template or (
            "Ты эксперт по банковской отчётности. Дана таблица с заголовками "
            "столбцов: {header}. Опиши следующую строку (или строки) данных "
            "одним-двумя предложениями, отражая все показатели. "
            "Не придумывай ничего, чего нет в данных.\n\n"
            "Строка данных:\n{row_text}\n\nОписание:"
        )

    @staticmethod
    def _is_table_chunk(chunk_text: str) -> bool:
        """Return True if chunk_text appears to be a Markdown table."""
        lines = chunk_text.strip().splitlines()
        if not lines:
            return False
        # A table must have at least one line that starts with '|'
        return any(line.strip().startswith('|') for line in lines)

    @staticmethod
    def _parse_table(chunk_text: str):
        """Extract header and data rows from a Markdown table chunk."""
        lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        if not lines:
            return None, []

        # The first line may be the header; check if the second is a separator
        header = lines[0]
        data_rows = []
        start_idx = 1
        if len(lines) > 1 and all(c in '-|: ' for c in lines[1]):
            # Second line is separator, header is first line
            start_idx = 2
        else:
            # No separator, treat first line as data as well? Usually tables have separator.
            # We'll assume first line is header, but if no separator, treat all as data.
            # For safety, if no separator, we use header=None and all lines as data.
            header = None
            start_idx = 0

        data_rows = lines[start_idx:]
        return header, data_rows

    def _get_gigachat_token(self) -> str:
        if self._gigachat_token and time.time() < self._gigachat_token_expires_at - 60:
            return self._gigachat_token
        resp = requests.post(
            _GIGACHAT_OAUTH_URL,
            headers={
                "Authorization": f"Basic {self.gigachat_credentials}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": self.gigachat_scope},
            verify=False,
        )
        resp.raise_for_status()
        data = resp.json()
        self._gigachat_token = data["access_token"]
        self._gigachat_token_expires_at = data["expires_at"] / 1000.0
        return self._gigachat_token

    def _call_gigachat(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                token = self._get_gigachat_token()
                resp = requests.post(
                    _GIGACHAT_CHAT_URL,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    verify=False,
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt + 10
                    print(f"GigaChat 429 (attempt {attempt + 1}/{self.max_retries}), sleeping {wait}s…")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"GigaChat error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return ""
        return ""

    def _call_openrouter(self, prompt: str) -> str:
        if self.gigachat_credentials:
            return self._call_gigachat(prompt)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"OpenRouter error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return ""

    def _generate_row_descriptions(self, header: str, rows: list) -> list[str]:
        descriptions = []
        for i in range(0, len(rows), self.rows_per_summary):
            batch = rows[i:i + self.rows_per_summary]
            row_text = "\n".join(batch)
            prompt = self.prompt_template.format(
                header=header or "не указаны",
                row_text=row_text
            )
            desc = self._call_openrouter(prompt)
            if desc:
                descriptions.append(desc)
        return descriptions

    def chunk(self, filepath: str = "") -> list[Chunk]:
        # 1. Получаем исходные чанки от базового чанкера
        original_chunks = self.base_chunker.chunk(filepath)
        all_chunks = []

        # 2. Обрабатываем чанки и вставляем описания сразу после таблиц
        for ch in original_chunks:
            all_chunks.append(ch)
            
            if self._is_table_chunk(ch.text):
                header, rows = self._parse_table(ch.text)
                if rows:
                    descriptions = self._generate_row_descriptions(header, rows)
                    for desc in descriptions:
                        all_chunks.append(Chunk(
                            text=desc,
                            doc=ch.doc,
                            pos=0,  # Временное значение, исправим на шаге 3
                            synth=True
                        ))

        # 3. Пересчитываем сквозные позиции (pos) для всех элементов
        for idx, ch in enumerate(all_chunks):
            ch.pos = idx

        return all_chunks
