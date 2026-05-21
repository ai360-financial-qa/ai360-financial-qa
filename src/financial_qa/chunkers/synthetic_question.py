import time
import uuid
import threading
import urllib3
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from openai import OpenAI
from tqdm import tqdm
from financial_qa.base import BaseChunker, Chunk
from financial_qa.chunkers.semantic import SemanticChunker

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


class SyntheticQuestionChunker(BaseChunker):
    """
    Generates synthetic questions for each original chunk and indexes them
    as separate chunks (with synth=True).
    """

    def __init__(
            self,
            base_chunker: BaseChunker = SemanticChunker,
            api_key: str = "",
            model: str = "google/gemini-2.0-flash-lite-001",
            questions_per_chunk: int = 3,
            temperature: float = 0.3,
            max_tokens: int = 256,
            max_retries: int = 3,
            prompt_template: str = None,
            gigachat_credentials: str = "",
            gigachat_scope: str = "GIGACHAT_API_PERS",
            max_workers: int = 1,
    ):
        self.base_chunker = base_chunker
        self.questions_per_chunk = questions_per_chunk
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.gigachat_credentials = gigachat_credentials
        self.gigachat_scope = gigachat_scope
        self.max_workers = max_workers
        self._gigachat_token: str = ""
        self._gigachat_token_expires_at: float = 0.0
        self._token_lock = threading.Lock()

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        ) if not gigachat_credentials else None

        self.prompt_template = prompt_template or (
            "Ты эксперт по банковской отчётности. На основе приведённого фрагмента "
            "придумай {n} различных вопросов, на которые можно найти ответ в этом тексте. "
            "Вопросы должны быть сформулированы естественно, как задал бы их аналитик. "
            "Выведи только вопросы, каждый на отдельной строке, без нумерации.\n\n"
            "Фрагмент:\n{chunk_text}"
        )

    def _get_gigachat_token(self) -> str:
        if self._gigachat_token and time.time() < self._gigachat_token_expires_at - 60:
            return self._gigachat_token
        with self._token_lock:
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

    def _call_gigachat(self, prompt: str) -> list[str]:
        for attempt in range(self.max_retries):
            try:
                token = self._get_gigachat_token()
                resp = requests.post(
                    _GIGACHAT_CHAT_URL,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    verify=False,
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt + 10
                    print(f"GigaChat 429 (attempt {attempt + 1}/{self.max_retries}), sleeping {wait}s…")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                questions = [line.strip() for line in content.splitlines() if line.strip()]
                return questions[:self.questions_per_chunk]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"GigaChat error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return []
        return []

    def _call_openrouter(self, chunk_text: str) -> list[str]:
        prompt = self.prompt_template.format(n=self.questions_per_chunk, chunk_text=chunk_text)
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
                content = response.choices[0].message.content.strip()
                questions = [line.strip() for line in content.splitlines() if line.strip()]
                return questions[:self.questions_per_chunk]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"OpenRouter error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return []
        return []

    def chunk(self, filepath: str = "") -> List[Chunk]:
        original_chunks = self.base_chunker.chunk(filepath)
        all_chunks = list(original_chunks)
        next_pos = len(all_chunks)

        synth_by_idx: dict[int, list[str]] = {}

        def _generate(idx: int, ch: Chunk) -> tuple[int, list[str]]:
            return idx, self._call_openrouter(ch.text)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_generate, i, ch): i for i, ch in enumerate(original_chunks)}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"synth-q {filepath.split('/')[-1]}", leave=False):
                idx, questions = future.result()
                synth_by_idx[idx] = questions

        for i in range(len(original_chunks)):
            for q in synth_by_idx.get(i, []):
                all_chunks.append(Chunk(text=q, doc=filepath, pos=next_pos, synth=True))
                next_pos += 1

        return all_chunks
