import asyncio
import random
import time
import threading
from typing import List

from gigachat import GigaChat
from tqdm import auto as tqdm

from financial_qa.base import BaseEmbedder

_EMBED_MAX_RETRIES = 6
_EMBED_BASE_DELAY = 10.0
_EMBED_MAX_DELAY = 60.0
_EMBED_MAX_CHARS = 4000


def _truncate(text: str) -> str:
    return text[:_EMBED_MAX_CHARS] if len(text) > _EMBED_MAX_CHARS else text


def _embed_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on 429 with capped exponential backoff."""
    for attempt in range(_EMBED_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" not in str(e) or attempt == _EMBED_MAX_RETRIES - 1:
                raise
            delay = min(_EMBED_BASE_DELAY * (1.5 ** attempt) + random.uniform(0, 3), _EMBED_MAX_DELAY)
            print(f"[GigaEmbedder] 429 on embeddings (attempt {attempt + 1}/{_EMBED_MAX_RETRIES}), sleeping {delay:.0f}s…", flush=True)
            time.sleep(delay)


class GigaEmbedder(BaseEmbedder):
    """Embedder backed by the GigaChat Embeddings REST API."""

    def __init__(
        self,
        credentials: str,
        model: str = "EmbeddingsGigaR",
        verify_ssl_certs: bool = False,
        scope: str = "GIGACHAT_API_PERS",
        batch_size: int = 100,
    ):
        self.credentials = credentials
        self.model = model
        self.verify_ssl_certs = verify_ssl_certs
        self.scope = scope
        self.batch_size = batch_size
        self._giga: GigaChat | None = None
        self._giga_lock = threading.Lock()

    def _get_client(self) -> GigaChat:
        if self._giga is None:
            with self._giga_lock:
                if self._giga is None:
                    self._giga = GigaChat(
                        credentials=self.credentials,
                        verify_ssl_certs=self.verify_ssl_certs,
                        scope=self.scope,
                    )
                    self._giga.__enter__()
        return self._giga

    def embed(self, text: str) -> List[float]:
        response = _embed_with_retry(self._get_client().embeddings, [_truncate(text)], model=self.model)
        return response.data[0].embedding

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        texts = [_truncate(t) for t in texts]
        giga = self._get_client()
        with tqdm.tqdm(total=len(texts), desc="Embedding", unit="chunk", leave=False) as pbar:
            self._embed_batch(giga, texts, results, pbar)
        return results

    def _embed_batch(self, giga, texts: List[str], out: List[List[float]], pbar=None) -> None:
        if not texts:
            return
        try:
            response = _embed_with_retry(giga.embeddings, texts, model=self.model)
            out.extend(e.embedding for e in response.data)
            if pbar is not None:
                pbar.update(len(texts))
        except Exception as e:
            if "413" in str(e) and len(texts) > 1:
                mid = len(texts) // 2
                print(f"[GigaEmbedder] 413 — splitting batch {len(texts)} → {mid}+{len(texts)-mid}", flush=True)
                self._embed_batch(giga, texts[:mid], out, pbar)
                self._embed_batch(giga, texts[mid:], out, pbar)
            else:
                raise

    async def aembed(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed, text)

    async def aembed_passages(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_passages, texts)
