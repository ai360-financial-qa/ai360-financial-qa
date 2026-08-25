"""Minimal GigaChat async LLM client for summarization tasks."""

import asyncio
import os
import random
import ssl
import time
from typing import Any, Dict, Optional

import aiohttp
import httpx
from gigachat.api.auth import auth_async

_TOKEN_ENDPOINT = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
_MAX_RETRIES = 3
_MAX_RATE_LIMIT_RETRIES = 8
_RATE_LIMIT_BASE_DELAY = 15.0
_RATE_LIMIT_MAX_DELAY = 60.0
_BASE_DELAY = 2.0


class GigaChatLLM:
    """OAuth2-authenticated GigaChat client for async completion calls."""

    def __init__(
        self,
        model: str = "GigaChat-2-Pro",
        credentials: Optional[str] = None,
        scope: str = "GIGACHAT_API_PERS",
    ):
        self.model = model
        self._credentials = credentials or os.environ.get("GIGACHAT_CREDENTIALS")
        if not self._credentials:
            raise EnvironmentError("GIGACHAT_CREDENTIALS is not set")
        self._scope = scope
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    async def _fetch_token_once(self) -> None:
        async with httpx.AsyncClient(verify=False) as client:
            token = await auth_async(
                client,
                url=_TOKEN_ENDPOINT,
                credentials=self._credentials,
                scope=self._scope,
            )
        self._access_token = token.access_token
        self._token_expires_at = token.expires_at / 1000.0

    async def _get_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        async with self._token_lock:
            if self._access_token and time.time() < self._token_expires_at - 60:
                return self._access_token
            await self._fetch_token_once()
        return self._access_token  # type: ignore[return-value]

    async def complete(
        self,
        prompt: str,
        session: aiohttp.ClientSession,
        max_tokens: int = 512,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }

        rate_limit_attempts = 0
        attempt = 0
        while True:
            try:
                token = await self._get_token()
                async with session.post(
                    _API_URL,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    ssl=self._ssl_ctx,
                ) as resp:
                    if resp.status == 429:
                        rate_limit_attempts += 1
                        if rate_limit_attempts > _MAX_RATE_LIMIT_RETRIES:
                            resp.raise_for_status()
                        retry_after = resp.headers.get("Retry-After")
                        delay = min(
                            float(retry_after) if retry_after else (
                                _RATE_LIMIT_BASE_DELAY * (1.5 ** (rate_limit_attempts - 1))
                                + random.uniform(0, 5)
                            ),
                            _RATE_LIMIT_MAX_DELAY,
                        )
                        print(
                            f"[GigaChat] 429 rate limit (attempt {rate_limit_attempts}), "
                            f"sleeping {delay:.0f}s…",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    body = await resp.json()

                if "error" in body:
                    raise RuntimeError(f"GigaChat API error: {body['error']}")
                return (body["choices"][0]["message"]["content"] or "").strip()

            except (aiohttp.ClientError, httpx.HTTPError, RuntimeError) as e:
                is_401 = (
                    isinstance(e, aiohttp.ClientResponseError) and e.status == 401
                ) or (
                    isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 401
                )
                if is_401:
                    self._access_token = None
                if attempt >= _MAX_RETRIES - 1:
                    raise
                attempt += 1
                await asyncio.sleep(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
        return ""  # unreachable
