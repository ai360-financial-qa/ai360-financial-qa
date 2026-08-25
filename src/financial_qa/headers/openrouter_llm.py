"""OpenRouter async LLM client for summarization tasks."""

import os
from typing import Optional

import aiohttp

_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLM:
    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-lite-001",
        api_key: Optional[str] = None,
        base_url: Optional[str] = _API_URL,
    ):
        self.model = model
        self.base_url = base_url
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")

    async def complete(
        self,
        prompt: str,
        session: aiohttp.ClientSession,
        max_tokens: int = 512,
    ) -> str:
        async with session.post(
            self.base_url or _API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()

        if "error" in body:
            raise RuntimeError(f"OpenRouter error: {body['error']}")
        return (body["choices"][0]["message"]["content"] or "").strip()
