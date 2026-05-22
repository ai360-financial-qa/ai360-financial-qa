import asyncio
import ast
import hashlib
import json
import re
import random
import ssl
import time
import uuid
from typing import Any, Optional

import aiohttp
import httpx
from gigachat.api.auth import auth_async

from financial_qa.agent.agent_loop import OpenRouterAgentLoop, _clean_path, _query_log_path
from financial_qa.base import BaseRAG

_TOKEN_ENDPOINT = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_MAX_RETRIES = 3
_MAX_RATE_LIMIT_RETRIES = 8
_BASE_DELAY = 2.0
_RATE_LIMIT_BASE_DELAY = 15.0
_RATE_LIMIT_MAX_DELAY = 60.0


class GigaChatAgentLoop(OpenRouterAgentLoop):
    """Agent loop that authenticates against GigaChat's OAuth2 endpoint.

    Keeps a cached access token and refreshes it when it expires.
    Uses GigaChat's `functions`/`function_call` API format instead of OpenAI's
    `tools`/`tool_calls` format. Falls back to parsing text-based function calls
    when the model emits them in prose instead of structured API calls.
    """

    def __init__(
        self,
        rag: Optional[BaseRAG] = None,
        model: str = "GigaChat-2-Pro",
        max_turns: int = 4,
        log_dir: Optional[str] = None,
        base_url: str = "https://gigachat.devices.sberbank.ru/api/v1",
        credentials: Optional[str] = None,
        scope: str = "GIGACHAT_API_PERS",
    ):
        import os
        self._credentials = credentials or os.environ.get("GIGACHAT_CREDENTIALS")
        if not self._credentials:
            raise EnvironmentError("GIGACHAT_CREDENTIALS is not set")
        self._scope = scope
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        # Lock serializes refreshes; held only for the duration of the HTTP call,
        # NOT during backoff sleeps, so waiting tasks aren't blocked for multiple retries.
        self._token_lock = asyncio.Lock()
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE
        super().__init__(rag=rag, model=model, max_turns=max_turns, log_dir=log_dir, base_url=base_url)

    async def _fetch_token_once(self) -> None:
        """Single attempt to refresh the token via the gigachat SDK auth helper."""
        async with httpx.AsyncClient(verify=False) as client:
            token = await auth_async(client, url=_TOKEN_ENDPOINT, credentials=self._credentials, scope=self._scope)
        self._access_token = token.access_token
        self._token_expires_at = token.expires_at / 1000.0  # ms → s

    async def _get_token(self) -> str:
        """Return a valid access token, fetching one if needed."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        # Serialize refreshes: only one task fetches at a time, others wait and
        # then return the cached result. The lock is NOT held during backoff sleep.
        async with self._token_lock:
            if self._access_token and time.time() < self._token_expires_at - 60:
                return self._access_token
            await self._fetch_token_once()
        return self._access_token  # type: ignore[return-value]

    async def warmup(self) -> None:
        """Pre-fetch the OAuth token so parallel queries start with a valid token.

        Call this once before launching concurrent aquery() tasks to avoid all
        workers racing to hit the token endpoint simultaneously.
        """
        await self._get_token()

    def _openai_tools_to_giga_functions(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        functions: list[dict[str, Any]] = []
        for t in tools:
            name = t["function"]["name"]
            return_parameters: dict[str, Any]
            if name == "retrieve_file_chunks":
                return_parameters = {
                    "type": "object",
                    "properties": {
                        "chunks": {"type": "array", "description": "Retrieved text chunks"},
                    },
                }
            elif name == "calculate":
                return_parameters = {
                    "type": "object",
                    "properties": {
                        "result": {"type": "number", "description": "Calculated numeric result"},
                    },
                }
            else:
                return_parameters = {"type": "object"}
            functions.append(
                {
                    "name": name,
                    "description": t["function"]["description"],
                    "parameters": t["function"]["parameters"],
                    "return_parameters": return_parameters,
                }
            )
        return functions

    def _tool_spec(self) -> list[dict[str, Any]]:
        tools = super()._tool_spec()
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": (
                        "Safely evaluate a numeric arithmetic expression. "
                        "Input: {expression: string}. Supports parentheses, unary +/-, and operators +, -, *, /. "
                        "Returns {result: number}. Does not allow variables, functions, exponentiation, modulo, or other syntax."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                },
            }
        )
        return tools

    def _build_messages(self, query_str: str) -> list[dict[str, Any]]:
        structure = self.rag.get_structure()
        system_prompt = (
            "Ты — ассистент для ответов на вопросы по финансовым отчётам банков. "
            "Перед тем как ответить, ОБЯЗАТЕЛЬНО вызови функцию retrieve_file_chunks "
            "для получения нужных фрагментов из документов. "
            "Для любых арифметических вычислений (суммы, разницы, проценты, доли) "
            "используй функцию calculate. "
            "Интерфейс calculate: вход — {expression: строка}, выход — {result: число}. "
            "Поддерживает скобки, унарные +/-, и операторы +, -, *, /. "
            "Не поддерживает переменные, функции, возведение в степень или модуль. "
            "Используй каталог файлов ниже, чтобы выбрать подходящий файл. "
            "Можешь вызывать функцию несколько раз для разных файлов.\n\n"
            "Правила извлечения данных:\n"
            "- Для вопросов о доле, соотношении или проценте (слова «доля», «отношение», «процент», «во сколько раз») "
            "ВСЕГДА делай два отдельных вызова retrieve_file_chunks: "
            "первый — с запросом на числитель, второй — с запросом на знаменатель. "
            "В запросах указывай точную сущность, дату и строку таблицы.\n"
            "- В параметре query каждого вызова указывай конкретную строку/статью и дату, "
            "например: «итого активы Альфа-Банк 31 декабря 2024».\n\n"
            "Правила ответа:\n"
            "- Финальный ответ возвращай СТРОГО в формате JSON (без markdown-обёртки):\n"
            '  {"answer": "...", "confidence": 0.0}\n'
            "- answer — ТОЛЬКО итоговое число, дату или факт — без объяснений и шагов.\n"
            "- confidence — число от 0.0 до 1.0, отражающее уверенность в ответе.\n"
            "  Основывай confidence на поле score фрагментов, полученных из retrieve_file_chunks:\n"
            "  возьми максимальный score среди всех полученных фрагментов и используй его как базу.\n"
            "  Если нужный факт явно присутствует в топ-фрагменте (score > 0.85), ставь confidence близко к score.\n"
            "  Если пришлось собирать из нескольких слабых фрагментов или данные неточны — снижай confidence.\n"
            "  Если информация не найдена — confidence = 0.0.\n"
            "- Если нужна доля/процент: укажи оба числа из фрагментов и вычисленный результат. "
            "Пример: «X составляет 1 234 млн руб., Y составляет 5 678 млн руб., доля = 21,7%».\n"
            "- Если нужно изменение: вычисли разницу и напиши. "
            "Пример: «выросли на 450 млн руб. (с 1 369 до 1 819 млн руб.)».\n"
            "- НИКОГДА не пиши 'необходимо', 'следует', 'для расчёта нужно' и т.п.\n"
            "- Опирайся только на данные из полученных фрагментов."
        )
        user_prompt = (
            f"Вопрос:\n{query_str}\n\n"
            "Каталог доступных файлов отчётов:\n"
            f"{structure}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_text_function_call(self, content: str, fallback_query: str) -> Optional[dict[str, Any]]:
        """Parse a text-based function call when GigaChat emits one as prose instead of a structured API call."""
        patterns = [
            r'retrieve_file_chunks\s*\(\s*["\']([^"\']+\.md)["\']',
            r'retrieve_file_chunks\s*\(\s*path\s*=\s*["\']([^"\']+\.md)["\']',
            r'retrieve_file_chunks\s*\(\s*["\']([^"\']+)["\']',
        ]
        for pat in patterns:
            m = re.search(pat, content or "")
            if m:
                return {
                    "name": "retrieve_file_chunks",
                    "arguments": {"path": m.group(1), "query": fallback_query},
                }
        calc_patterns = [
            r'calculate\s*\(\s*expression\s*=\s*["\']([^"\']+)["\']\s*\)',
            r'calculate\s*\(\s*["\']([^"\']+)["\']\s*\)',
        ]
        for pat in calc_patterns:
            m = re.search(pat, content or "")
            if m:
                return {
                    "name": "calculate",
                    "arguments": {"expression": m.group(1)},
                }
        return None

    def _safe_eval_expression(self, expression: str) -> float:
        expr = expression.strip()
        if not expr:
            raise ValueError("Empty expression")

        node = ast.parse(expr, mode="eval")

        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                value = _eval(n.operand)
                return value if isinstance(n.op, ast.UAdd) else -value
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left = _eval(n.left)
                right = _eval(n.right)
                if isinstance(n.op, ast.Add):
                    return left + right
                if isinstance(n.op, ast.Sub):
                    return left - right
                if isinstance(n.op, ast.Mult):
                    return left * right
                return left / right
            raise ValueError("Unsupported expression")

        return _eval(node)

    async def _call_llm(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], force_call: bool = False) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=90)
        functions = self._openai_tools_to_giga_functions(tools)
        function_call: Any = {"name": "retrieve_file_chunks"} if force_call else "auto"
        rate_limit_attempts = 0
        attempt = 0
        while True:
            try:
                token = await self._get_token()
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.api_url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "functions": functions,
                            "function_call": function_call,
                            "temperature": 0.0,
                        },
                        ssl=self._ssl_ctx,
                    ) as resp:
                        if resp.status == 429:
                            rate_limit_attempts += 1
                            if rate_limit_attempts > _MAX_RATE_LIMIT_RETRIES:
                                resp.raise_for_status()
                            retry_after = resp.headers.get("Retry-After")
                            delay = min(
                                float(retry_after) if retry_after else (
                                    _RATE_LIMIT_BASE_DELAY * (1.5 ** (rate_limit_attempts - 1)) + random.uniform(0, 5)
                                ),
                                _RATE_LIMIT_MAX_DELAY,
                            )
                            print(f"[GigaChat] 429 rate limit (attempt {rate_limit_attempts}/{_MAX_RATE_LIMIT_RETRIES}), sleeping {delay:.0f}s…", flush=True)
                            await asyncio.sleep(delay)
                            continue
                        resp.raise_for_status()
                        body = await resp.json()
                if "error" in body:
                    raise RuntimeError(f"LLM API error: {body['error']}")
                return body
            except (aiohttp.ClientError, httpx.HTTPError, RuntimeError) as e:
                # Only invalidate the token on 401 — any other error means the token
                # is still valid, and requesting a new one would trigger a 400 from
                # GigaChat's OAuth endpoint (it rejects refreshes before expiry).
                is_401 = (
                    (isinstance(e, aiohttp.ClientResponseError) and e.status == 401)
                    or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 401)
                )
                if is_401:
                    self._access_token = None
                if attempt >= _MAX_RETRIES - 1:
                    raise
                attempt += 1
                await asyncio.sleep(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
        return {}  # unreachable

    async def aquery(self, query_str: str, query_id: Optional[str] = None) -> tuple[str, Optional[float]]:
        self._last_confidence = None
        query_id = query_id or hashlib.md5(query_str.encode("utf-8")).hexdigest()
        log_path = _query_log_path(self.log_root, query_id) if self.log_root else None
        self._log_event(
            log_path,
            "query_start",
            {"query_id": query_id, "question": query_str, "model": self.model},
        )
        messages = self._build_messages(query_str)
        tools = self._tool_spec()
        used_scores: list[float] = []

        for turn in range(1, self.max_turns + 1):
            if turn == self.max_turns:
                messages.append({
                    "role": "user",
                    "content": (
                        "Это последний шаг. НЕ вызывай функции. "
                        'Дай финальный ответ строго в формате JSON: {"answer": "...", "confidence": 0.0}. '
                        "Никакого текста вне JSON."
                    ),
                })

            self._log_event(
                log_path,
                "llm_request",
                {
                    "query_id": query_id,
                    "turn": turn,
                    "message_count": len(messages),
                    "tool_count": len(tools),
                    "messages": messages,
                    "tools": tools,
                },
            )
            body = await self._call_llm(messages=messages, tools=tools, force_call=(turn == 1))
            choice = body["choices"][0]
            message = choice["message"]

            # GigaChat returns a single function_call dict, not a tool_calls array.
            # Fall back to parsing a text-based call when the model emits one as prose.
            fc = message.get("function_call")
            if not fc and turn < self.max_turns:
                fc = self._parse_text_function_call(message.get("content", ""), query_str)

            # If turn 1 produced no function call at all, inject a corrective prompt
            # and let the loop continue so the model is forced to retrieve on turn 2.
            if not fc and turn == 1:
                messages.append({
                    "role": "assistant",
                    "content": message.get("content") or "",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Ты не вызвал функцию retrieve_file_chunks. "
                        "Это ОБЯЗАТЕЛЬНЫЙ первый шаг. "
                        "Выбери подходящий файл из каталога и вызови функцию сейчас."
                    ),
                })
                self._log_event(log_path, "llm_response", {
                    "query_id": query_id, "turn": turn,
                    "tool_calls": 0, "has_content": bool(message.get("content")),
                    "response": body,
                })
                continue

            self._log_event(
                log_path,
                "llm_response",
                {
                    "query_id": query_id,
                    "turn": turn,
                    "tool_calls": 1 if fc else 0,
                    "has_content": bool(message.get("content")),
                    "response": body,
                },
            )

            if fc:
                args = fc.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                fn = fc["name"]

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "function_call": fc,
                }
                if "functions_state_id" in message:
                    assistant_msg["functions_state_id"] = message["functions_state_id"]
                messages.append(assistant_msg)

                # Execute the tool
                tool_payload: dict[str, Any]
                scores: list[float]
                if fn == "retrieve_file_chunks":
                    try:
                        path = _clean_path(args.get("path", ""))
                        sub_query = args.get("query", query_str)
                        self._log_event(log_path, "tool_call", {
                            "query_id": query_id, "tool": fn,
                            "path": path, "query": sub_query, "tool_call": fc,
                        })
                        results = await self.rag.aretrieve(path, sub_query)
                        chunks = [{"text": c.text, "doc": c.doc, "pos": c.pos, "score": s} for c, s in results]
                        scores = [c["score"] for c in chunks]
                        tool_payload = {"path": path, "query": sub_query, "chunks": chunks}
                        self._log_event(log_path, "tool_result", {
                            "query_id": query_id, "tool": fn, "path": path,
                            "chunks": len(chunks), "max_score": max(scores) if scores else None,
                            "result": tool_payload,
                        })
                    except Exception as e:
                        self._log_event(log_path, "tool_error", {
                            "query_id": query_id, "tool": fn, "tool_call": fc, "error": str(e),
                        })
                        tool_payload = {"error": str(e)}
                        scores = []
                elif fn == "calculate":
                    try:
                        expression = args.get("expression", "")
                        self._log_event(log_path, "tool_call", {
                            "query_id": query_id, "tool": fn, "expression": expression, "tool_call": fc,
                        })
                        result = self._safe_eval_expression(expression)
                        tool_payload = {"expression": expression, "result": result}
                        scores = []
                        self._log_event(log_path, "tool_result", {
                            "query_id": query_id, "tool": fn, "result": tool_payload,
                        })
                    except Exception as e:
                        self._log_event(log_path, "tool_error", {
                            "query_id": query_id, "tool": fn, "tool_call": fc, "error": str(e),
                        })
                        tool_payload = {"error": str(e)}
                        scores = []
                else:
                    tool_payload = {"error": f"Unknown function: {fn}"}
                    scores = []

                used_scores.extend(scores)
                messages.append({
                    "role": "function",
                    "name": fn,
                    "content": json.dumps(tool_payload, ensure_ascii=False),
                })
                continue

            raw = (message.get("content") or "").strip()
            answer, model_confidence = self._parse_final_answer(raw)
            if model_confidence is not None:
                self._last_confidence = model_confidence
            elif used_scores:
                self._last_confidence = max(used_scores)
            self._log_event(
                log_path,
                "query_complete",
                {
                    "query_id": query_id,
                    "answer": answer,
                    "confidence": self._last_confidence,
                    "messages": messages,
                },
            )
            return answer, self._last_confidence

        self._log_event(
            log_path,
            "query_failed",
            {"query_id": query_id, "reason": "max_turns_exceeded", "messages": messages},
        )
        return "Unable to complete tool-calling loop within max_turns.", self._last_confidence
