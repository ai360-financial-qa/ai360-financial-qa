"""Hierarchical header-based agent loop using OpenRouter.

The agent navigates the three-level header index:
  1. get_level2_summaries(filename)              → topic groups
  2. get_level1_summaries(filename, level2_id)   → section summaries within a group
  3. get_segment_text(filename, level2_id, level1_id) → full section text
"""

import argparse
import asyncio
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp

from financial_qa.base import BaseAgentLoop
from financial_qa.headers.headers_index import HeadersIndex


def _resolve_log_dir(log_dir: Optional[str]) -> Optional[Path]:
    raw = log_dir or os.environ.get("AGENT_LOG_DIR")
    if raw is not None:
        if raw.strip().lower() in {"none", "off", "false"}:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    experiment_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    path = Path("logs") / "runs" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_LOG_DIR"] = str(path)
    return path


def _safe_query_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or uuid.uuid4().hex


def _query_log_path(log_dir: Path, query_id: str) -> Path:
    return log_dir / f"{_safe_query_id(query_id)}.jsonl"


class HeadersAgentLoop(BaseAgentLoop):
    """OpenRouter agent loop that navigates the hierarchical header index.

    Hierarchical navigation strategy:
      1. get_level2_summaries — browse topic groups for a file
      2. get_level1_summaries — read section summaries within a group
      3. get_segment_text     — fetch full text of a specific section
    """

    def __init__(
        self,
        index: Optional[HeadersIndex] = None,
        model: str = "google/gemini-2.0-flash-lite-001",
        max_turns: int = 10,
        log_dir: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        store_name: str = "default",
        api_key: Optional[str] = None,
    ):
        self.index = index or HeadersIndex(name=store_name)
        self.model = model
        self.max_turns = max_turns
        self.api_url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._last_confidence: Optional[float] = None
        self.log_root = _resolve_log_dir(log_dir)

    def _log_event(self, log_path: Optional[Path], event: str, payload: dict[str, Any]) -> None:
        if not log_path:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_answer: bool = False,
    ) -> dict[str, Any]:
        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "none" if force_answer else "auto",
                    "temperature": 0.0,
                },
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
        if "error" in body:
            raise RuntimeError(f"LLM API error: {body['error']}")
        return body

    def _tool_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_level2_summaries",
                    "description": (
                        "Get all topic groups (level-2) for a file. "
                        "Returns [{id, theme, summary}, ...]. "
                        "Call this first to understand which part of a file is relevant."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Path relative to the parsed data root, e.g. 'alfa/2022/annual_parsed.md'",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_level1_summaries",
                    "description": (
                        "Get section summaries (level-1) within a topic group. "
                        "Returns [{id, header, summary}, ...]. "
                        "Use this to identify which specific section has the data you need."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "level2_id": {
                                "type": "integer",
                                "description": "Topic group id from get_level2_summaries",
                            },
                        },
                        "required": ["filename", "level2_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_segment_text",
                    "description": (
                        "Get the full text of a specific section. "
                        "Call this after identifying the exact section via get_level1_summaries."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "level2_id": {"type": "integer"},
                            "level1_id": {
                                "type": "integer",
                                "description": "Section id within the group from get_level1_summaries",
                            },
                        },
                        "required": ["filename", "level2_id", "level1_id"],
                    },
                },
            },
        ]

    def _build_messages(self, query_str: str) -> list[dict[str, Any]]:
        structure = self.index.get_structure()
        system_prompt = (
            "Ты — ассистент для ответов на вопросы по финансовым отчётам банков. "
            "У тебя есть иерархический индекс отчётов. Навигируй по нему шаг за шагом:\n"
            "  1. Вызови get_level2_summaries(filename), чтобы увидеть тематические группы файла.\n"
            "  2. Вызови get_level1_summaries(filename, level2_id), чтобы увидеть краткие содержания разделов в группе.\n"
            "  3. Вызови get_segment_text(filename, level2_id, level1_id), чтобы прочитать полный текст раздела.\n"
            "Можешь вызывать инструменты для нескольких файлов или разделов.\n\n"
            "Правила навигации:\n"
            "- Не повторяй вызовы get_level2_summaries для одного и того же файла — запомни результат и используй его.\n"
            "- Если get_segment_text вернул ошибку «уже получен» — немедленно попробуй другой раздел (другой level1_id или level2_id).\n"
            "- Если нужный раздел не нашёлся в очевидной группе, пробуй смежные группы или group_id=0 (там часто находятся основные финансовые отчёты — баланс, P&L).\n"
            "- Числовые данные часто хранятся в сводных таблицах в начале документа, а не только в примечаниях.\n\n"
            "Правила извлечения данных:\n"
            "- При чтении таблицы всегда фиксируй единицу измерения из её заголовка (млрд, млн, тыс. руб.) и указывай её в ответе.\n"
            "- Для вопросов о доле, соотношении или проценте (слова «доля», «отношение», «процент», «во сколько раз») "
            "ВСЕГДА получай полный текст разделов, содержащих числитель и знаменатель по отдельности.\n\n"
            "Правила ответа:\n"
            "- Давай ТОЛЬКО итоговое число, дату или факт — без объяснений и шагов.\n"
            "- Если нужна доля/процент: укажи оба числа из разделов и вычисленный результат. "
            "Пример: «X составляет 1 234 млн руб., Y составляет 5 678 млн руб., доля = 21,7%».\n"
            "- Если нужно изменение: проверь знак разницы (новое минус старое) и напиши. "
            "Пример: «снизился на 54,1 млрд руб. (с 172,6 до 118,5 млрд руб.)».\n"
            "- НИКОГДА не пиши «необходимо», «следует», «для расчёта нужно» и т.п.\n"
            "- Опирайся только на данные из полученных разделов. Отвечай на русском языке."
        )
        user_prompt = (
            f"Вопрос:\n{query_str}\n\n"
            "Доступные индексированные файлы отчётов:\n"
            f"{structure}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _dispatch_tool(
        self,
        fn: str,
        args: dict[str, Any],
        seen_segments: set[tuple],
    ) -> Any:
        filename = args.get("filename", "")
        if fn == "get_level2_summaries":
            return self.index.get_level2_summaries(filename)
        if fn == "get_level1_summaries":
            return self.index.get_level1_summaries(filename, int(args["level2_id"]))
        if fn == "get_segment_text":
            key = (filename, int(args["level2_id"]), int(args["level1_id"]))
            if key in seen_segments:
                raise ValueError(
                    f"Раздел уже получен (filename={filename}, "
                    f"level2_id={key[1]}, level1_id={key[2]}). "
                    "Попробуй другой раздел."
                )
            seen_segments.add(key)
            return self.index.get_segment_text(filename, key[1], key[2])
        raise ValueError(f"Unknown tool: {fn}")

    async def _execute_tool_call(
        self,
        call: dict[str, Any],
        log_path: Optional[Path],
        query_id: Optional[str],
        seen_segments: set[tuple],
    ) -> tuple[str, dict[str, Any]]:
        fn = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"])
            self._log_event(log_path, "tool_call", {
                "query_id": query_id, "tool": fn, "args": args,
            })
            result = self._dispatch_tool(fn, args, seen_segments)
            self._log_event(log_path, "tool_result", {
                "query_id": query_id, "tool": fn, "args": args, "result": result,
            })
            return fn, {"result": result}
        except Exception as e:
            self._log_event(log_path, "tool_error", {
                "query_id": query_id, "tool": fn, "call": call, "error": str(e),
            })
            return fn, {"error": str(e)}

    async def aquery(
        self, query_str: str, query_id: Optional[str] = None
    ) -> tuple[str, Optional[float]]:
        self._last_confidence = None
        query_id = query_id or hashlib.md5(query_str.encode("utf-8")).hexdigest()
        log_path = _query_log_path(self.log_root, query_id) if self.log_root else None
        self._log_event(log_path, "query_start", {
            "query_id": query_id, "question": query_str, "model": self.model,
        })
        messages = self._build_messages(query_str)
        tools = self._tool_spec()
        seen_segments: set[tuple] = set()

        for turn in range(1, self.max_turns + 1):
            force_answer = turn == self.max_turns
            if force_answer:
                messages.append({
                    "role": "user",
                    "content": "Это последний шаг. НЕ вызывай инструменты. Дай финальный ответ на русском языке, опираясь на уже полученные данные.",
                })

            self._log_event(log_path, "llm_request", {
                "query_id": query_id, "turn": turn,
                "message_count": len(messages), "messages": messages, "tools": tools,
            })
            body = await self._call_llm(messages=messages, tools=tools, force_answer=force_answer)
            choice = body["choices"][0]
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []
            self._log_event(log_path, "llm_response", {
                "query_id": query_id, "turn": turn,
                "tool_calls": len(tool_calls), "has_content": bool(message.get("content")),
                "response": body,
            })

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                })
                for call in tool_calls:
                    fn, payload = await self._execute_tool_call(call, log_path, query_id, seen_segments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": fn,
                        "content": json.dumps(payload, ensure_ascii=False),
                    })
                continue

            answer = message.get("content") or ""
            if not answer:
                answer = "Model returned an empty response."
            self._log_event(log_path, "query_complete", {
                "query_id": query_id, "answer": answer, "messages": messages,
            })
            return answer, None

        self._log_event(log_path, "query_failed", {
            "query_id": query_id, "reason": "max_turns_exceeded", "messages": messages,
        })
        return "Unable to complete tool-calling loop within max_turns.", None

    def query(
        self, query_str: str, query_id: Optional[str] = None
    ) -> tuple[str, Optional[float]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aquery(query_str, query_id=query_id))
        raise RuntimeError(
            "HeadersAgentLoop.query cannot be called from an active event loop. Use 'await aquery(...)'."
        )


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Run hierarchical header-based agent loop."
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-lite-001")
    parser.add_argument("--store-name", type=str, default="default")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    index = HeadersIndex(name=args.store_name)
    loop = HeadersAgentLoop(
        index=index,
        model=args.model,
        max_turns=args.max_turns,
        log_dir=args.log_dir,
        base_url=args.base_url,
    )
    answer, _ = await loop.aquery(args.question)
    print(answer)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
