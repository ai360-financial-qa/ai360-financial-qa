import argparse
import hashlib
import importlib
import json
import os
import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp

from financial_qa.base import BaseAgentLoop, BaseRAG


def _load_rag_factory(factory_path: str) -> BaseRAG:
    module_path, _, attr = factory_path.partition(":")
    if not module_path or not attr:
        raise ValueError("RAG factory must be in form module:function")
    module = importlib.import_module(module_path)
    factory = getattr(module, attr)
    if not callable(factory):
        raise TypeError(f"RAG factory is not callable: {factory_path}")
    rag_obj = factory()
    if not isinstance(rag_obj, BaseRAG):
        raise TypeError("RAG factory must return BaseRAG instance")
    return rag_obj


def _resolve_rag(rag: Optional[BaseRAG]) -> BaseRAG:
    if rag is not None:
        return rag
    factory_path = os.environ.get("RAG_FACTORY")
    if factory_path:
        return _load_rag_factory(factory_path)
    raise ValueError(
        "BaseRAG instance is required (pass OpenRouterAgentLoop(rag=...) or set RAG_FACTORY)."
    )


def _clean_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("data/parsed/"):
        cleaned = cleaned[len("data/parsed/") :]
    return cleaned


def _resolve_log_dir(log_dir: Optional[str]) -> Optional[Path]:
    raw = log_dir or os.environ.get("AGENT_LOG_DIR")
    if raw is not None:
        if raw.strip().lower() in {"none", "off", "false"}:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    # Auto-generate a timestamped experiment directory; export via env so
    # child processes (e.g. judge workers) inherit the same path.
    experiment_id = (
        datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )
    path = Path("logs") / "runs" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_LOG_DIR"] = str(path)
    return path


def _safe_query_id(value: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip()
    )
    return safe or uuid.uuid4().hex


def _query_log_path(log_dir: Path, query_id: str) -> Path:
    return log_dir / f"{_safe_query_id(query_id)}.jsonl"


class OpenRouterAgentLoop(BaseAgentLoop):
    """OpenRouter-backed agent loop with optional action logging.

    Logs agent actions to `logs/agent` by default. Override with AGENT_LOG_DIR or
    pass log_dir, and disable by setting AGENT_LOG_DIR to 'none'/'off'.
    """

    def __init__(
        self,
        rag: Optional[BaseRAG] = None,
        model: str = "google/gemini-2.0-flash-lite-001",
        max_turns: int = 4,
        log_dir: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
    ):
        self.rag = _resolve_rag(rag)
        self.model = model
        self.max_turns = max_turns
        self.api_url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._last_confidence: Optional[float] = None
        self.log_root = _resolve_log_dir(log_dir)

    def _log_event(
        self, log_path: Optional[Path], event: str, payload: dict[str, Any]
    ) -> None:
        if not log_path:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _call_llm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")

        timeout = aiohttp.ClientTimeout(total=90)
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
                    "tool_choice": "auto",
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
                    "name": "retrieve_file_chunks",
                    "description": "Retrieve nearest chunks from one parsed markdown file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "query": {"type": "string"},
                        },
                        "required": ["path", "query"],
                    },
                },
            }
        ]

    def _build_messages(self, query_str: str) -> list[dict[str, Any]]:
        structure = self.rag.get_structure()
        system_prompt = (
            "You are a financial reports QA agent. "
            "First decide which parsed markdown files are relevant using the catalog. "
            "Then call retrieve_file_chunks for each relevant file. "
            "Files are already parsed into chunks with embeddings, "
            "so you needn't think of the RAG internals and can just call the retrieve() tool as a black box. "
            "You may call the tool multiple times. "
            "Answer only with facts grounded in retrieved chunks.\n\n"
            "FINAL ANSWER FORMAT: return strictly a JSON object with no markdown wrapping:\n"
            '{"answer": "...", "confidence": 0.0}\n'
            "confidence is a float 0.0–1.0 based on the max `score` field across all retrieved chunks. "
            "If the fact is clearly present in a top chunk (score > 0.85), set confidence close to that score. "
            "If data had to be assembled from weak chunks or is uncertain, lower confidence accordingly. "
            "If nothing relevant was found, set confidence to 0.0."
        )
        user_prompt = (
            f"Question:\n{query_str}\n\n"
            "Catalog of available report files (structure):\n"
            f"{structure}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _run_tool_call(
        self,
        call: dict[str, Any],
        query_str: str,
        log_path: Optional[Path],
        query_id: Optional[str] = None,
    ) -> tuple[str, dict[str, Any], list[float]]:
        fn = call["function"]["name"]
        if fn != "retrieve_file_chunks":
            return fn, {"error": f"Unknown tool: {fn}"}, []
        try:
            args = json.loads(call["function"]["arguments"])
            path = _clean_path(args["path"])
            sub_query = args.get("query", query_str)
            self._log_event(
                log_path,
                "tool_call",
                {
                    "query_id": query_id,
                    "tool": fn,
                    "path": path,
                    "query": sub_query,
                    "tool_call": call,
                },
            )
            results = await self.rag.aretrieve(path, sub_query)
            chunks = [
                {
                    "text": c.text,
                    "doc": c.doc,
                    "pos": c.pos,
                    "score": s,
                }
                for c, s in results
            ]
            scores = [c["score"] for c in chunks]
            tool_payload = {"path": path, "query": sub_query, "chunks": chunks}
            self._log_event(
                log_path,
                "tool_result",
                {
                    "query_id": query_id,
                    "tool": fn,
                    "path": path,
                    "chunks": len(chunks),
                    "max_score": max(scores) if scores else None,
                    "result": tool_payload,
                },
            )
            return fn, tool_payload, scores
        except Exception as e:
            self._log_event(
                log_path,
                "tool_error",
                {
                    "query_id": query_id,
                    "tool": fn,
                    "tool_call": call,
                    "error": str(e),
                },
            )
            return fn, {"error": str(e)}, []

    def _parse_final_answer(self, raw: str) -> tuple[str, Optional[float]]:
        """Extract answer and confidence from the model's JSON response.

        Strips optional markdown code fences, then parses JSON.
        Falls back to returning the raw text with no confidence on any error.
        """
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        try:
            data = json.loads(text)
            answer = str(data.get("answer", "")).strip() or raw
            confidence = data.get("confidence")
            if confidence is not None:
                confidence = max(0.0, min(1.0, float(confidence)))
            return answer, confidence
        except (json.JSONDecodeError, ValueError, TypeError):
            return raw or "Model returned an empty response.", None

    async def aquery(
        self, query_str: str, query_id: Optional[str] = None
    ) -> tuple[str, Optional[float]]:
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
                messages.append(
                    {
                        "role": "developer",
                        "content": (
                            "CRITICAL: This is the final turn. Do NOT call any tools. "
                            'Return the final answer strictly as JSON: {"answer": "...", "confidence": 0.0}. '
                            "No text outside the JSON object."
                        ),
                    }
                )

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
            body = await self._call_llm(messages=messages, tools=tools)
            choice = body["choices"][0]
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []
            self._log_event(
                log_path,
                "llm_response",
                {
                    "query_id": query_id,
                    "turn": turn,
                    "tool_calls": len(tool_calls),
                    "has_content": bool(message.get("content")),
                    "response": body,
                },
            )

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": tool_calls,
                    }
                )
                for call in tool_calls:
                    fn, tool_payload, scores = await self._run_tool_call(
                        call, query_str, log_path=log_path, query_id=query_id
                    )
                    used_scores.extend(scores)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": fn,
                            "content": json.dumps(tool_payload, ensure_ascii=False),
                        }
                    )
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
            {
                "query_id": query_id,
                "reason": "max_turns_exceeded",
                "messages": messages,
            },
        )
        return (
            "Unable to complete tool-calling loop within max_turns.",
            self._last_confidence,
        )

    def query(
        self, query_str: str, query_id: Optional[str] = None
    ) -> tuple[str, Optional[float]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aquery(query_str, query_id=query_id))
        raise RuntimeError(
            "OpenRouterAgentLoop.query cannot be called from an active event loop. Use 'await aquery(...)'."
        )


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenRouter-based file-selecting agent loop."
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-lite-001")
    parser.add_argument("--rag-factory", type=str, required=False)
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    args = parser.parse_args()

    rag = (
        _load_rag_factory(args.rag_factory) if args.rag_factory else _resolve_rag(None)
    )
    loop = OpenRouterAgentLoop(rag=rag, model=args.model, base_url=args.base_url)
    answer, confidence = await loop.aquery(args.question)
    print(answer)
    if confidence is not None:
        print(f"\nconfidence={confidence:.4f}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
