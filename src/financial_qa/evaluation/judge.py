import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, cast

from tqdm import tqdm

import aiohttp

# ============================================================
# Data Format Documentation
# ============================================================
# GOLDEN ANSWERS FORMAT (JSONL or Dict):
# {
#     "question_id": str,           # Unique identifier
#     "question": str,              # The question text
#     "gold_answer": str,           # The correct answer
#     "gold_evidence": list,        # Supporting evidence (can be empty list [])
# }
#
# PREDICTED ANSWERS FORMAT (JSONL or Dict):
# {
#     "question_id": str,           # Must match a golden question_id
#     "question": str,              # The question text
#     "answer": str,                # The predicted answer
#     "evidence": list,             # Supporting evidence (can be empty list [])
# }
#
# Both formats can be provided as:
# - JSONL file: one JSON object per line
# - Dict: dictionary keyed by question_id
# ============================================================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEFAULT_MODEL = "google/gemini-2.0-flash-lite-001"  # cheap, reliable
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds

# Required fields in each JSON object
REQUIRED_FIELDS = {"question_id", "question"}
GOLD_FIELD = "gold_answer"
GOLD_EVIDENCE_FIELD = "gold_evidence"  # field in golden answers
PRED_FIELD = "answer"  # expected field in predicted answers
PRED_EVIDENCE_FIELD = "evidence"  # expected field in predicted answers

# ------------------------------------------------------------
# Prompt template
# ------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for a question‑answering system. "
    "Your task is to decide whether the predicted answer correctly matches the gold answer "
    "for the given question. The answers are short, often containing technical data and numbers. "
    "Consider semantic equivalence: numbers may be expressed in different units or formats "
    "(e.g., '1,151 тыс. бел. руб.' vs '1151 тысяч белорусских рублей'), but the underlying "
    "value must be identical. Minor variations in wording that do not change the factual meaning "
    "are acceptable.\n\n"
    "Respond ONLY with a JSON object containing exactly two fields:\n"
    '- "score": 1 if the predicted answer is correct, 0 if incorrect.\n'
    '- "reasoning": a brief explanation of your decision in Russian (or English if preferred).\n\n'
    'Example: {"score": 1, "reasoning": "Both answers state the same amount."}'
)


def _resolve_log_dir() -> Optional[Path]:
    raw = os.environ.get("AGENT_LOG_DIR")
    if raw is not None:
        if raw.strip().lower() in {"none", "off", "false"}:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    # Auto-generate a timestamped experiment directory; export via env so
    # child processes inherit the same path.
    experiment_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    path = Path("logs") / "runs" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_LOG_DIR"] = str(path)
    return path


def _safe_query_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or uuid.uuid4().hex


def _query_log_path(query_id: str) -> Optional[Path]:
    log_root = _resolve_log_dir()
    if not log_root:
        return None
    return log_root / f"{_safe_query_id(query_id)}.jsonl"


def _append_query_log(query_id: str, event: str, payload: Dict[str, Any]) -> None:
    log_path = _query_log_path(query_id)
    if not log_path:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "query_id": query_id,
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_judge_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    evidence: Optional[List[Dict]] = None,
    include_evidence: bool = True,
) -> str:
    """Construct a user message for the judge."""
    prompt = (
        f"Question: {question}\n\n"
        f"Gold answer: {gold_answer}\n\n"
        f"Predicted answer: {predicted_answer}\n\n"
    )

    if include_evidence and evidence:
        evidence_str = json.dumps(evidence, ensure_ascii=False, indent=2)
        prompt += f"Supporting evidence (document references): {evidence_str}\n\n"

    prompt += "Evaluate the predicted answer against the gold answer."
    return prompt


# ------------------------------------------------------------
# File validation and loading
# ------------------------------------------------------------
def load_jsonl(filepath: str) -> Dict[str, dict]:
    """
    Load a JSONL file and return a dict keyed by question_id.
    Raises ValueError on any format error.
    """
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{filepath}:{line_no} – invalid JSON: {e}")

            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                raise ValueError(
                    f"{filepath}:{line_no} – missing required fields: {missing}"
                )

            qid = record["question_id"]
            if qid in data:
                raise ValueError(
                    f"{filepath}:{line_no} – duplicate question_id '{qid}'"
                )

            data[qid] = record
    if not data:
        raise ValueError(f"{filepath} – file contains no valid records")
    return data


# ------------------------------------------------------------
# OpenRouter API call
# ------------------------------------------------------------
async def call_openrouter(
    model: str,
    system_prompt: str,
    user_message: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """Send a chat completion request to OpenRouter and return the assistant message."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    # url = "https://api.ohmycode.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                return result["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, KeyError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"OpenRouter API call failed after {MAX_RETRIES} attempts: {e}"
                    )
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return ""  # unreachable


def parse_judge_response(raw_response: str) -> Dict[str, Any]:
    """Extract JSON from the LLM response, expecting {'score': int, 'reasoning': str}."""
    clean = raw_response.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse JSON from response:\n{raw_response}")
        else:
            raise ValueError(f"No JSON object found in response:\n{raw_response}")

    if "score" not in result or "reasoning" not in result:
        raise ValueError(f"Missing 'score' or 'reasoning' in judge output: {result}")
    if result["score"] not in (0, 1):
        raise ValueError(f"Invalid score value: {result['score']} (must be 0 or 1)")
    return result


def _build_result(payload: Dict[str, Any], score: Optional[int], reasoning: str) -> Dict[str, Any]:
    return {
        "index": payload.get("index", 0),
        "question_id": payload.get("question_id", ""),
        "question": payload.get("question", ""),
        "gold_answer": payload.get("gold_answer", ""),
        "predicted_answer": payload.get("predicted_answer", ""),
        "judge_score": score,
        "judge_reasoning": reasoning,
    }


def _build_error_result(payload: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    return _build_result(payload, None, f"JUDGE ERROR: {error}")


async def _evaluate_single_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    question_id = payload.get("question_id", "")
    question = payload["question"]
    # Derive the same file key as the agent: MD5 of the question text.
    # This guarantees agent and judge always append to the same JSONL file.
    log_key = hashlib.md5(question.encode("utf-8")).hexdigest()
    prompt = build_judge_prompt(
        question,
        payload["gold_answer"],
        payload["predicted_answer"],
        payload.get("evidence", []),
        include_evidence=payload.get("include_evidence", True),
    )
    _append_query_log(
        log_key,
        "judge_request",
        {
            "question_id": question_id,
            "model": payload["model"],
            "system_prompt": JUDGE_SYSTEM_PROMPT,
            "user_prompt": prompt,
        },
    )
    try:
        raw_response = await call_openrouter(
            payload["model"],
            JUDGE_SYSTEM_PROMPT,
            prompt,
            payload["api_key"],
            temperature=payload.get("temperature", 0.0),
            max_tokens=payload.get("max_tokens", 256),
        )
        judge_result = parse_judge_response(raw_response)
        _append_query_log(
            log_key,
            "judge_response",
            {
                "question_id": question_id,
                "raw_response": raw_response,
                "parsed_response": judge_result,
                "verdict": judge_result.get("score"),
            },
        )
        return _build_result(payload, judge_result["score"], judge_result["reasoning"])
    except Exception as e:
        _append_query_log(
            log_key,
            "judge_error",
            {
                "question_id": question_id,
                "error": str(e),
            },
        )
        return _build_error_result(payload, e)


def _evaluate_single_process(payload: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(_evaluate_single_async(payload))


# ------------------------------------------------------------
# Core evaluation function (module‑level)
# ------------------------------------------------------------
async def evaluate_async(
    golden: Union[str, Dict[str, dict]],
    predicted: Union[str, Dict[str, dict]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    detailed_result: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 256,
    include_evidence: bool = True,
    use_processes: bool = True,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    progress_desc: str = "Judging",
) -> Union[int, Dict[str, Any]]:
    """
    Run the LLM‑as‑judge pipeline.

    Args:
        golden: Path to JSONL file with gold answers, or parsed data dict.
        predicted: Path to JSONL file with predicted answers, or parsed data dict.
        model: OpenRouter model name (default: 'openai/gpt-4o-mini').
        api_key: OpenRouter API key; if None, reads OPENROUTER_API_KEY env var.
        detailed_result: If False, returns the number of correct answers (int).
                         If True, returns a dict with keys:
                             'correct' (int),
                             'total' (int),
                             'errors' (int),
                             'results' (list of dicts).
        temperature: LLM temperature (default 0.0).
        max_tokens: Max tokens for the judge response (default 256).
        include_evidence: If True, include evidence in the judge prompt; if False, omit it (default: True).
        use_processes: If True, spawn a separate process per query by default (can be capped by max_workers).
            Uses the "spawn" context for cross-platform compatibility.
        max_workers: Max number of judge worker processes (defaults to total questions if use_processes).
        show_progress: If True, render tqdm progress.
        progress_desc: Description label for tqdm.

    Returns:
        int or dict as described above.
    Raises:
        ValueError: on invalid input files or no common question_ids.
        RuntimeError: on API failure after retries.
    """
    model = model or DEFAULT_MODEL
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenRouter API key provided. Set OPENROUTER_API_KEY or pass api_key."
        )

    # Load files or use provided dicts
    if isinstance(golden, str):
        golden_data = load_jsonl(golden)
    elif isinstance(golden, dict):
        golden_data = golden
    else:
        raise ValueError(f"golden must be str or dict, got {type(golden)}")

    if isinstance(predicted, str):
        predicted_data = load_jsonl(predicted)
    elif isinstance(predicted, dict):
        predicted_data = predicted
    else:
        raise ValueError(f"predicted must be str or dict, got {type(predicted)}")

    # Align
    common_ids = set(golden_data) & set(predicted_data)
    if not common_ids:
        raise ValueError("No common question_ids found between the two files.")

    payloads = []
    for idx, qid in enumerate(sorted(common_ids), start=1):
        gold_rec = golden_data[qid]
        pred_rec = predicted_data[qid]

        payloads.append(
            {
                "index": idx,
                "question_id": qid,
                "question": gold_rec["question"],
                "gold_answer": gold_rec[GOLD_FIELD],
                "predicted_answer": pred_rec[PRED_FIELD],
                "evidence": pred_rec.get(PRED_EVIDENCE_FIELD, []),
                "model": model,
                "api_key": api_key,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "include_evidence": include_evidence,
            }
        )

    total = len(payloads)
    correct = 0
    errors = 0
    results = []

    # Resolve (and set AGENT_LOG_DIR) in the parent process so spawned workers inherit it.
    _resolve_log_dir()

    if use_processes:
        if max_workers is None:
            max_workers = total
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            loop = asyncio.get_running_loop()
            futures = {
                loop.run_in_executor(executor, _evaluate_single_process, payload): payload
                for payload in payloads
            }
            progress = (
                tqdm(total=total, desc=progress_desc, unit="question")
                if show_progress
                else None
            )
            for future in futures.keys():
                payload = futures[future]
                try:
                    result = await future
                except Exception as e:
                    result = _build_error_result(payload, e)
                results.append(result)
                if result["judge_score"] == 1:
                    correct += 1
                if result["judge_score"] is None:
                    errors += 1
                if progress:
                    progress.update(1)
                    progress.set_postfix(
                        correct=correct,
                        errors=errors,
                        accuracy=f"{(correct / len(results)):.2%}",
                    )
            if progress:
                progress.close()
    else:
        if max_workers is None:
            max_workers = min(16, total)
        semaphore = asyncio.Semaphore(max_workers)

        async def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await _evaluate_single_async(payload)

        tasks = {
            asyncio.create_task(_run(payload)): payload for payload in payloads
        }
        progress = (
            tqdm(total=total, desc=progress_desc, unit="question")
            if show_progress
            else None
        )
        for task in asyncio.as_completed(tasks):
            payload = tasks[task]
            try:
                result = await task
            except Exception as e:
                result = _build_error_result(payload, e)
            results.append(result)
            if result["judge_score"] == 1:
                correct += 1
            if result["judge_score"] is None:
                errors += 1
            if progress:
                progress.update(1)
                progress.set_postfix(
                    correct=correct,
                    errors=errors,
                    accuracy=f"{(correct / len(results)):.2%}",
                )
        if progress:
            progress.close()

    results.sort(key=lambda r: r.get("index", 0))
    for rec in results:
        rec.pop("index", None)

    if detailed_result:
        return {
            "correct": correct,
            "total": total,
            "errors": errors,
            "results": results,
        }
    else:
        return correct


def evaluate(
    golden: Union[str, Dict[str, dict]],
    predicted: Union[str, Dict[str, dict]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    detailed_result: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 256,
    include_evidence: bool = True,
    use_processes: bool = True,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    progress_desc: str = "Judging",
) -> Union[int, Dict[str, Any]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            evaluate_async(
                golden=golden,
                predicted=predicted,
                model=model,
                api_key=api_key,
                detailed_result=detailed_result,
                temperature=temperature,
                max_tokens=max_tokens,
                include_evidence=include_evidence,
                use_processes=use_processes,
                max_workers=max_workers,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
        )
    raise RuntimeError("evaluate() cannot run inside an active event loop. Use await evaluate_async().")


# ------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------
async def main_async():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted answers against golden answers using an LLM judge via OpenRouter."
    )
    parser.add_argument("golden_file", help="Path to JSONL file with golden answers")
    parser.add_argument(
        "predicted_file", help="Path to JSONL file with predicted answers"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSONL file for detailed results (if omitted, only the number of correct answers is printed)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key", help="OpenRouter API key (overrides env var OPENROUTER_API_KEY)"
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Exclude evidence from the judge prompt",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of judge worker processes (default: one per question)",
    )
    parser.add_argument(
        "--no-processes",
        action="store_true",
        help="Run judge in the current process instead of spawning workers",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress output",
    )
    args = parser.parse_args()

    try:
        if args.output:
            # Full evaluation with detailed output
            result = cast(
                Dict[str, Any],
                await evaluate_async(
                    golden=args.golden_file,
                    predicted=args.predicted_file,
                    model=args.model,
                    api_key=args.api_key,
                    detailed_result=True,
                    include_evidence=not args.no_evidence,
                    use_processes=not args.no_processes,
                    max_workers=args.processes,
                    show_progress=not args.no_progress,
                ),
            )
            # Write JSONL
            with open(args.output, "w", encoding="utf-8") as f:
                for rec in result["results"]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Print summary to stderr
            print(f"Total evaluated: {result['total']}", file=sys.stderr)
            print(f"Correct: {result['correct']}", file=sys.stderr)
            if result["total"] > 0:
                print(
                    f"Accuracy: {result['correct']/result['total']:.2%}",
                    file=sys.stderr,
                )
            if result["errors"]:
                print(
                    f"Errors (score not obtained): {result['errors']}", file=sys.stderr
                )
        else:
            # Simple count mode
            correct_count = await evaluate_async(
                golden=args.golden_file,
                predicted=args.predicted_file,
                model=args.model,
                api_key=args.api_key,
                detailed_result=False,
                include_evidence=not args.no_evidence,
                use_processes=not args.no_processes,
                max_workers=args.processes,
                show_progress=not args.no_progress,
            )
            print(correct_count)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
