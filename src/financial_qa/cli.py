"""Simple command-line interface for the financial QA agent."""

from __future__ import annotations

import argparse
import asyncio
import sys

from financial_qa.agent.agent_loop import OpenRouterAgentLoop as AgentLoop


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Ask questions to the financial QA agent")
	parser.add_argument(
		"question",
		nargs="?",
		help="Single question to ask; if omitted, questions are read from stdin",
	)
	return parser


def print_result(answer: str, confidence: float | None = None) -> None:
	print(answer)
	if confidence is not None:
		print(f"confidence: {confidence:.3f}")


async def run_once(agent: AgentLoop, question: str) -> None:
	answer, confidence = await agent.aquery(question)
	print_result(answer, confidence)


async def main_async() -> int:
	parser = build_parser()
	args = parser.parse_args()

	agent = AgentLoop()

	if args.question:
		await run_once(agent, args.question)
		return 0

	if sys.stdin.isatty():
		print("Enter questions, one per line. Press Ctrl+Z then Enter to exit.")

	for line in sys.stdin:
		question = line.strip()
		if not question:
			continue
		await run_once(agent, question)

	return 0


def main() -> int:
	return asyncio.run(main_async())


if __name__ == "__main__":
	raise SystemExit(main())
