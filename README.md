# Financial QA

Retrieval-augmented question answering over annual reports of nine Russian banks
(Alfa, DOM.RF, Gazprombank, MKB, RSHB, Sber, Sovcombank, T-Bank, VTB; 2022–2025).

The pipeline turns PDF reports into markdown, chunks them with table-aware
strategies, embeds the chunks, and answers questions through a tool-calling agent
that retrieves per file. Answers are scored with an LLM-as-judge harness.

## Layout

```
src/financial_qa/
├── base.py           # abstract Chunker / Embedder / RAG / AgentLoop interfaces
├── cli.py            # `financial-qa` command
├── preprocessors/    # markdown cleanup of MinerU output
├── chunkers/         # fixed-size, sliding-window, semantic, table-aware strategies
├── embedders/        # HuggingFace and GigaChat embedding backends
├── rag/              # chunk-embedding index building and retrieval
├── headers/          # hierarchical header index + navigating agent loop
├── agent/            # MCP server + OpenRouter / GigaChat agent loops
└── evaluation/       # LLM-as-judge scoring

data/raw/             # source PDFs, by bank and year
data/parsed/          # MinerU markdown
data/preprocessed/    # cleaned markdown (retrieval input)
data/dataset.jsonl    # evaluation questions with golden answers
indexes/              # embedding and header indexes (generated, not tracked)
notebooks/            # chunking, agent, and scratch experiments
```

## Install

```bash
uv sync
```

Set the API keys you plan to use:

```bash
OPENROUTER_API_KEY=...   # agent loop
GIGACHAT_CREDENTIALS=... # GigaChat embedder / agent loop
```

## Usage

Build an index (writes to `indexes/`, which is not tracked in git):

```bash
python -m financial_qa.rag.precalc --data-dir data/preprocessed --store-dir indexes
```

Ask a question:

```bash
financial-qa "Как изменилась чистая прибыль ВТБ в 2024?"
```

Run the MCP server over stdio:

```bash
financial-qa-mcp
```

The server exposes two tools:

- `list_parsed_data` — the parsed file tree plus markdown headers.
- `retrieve_file_chunks(path, query, k)` — top-k nearest chunks within one file.

The agent loop injects that catalog into the prompt, lets the model pick files,
calls retrieval, and returns a grounded answer with a confidence score.

## Header-based search

The strongest configuration so far. Instead of embedding chunks, it builds a
three-level index per report — topic groups, section summaries, section text —
and lets the agent navigate it:

```bash
python -m financial_qa.headers.precalc --data-dir data/parsed --store-dir indexes/headers
financial-qa-headers --question "..."
```

Scored **86.6% (389/449)** with `Anthropic/opus-4.7` as the generator, against
83.9% for the next-best chunking strategy (summary + table-split semantic).
See `notebooks/agents/headers-search-agent.ipynb`.

## Evaluation

```python
from financial_qa.evaluation import evaluate, load_jsonl

questions = load_jsonl("data/dataset.jsonl")
scores = evaluate(questions, predictions)
```

Run logs land in `logs/` (untracked).

## Development

```bash
uv sync --extra dev
pytest
ruff check src tests
```

## License

MIT — see [LICENSE](LICENSE).
