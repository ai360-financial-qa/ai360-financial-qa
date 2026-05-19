import importlib
import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from financial_qa.base import BaseRAG

FileScopedRAG = BaseRAG
FixedSizeChunker = None
HuggingFaceEmbedder = None


def _load_rag_factory(factory_path: str) -> BaseRAG:
    module_path, _, attr = factory_path.partition(":")
    if not module_path or not attr:
        raise ValueError("RAG_FACTORY must be in form module:function")
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
    raise ValueError("BaseRAG instance is required (pass create_mcp_server(rag=...) or set RAG_FACTORY).")


def _clean_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("data/parsed/"):
        cleaned = cleaned[len("data/parsed/") :]
    return cleaned


def create_mcp_server(rag: Optional[BaseRAG] = None) -> FastMCP:
    rag_obj = _resolve_rag(rag)
    rag_obj.precalc()
    mcp = FastMCP("financial-rag")

    @mcp.tool(description="Show parsed data structure (indexed files).")
    async def list_parsed_data() -> dict[str, Any]:
        return {"structure": await rag_obj.aget_structure()}

    @mcp.tool(description="Retrieve nearest chunks from one parsed markdown file for a query.")
    async def retrieve_file_chunks(path: str, query: str) -> dict[str, Any]:
        cleaned_path = _clean_path(path)
        retrieved = await rag_obj.aretrieve(cleaned_path, query)
        return {
            "path": cleaned_path,
            "query": query,
            "chunks": [
                {
                    "text": chunk.text,
                    "doc": chunk.doc,
                    "pos": chunk.pos,
                    "score": score,
                }
                for chunk, score in retrieved
            ],
        }

    return mcp


def main() -> None:
    create_mcp_server(rag=None).run(transport="stdio")


if __name__ == "__main__":
    main()
