import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from tqdm import auto as tqdm

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_qa.base import BaseRAG, BaseChunker, BaseEmbedder, Chunk


class RAG(BaseRAG):
    """
    Concrete RAG implementation.

    precalc()  — reads every *.md under data_dir, chunks, embeds, saves .npz to store_dir.
    retrieve() — loads stored embeddings for a given file key, embeds the query,
                 returns top-k (Chunk, cosine_score) pairs.
    get_structure() — lists all indexed files under data_dir.

    File keys are relative paths from data_dir, e.g. "sovkom/2022/annual_parsed.md".
    Pass the same key to retrieve() that you see in get_structure().
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        data_dir: str | Path = "data/parsed",
        store_dir: str | Path = "indexes",
        name: str = "default",
        top_k: int = 5,
        min_chunk_len: int = 150,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.min_chunk_len = min_chunk_len

        current_dir = Path(__file__).parent.resolve()
        if current_dir.name == "rag":
            project_root = current_dir.parent
        else:
            project_root = current_dir
            
        self.data_dir = project_root / data_dir
        self.store_dir = project_root / store_dir / name

        self.name = name
        self.top_k = top_k

    # ------------------------------------------------------------------
    # internal helpers

    def _store_path(self, file_key: str) -> Path:
        safe = file_key.replace("/", "__").replace("\\", "__")
        return self.store_dir / (safe + ".npz")

    def _file_key(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.data_dir.resolve()))

    def precalc(self, max_workers: int = 4) -> None:
        """Chunk and embed every *.md in data_dir, saving results to store_dir."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        md_files = sorted(self.data_dir.rglob("*.md"))

        def process_file(path: Path) -> None:
            file_key = self._file_key(path)
            chunks = self.chunker.chunk(str(path))
            if not chunks:
                return
            if self.min_chunk_len > 0:
                chunks = [c for c in chunks if len(c.text) >= self.min_chunk_len]
            if not chunks:
                return
            texts = [c.text for c in chunks]
            if hasattr(self.embedder, "embed_passages"):
                embeddings = self.embedder.embed_passages(texts)
            else:
                embeddings = [self.embedder.embed(t) for t in texts]
            emb_array = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            emb_array = np.where(norms > 0, emb_array / norms, emb_array)
            chunk_meta = np.array(
                [json.dumps(
                    {
                        "text": c.text,
                        "doc": c.doc,
                        "pos": c.pos,
                        "synth": c.synth,
                        **({"full_text": c.full_text} if hasattr(c, "full_text") and c.full_text else {}),
                    },
                    ensure_ascii=False,
                )
                for c in chunks]
            )
            np.savez_compressed(self._store_path(file_key), embeddings=emb_array, chunks=chunk_meta)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_file, p): p for p in md_files}
            with tqdm.tqdm(total=len(md_files), desc="Precalculating RAG indices", unit="file") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    def retrieve(self, filename: str, query: str) -> List[Tuple[Chunk, float]]:
        """
        Return top-k (Chunk, cosine_similarity) pairs from the given file.
        filename must match a key shown in get_structure().
        """
        store_path = self._store_path(filename)
        if not store_path.exists():
            raise FileNotFoundError(
                f"No index for '{filename}'. Run precalc() first.\n"
                f"Available files:\n{self.get_structure()}"
            )

        data = np.load(store_path, allow_pickle=True)
        embeddings: np.ndarray = data["embeddings"]   # (n, dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.where(norms > 0, embeddings / norms, embeddings)
        chunk_metas = data["chunks"]                  # (n,) of JSON strings

        query_emb = np.array(self.embedder.embed(query), dtype=np.float32)
        norm = np.linalg.norm(query_emb)
        query_emb = query_emb / norm if norm > 0 else query_emb
        scores = embeddings @ query_emb

        sorted_idx = np.argsort(scores)[::-1]
        results: List[Tuple[Chunk, float]] = []
        for idx in sorted_idx:
            meta = json.loads(chunk_metas[idx])
            if self.min_chunk_len > 0 and len(meta["text"]) < self.min_chunk_len:
                continue
            text = meta.get("full_text") or meta["text"]
            chunk = Chunk(text=text, doc=meta["doc"], pos=meta["pos"], synth=meta.get("synth"))
            results.append((chunk, float(scores[idx])))
            if len(results) >= self.top_k:
                break
        return results

    async def aretrieve(self, filename: str, query: str) -> List[Tuple[Chunk, float]]:
        return await asyncio.to_thread(self.retrieve, filename, query)

    def get_structure(self) -> str:
        lines = [f"data/parsed/  (db: {self.name})"]
        for path in sorted(self.data_dir.rglob("*.md")):
            key = self._file_key(path)
            indexed = " [indexed]" if self._store_path(key).exists() else ""
            lines.append(f"  {key}{indexed}")
        return "\n".join(lines)

    def aget_structure(self) -> str:
        return self.get_structure()

    @staticmethod
    def list_databases(store_dir: str | Path = "indexes") -> List[str]:
        """Return names of all existing experiment databases under store_dir."""
        root = Path(store_dir)
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())
