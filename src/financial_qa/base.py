from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ==========================================
# Data Structures
# ==========================================


@dataclass
class Chunk:
    """
    Represents a chunk of text as defined in the notes:
    text: str
    doc: str (assuming document ID or reference)
    pos: int (position/offset)
    synth?: bool (optional flag for synthetic chunks)
    """

    text: str
    doc: str
    pos: int
    synth: Optional[bool] = None


# ==========================================
# Component Interfaces
# ==========================================


class BaseChunker(ABC):
    """
    Chunker exposes: chunk: str -> list[Chunk] (+ synth)
    """

    @abstractmethod
    def chunk(self, filepath: str) -> List[Chunk]:
        """Splits a document into a list of Chunks from path."""
        pass


class BasePreprocessor(ABC):
    """
    Preprocessor exposes: preprocess: str -> str

    Takes raw document text (typically markdown produced by a PDF parser) and
    returns a cleaned version ready for chunking/embedding.
    """

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Return a cleaned version of ``text``."""
        pass


class BaseEmbedder(ABC):
    """
    Embedder exp: embed: str -> R^n
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Takes a string and returns an n-dimensional vector (R^n)."""
        pass

    @abstractmethod
    async def aembed(self, text: str) -> List[float]:
        """Asynchronous embedding method."""
        pass

    @abstractmethod
    async def aembed_passages(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous passage embedding method."""
        pass


# ==========================================
# System Interfaces
# ==========================================


class BaseRAG(ABC):
    """
    RAG obj: contains chunker and embedder.
    Exposes retrieval and precalculation methods.
    Stores embeddings for chunks in each file.
    """

    chunker: BaseChunker
    embedder: BaseEmbedder

    @abstractmethod
    def retrieve(self, filename: str, query: str) -> List[Tuple[Chunk, float]]:
        """
        Retrieves the most relevant chunks from the given file, and returns pairs of (chunk, confidence score).
        """
        pass

    @abstractmethod
    def precalc(self) -> None:
        """
        Handles precalculation phase (chunking and embedding into the DB).
        """
        pass

    @abstractmethod
    def get_structure(self) -> str:
        """
        Returns the structure of the data directory, so as to allow the agent to choose which file to pull the context from.
        """
        pass

    @abstractmethod
    async def aretrieve(self, filename: str, query: str) -> List[Tuple[Chunk, float]]:
        """Asynchronous retrieval method."""
        pass
    
    @abstractmethod
    async def aget_structure(self) -> str:
        """Asynchronous structure retrieval method."""
        pass


class BaseAgentLoop(ABC):
    """
    Agent loop (AL) interacting with RAG and LLM-as-judge.
    """

    rag: BaseRAG

    @abstractmethod
    def query(self, query_str: str) -> Tuple[str, Optional[float]]:
        """
        AL.query: str -> (str, conf?)
        Takes a query string and returns a response string and an optional confidence score.
        """
        pass

    @abstractmethod
    async def aquery(self, query_str: str) -> Tuple[str, Optional[float]]:
        """Asynchronous query method."""
        pass
