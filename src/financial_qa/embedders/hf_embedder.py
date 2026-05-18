import asyncio
from financial_qa.base import BaseEmbedder
from sentence_transformers import SentenceTransformer
from typing import List


class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "deepvk/USER2-base"):
        """
        Initialise with a Hugging Face sentence-transformers model.
        """
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(self, text: str) -> List[float]:
        """
        Implementation of BaseEmbedder.embed.
        Converts the numpy array to a plain list of floats.
        """
        return self._model.encode(text).tolist()

    async def aembed(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._model.encode, text).tolist()

    async def aembed_passages(self, texts: List[str]) -> List[List[float]]:
        return await [asyncio.to_thread(self.embed, t) for t in texts]

    def update(self, model_name: str) -> None:
        """
        Replace the underlying Hugging Face model.
        """
        if model_name != self.model_name:
            self.model_name = model_name
            self._model = SentenceTransformer(model_name)
