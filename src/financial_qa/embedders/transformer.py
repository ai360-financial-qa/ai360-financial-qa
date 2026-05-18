import asyncio
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from financial_qa.base import BaseEmbedder


class TransformerEmbedder(BaseEmbedder):
    """
    Dense embedder using a HuggingFace transformer with mean pooling + L2 norm.

    Works well with multilingual-e5 variants. The query_prefix / passage_prefix
    parameters let you enable instruction-following behaviour (e5 expects
    "query: ..." for queries and "passage: ..." for indexed text).
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        pooled = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=self.max_length, padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            emb = self._mean_pool(out.last_hidden_state, inputs["attention_mask"])
            results.extend(emb.cpu().tolist())
        return results

    def embed(self, text: str) -> List[float]:
        """Embed a query string."""
        return self._encode([self.query_prefix + text])[0]

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of passage strings (used during precalc)."""
        return self._encode([self.passage_prefix + t for t in texts])

    async def aembed(self, text: str) -> List[float]:
        """Asynchronously embed a query string."""
        return await asyncio.to_thread(self.embed, text)

    async def aembed_passages(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a batch of passage strings."""
        return await asyncio.to_thread(self.embed_passages, texts)
