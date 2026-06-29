"""
Embedding index over the book chunks.

All chapters are embedded once into a single FAISS flat-IP (cosine) index. The
bounded-vs-unbounded distinction is NOT two indices — it is a query-time
metadata filter on chapter_index (see retrieval.py). This mirrors how
production vector stores (Pinecone, Weaviate, Bedrock Knowledge Bases) apply
pre-retrieval filters, and keeps the anti-spoiler mechanism as a single
load-bearing line rather than a data-duplication scheme.

Heavy imports (faiss, sentence-transformers/torch) are deferred to build time
so the rest of the package stays cheap to import.
"""

from __future__ import annotations

import os

from . import config
from .book import Chunk


class EmbeddingIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        embed_model: str,
        hf_token: str | None = None,
        title: str | None = None,
        author: str | None = None,
    ):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

        self.chunks = chunks
        # Book identity travels with the index so the response prompts can name the
        # right book (the index is the per-book corpus). Optional: callers that don't
        # set it (eval, notebook) fall back to the config default in respond.py.
        self.title = title
        self.author = author
        self.embedder = SentenceTransformer(embed_model)

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)  # cosine similarity via inner product

        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self._np = np
        self._faiss = faiss

    def embed_query(self, query: str):
        vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        self._faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, k: int) -> list[int]:
        """Return chunk indices for the top-k nearest neighbours."""
        k = min(k, len(self.chunks))
        _, indices = self.index.search(self.embed_query(query), k)
        return [i for i in indices[0] if 0 <= i < len(self.chunks)]


def build_index(
    chunks: list[Chunk],
    embed_model: str = config.EMBED_MODEL,
    hf_token: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> EmbeddingIndex:
    if hf_token is None:
        hf_token = config.get_hf_token()
    return EmbeddingIndex(chunks, embed_model, hf_token, title=title, author=author)
