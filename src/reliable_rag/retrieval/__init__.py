"""Dense retrieval adapters."""

from reliable_rag.retrieval.embedder import (
    Embedder,
    SentenceTransformerEmbedder,
    load_embedding_cache,
    save_embedding_cache,
)
from reliable_rag.retrieval.retriever import DenseRetriever

__all__ = [
    "DenseRetriever",
    "Embedder",
    "SentenceTransformerEmbedder",
    "load_embedding_cache",
    "save_embedding_cache",
]
