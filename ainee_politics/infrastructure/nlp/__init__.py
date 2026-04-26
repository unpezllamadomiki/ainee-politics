"""NLP infrastructure: spaCy enrichment, ML classifiers and RAG helpers."""

from .rag import answer_question, build_vector_store

__all__ = ["answer_question", "build_vector_store"]
