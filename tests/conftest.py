"""Shared test fixtures."""

import pytest

from rag.registry import rag_registry
from rag.retrieval import create_naive_retriever
from rag.vectorstore import get_vector_store

_initialized = False


@pytest.fixture(autouse=True, scope="session")
def _init_vector_store():
    """Initialize the retriever and style tool once for all tests."""
    global _initialized
    if not _initialized:
        rag_registry.vector_store = get_vector_store()
        rag_registry.retriever = create_naive_retriever(rag_registry.vector_store, k=10)
        _initialized = True


@pytest.fixture
def sample_query():
    return "What colors look best for a Deep Autumn color season?"
