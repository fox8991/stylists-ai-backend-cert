"""Shared test fixtures."""

import pytest

from app.tools.style_knowledge import init_style_tool
from rag.chunking import chunk_documents
from rag.loader import load_knowledge_files
from rag.vectorstore import create_vector_store

_initialized = False


@pytest.fixture(autouse=True, scope="session")
def _init_vector_store():
    """Initialize the vector store once for all tests."""
    global _initialized
    if not _initialized:
        docs = load_knowledge_files()
        chunks = chunk_documents(docs)
        vs = create_vector_store(chunks)
        init_style_tool(vs)
        _initialized = True


@pytest.fixture
def sample_query():
    return "What colors look best for a Deep Autumn color season?"
