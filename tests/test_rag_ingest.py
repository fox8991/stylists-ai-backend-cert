# tests/test_rag_ingest.py
"""Tests for RAG ingestion pipeline."""

from rag.loader import load_knowledge_files
from rag.chunking import chunk_documents
from rag.vectorstore import create_vector_store


def test_load_knowledge_files():
    """Should load all 24 markdown files from knowledge/ directory."""
    docs = load_knowledge_files()
    assert len(docs) == 24
    assert all(doc.metadata.get("domain") for doc in docs)


def test_chunk_documents():
    """Should split documents into chunks with metadata."""
    docs = load_knowledge_files()
    chunks = chunk_documents(docs)
    assert len(chunks) > 50
    assert all(chunk.metadata.get("domain") for chunk in chunks)
    assert all(chunk.metadata.get("source") for chunk in chunks)


def test_create_vector_store():
    """Should create Qdrant vector store and return retriever."""
    docs = load_knowledge_files()
    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks)
    results = vector_store.similarity_search("Deep Autumn color palette", k=3)
    assert len(results) == 3
    assert any("autumn" in r.page_content.lower() for r in results)
