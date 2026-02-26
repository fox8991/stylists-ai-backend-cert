# app/tools/style_knowledge.py
"""RAG retrieval tool for fashion knowledge base."""

from langchain_core.tools import tool
from qdrant_client.http import models

from app.rag.ingest import (
    build_retriever,
    chunk_documents,
    create_vector_store,
    load_knowledge_files,
)

_vector_store = None


def get_vector_store():
    """Get or create the vector store (singleton)."""
    global _vector_store
    if _vector_store is None:
        docs = load_knowledge_files()
        chunks = chunk_documents(docs)
        _vector_store = create_vector_store(chunks)
    return _vector_store


def search_style_knowledge_func(query: str, domain: str | None = None) -> str:
    """Search fashion knowledge base. Callable for testing.

    Args:
        query: What styling knowledge to search for.
        domain: Optional filter — one of: color_theory, body_shapes,
                style_archetypes, occasion_dressing, wardrobe_building,
                fundamentals.

    Returns:
        Retrieved styling knowledge relevant to the query.
    """
    vs = get_vector_store()
    search_kwargs: dict = {"k": 5}

    if domain:
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.domain",
                    match=models.MatchValue(value=domain),
                )
            ]
        )

    results = vs.similarity_search(query, **search_kwargs)
    if not results:
        return "No relevant fashion knowledge found."

    formatted = []
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source {i + 1}: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


@tool
def search_style_knowledge(query: str, domain: str | None = None) -> str:
    """Search the fashion knowledge base for styling advice.

    Use this tool to find information about color theory, body shapes,
    style archetypes, occasion dressing, wardrobe building, and
    clothing fundamentals.

    Args:
        query: What styling knowledge to search for (e.g., "Deep Autumn
               color palette", "inverted triangle body shape styling tips",
               "business casual dress code").
        domain: Optional filter — one of: color_theory, body_shapes,
                style_archetypes, occasion_dressing, wardrobe_building,
                fundamentals.

    Returns:
        Retrieved styling knowledge relevant to the query.
    """
    return search_style_knowledge_func(query, domain)
