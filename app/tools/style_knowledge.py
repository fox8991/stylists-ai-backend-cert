"""RAG retrieval tool for fashion knowledge base."""

from langchain_core.tools import tool
from qdrant_client.http import models

_vector_store = None


def init_style_tool(vector_store) -> None:
    """Initialize the tool with a vector store instance.

    Called once at app startup from lifespan.

    Args:
        vector_store: A QdrantVectorStore to query.
    """
    global _vector_store
    _vector_store = vector_store


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
    if _vector_store is None:
        raise RuntimeError("Style tool not initialized. Call init_style_tool() first.")

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

    results = _vector_store.similarity_search(query, **search_kwargs)
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
