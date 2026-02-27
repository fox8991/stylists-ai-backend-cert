"""RAG retrieval tool for fashion knowledge base."""

from langchain_core.tools import tool

from rag.registry import rag_registry


async def search_style_knowledge_func(query: str, domain: str | None = None) -> str:
    """Search fashion knowledge base. Callable for testing.

    Args:
        query: What styling knowledge to search for.
        domain: Optional post-retrieval filter — one of: color_theory,
                body_shapes, style_archetypes, occasion_dressing,
                wardrobe_building, fundamentals.

    Returns:
        Retrieved styling knowledge relevant to the query.
    """
    if rag_registry.retriever is None:
        raise RuntimeError("Retriever not initialized. Set rag_registry.retriever first.")

    docs = await rag_registry.retriever.ainvoke(query)

    if domain:
        docs = [d for d in docs if d.metadata.get("domain") == domain]

    if not docs:
        return "No relevant fashion knowledge found."

    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source {i + 1}: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


@tool
async def search_style_knowledge(query: str, domain: str | None = None) -> str:
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
    return await search_style_knowledge_func(query, domain)
