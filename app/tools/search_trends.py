"""Fashion trend search tool using Tavily web search API."""

from tavily import AsyncTavilyClient
from langchain_core.tools import tool

from config import settings


def _get_tavily_client() -> AsyncTavilyClient:
    """Get an async Tavily client.

    Returns:
        Configured AsyncTavilyClient.

    Raises:
        RuntimeError: If TAVILY_API_KEY is not set.
    """
    if not settings.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY not set in .env")
    return AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)


async def search_trends_func(query: str, time_range: str = "month") -> str:
    """Search the web for current fashion trends. Callable for testing.

    Args:
        query: Fashion trend to search for (e.g., "spring 2026 color trends",
               "streetwear trends").
        time_range: How recent the results should be — one of:
                    "day", "week", "month", "year". Defaults to "month".

    Returns:
        Formatted search results with titles, URLs, and content snippets.
    """
    client = _get_tavily_client()

    response = await client.search(
        query=f"fashion {query}",
        topic="general",
        search_depth="basic",
        max_results=5,
        time_range=time_range,
        include_answer=True,
    )

    parts: list[str] = []

    answer = response.get("answer")
    if answer:
        parts.append(f"Summary: {answer}")

    for i, result in enumerate(response.get("results", []), 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        parts.append(f"[{i}] {title}\n    {url}\n    {content[:300]}")

    return "\n\n".join(parts) if parts else "No trend results found."


@tool
async def search_trends(query: str, time_range: str = "month") -> str:
    """Search the web for current fashion and style trends.

    Use this tool when the user asks about current trends, what's popular
    right now, seasonal fashion, or anything that requires up-to-date
    information beyond the knowledge base.

    Args:
        query: Fashion trend to search for (e.g., "spring 2026 color trends",
               "streetwear trends", "business casual trends").
        time_range: How recent — "day", "week", "month" (default), or "year".

    Returns:
        Current fashion trend information from the web.
    """
    return await search_trends_func(query, time_range)
