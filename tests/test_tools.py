# tests/test_tools.py
"""Tests for agent tools."""

import pytest

from app.tools.search_trends import search_trends_func
from app.tools.style_knowledge import search_style_knowledge_func


@pytest.mark.asyncio
async def test_search_style_knowledge_returns_results(sample_query):
    """Should return relevant fashion knowledge for a style query."""
    result = await search_style_knowledge_func(sample_query)
    assert isinstance(result, str)
    assert len(result) > 100
    assert "autumn" in result.lower()


@pytest.mark.asyncio
async def test_search_style_knowledge_with_domain():
    """Should filter by domain when provided."""
    result = await search_style_knowledge_func(
        "what colors suit me", domain="color_theory"
    )
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.asyncio
async def test_search_trends_returns_results():
    """Should return current fashion trend results from the web."""
    result = await search_trends_func("spring 2026 color trends")
    assert isinstance(result, str)
    assert len(result) > 50
    assert result != "No trend results found."


@pytest.mark.asyncio
async def test_search_trends_with_time_range():
    """Should respect time_range parameter."""
    result = await search_trends_func("streetwear trends", time_range="week")
    assert isinstance(result, str)
    assert len(result) > 50
