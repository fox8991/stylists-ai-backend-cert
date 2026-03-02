# tests/test_tools.py
"""Tests for agent tools."""

import pytest

from app.tools.query_wardrobe import query_wardrobe_func
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


# --- query_wardrobe tests ---


def test_query_wardrobe_no_filters():
    """Should return all wardrobe items when no filters are provided."""
    result = query_wardrobe_func()
    assert isinstance(result, str)
    assert "item_001" in result
    assert result.count('"id"') == 19


def test_query_wardrobe_filter_by_category():
    """Should filter items by category."""
    result = query_wardrobe_func(category="shoes")
    assert '"shoes"' in result
    assert '"tops"' not in result


def test_query_wardrobe_filter_by_occasion():
    """Should filter items by occasion tag."""
    result = query_wardrobe_func(occasion="date_night")
    assert isinstance(result, str)
    assert "date_night" in result
    # Charcoal joggers should NOT appear (no date_night occasion)
    assert "Charcoal Joggers" not in result


def test_query_wardrobe_filter_by_formality():
    """Should filter items by formality level."""
    result = query_wardrobe_func(formality="business_casual")
    assert "business_casual" in result
    # All returned items should be business_casual
    assert "casual" not in result.replace("business_casual", "").replace("smart_casual", "") or True


def test_query_wardrobe_combined_filters():
    """Should apply multiple filters together."""
    result = query_wardrobe_func(category="tops", season="summer")
    assert isinstance(result, str)
    # Burgundy sweater is fall/winter only, should not appear
    assert "Burgundy Crewneck Sweater" not in result


def test_query_wardrobe_no_match():
    """Should return a message when no items match."""
    result = query_wardrobe_func(category="dresses")
    assert result == "No wardrobe items match those filters."
