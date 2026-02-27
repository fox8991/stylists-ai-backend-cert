# tests/test_tools.py
"""Tests for agent tools."""

from app.tools.style_knowledge import search_style_knowledge_func


def test_search_style_knowledge_returns_results(sample_query):
    """Should return relevant fashion knowledge for a style query."""
    result = search_style_knowledge_func(sample_query)
    assert isinstance(result, str)
    assert len(result) > 100
    assert "autumn" in result.lower()


def test_search_style_knowledge_with_domain():
    """Should filter by domain when provided."""
    result = search_style_knowledge_func(
        "what colors suit me", domain="color_theory"
    )
    assert isinstance(result, str)
    assert len(result) > 100
