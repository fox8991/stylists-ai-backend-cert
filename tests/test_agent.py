# tests/test_agent.py
"""Tests for the LangGraph agent."""

import pytest
from langchain_core.messages import HumanMessage

from app.agent.graph import create_graph


@pytest.mark.asyncio
async def test_agent_responds_to_style_question():
    """Agent should answer a basic style question using RAG."""
    graph = create_graph()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What colors suit a Deep Autumn?")],
            "user_id": "test_user",
            "user_profile": {},
            "observations": [],
        },
        config={"configurable": {"thread_id": "test-1"}},
    )
    response = result["messages"][-1].content
    assert len(response) > 50
    assert any(
        word in response.lower()
        for word in ["autumn", "warm", "earth", "rust", "olive", "burgundy"]
    )


@pytest.mark.asyncio
async def test_agent_uses_search_tool():
    """Agent should call search_style_knowledge for style questions."""
    graph = create_graph()
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="What's the difference between business casual and smart casual?"
                )
            ],
            "user_id": "test_user",
            "user_profile": {},
            "observations": [],
        },
        config={"configurable": {"thread_id": "test-2"}},
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) > 0
