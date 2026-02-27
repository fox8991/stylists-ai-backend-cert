"""Tests for the FastAPI endpoints."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health should return healthy status."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_chat_non_streaming():
    """POST /chat?stream=false should return full JSON response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            params={"stream": "false"},
            json={
                "message": "What is business casual?",
                "user_id": "test_user",
                "thread_id": "test-api-non-stream",
            },
            timeout=120.0,
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["response"]) > 50
    assert isinstance(data["tool_calls"], list)


@pytest.mark.asyncio
async def test_chat_streaming():
    """POST /chat (default) should return SSE stream with tokens."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/chat",
            json={
                "message": "What is business casual?",
                "user_id": "test_user",
                "thread_id": "test-api-stream",
            },
            timeout=120.0,
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            # Collect all SSE events
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    events.append(event)

    # Should have token events and an end event
    token_events = [e for e in events if e["type"] == "token"]
    end_events = [e for e in events if e["type"] == "end"]
    assert len(token_events) > 0, "Should have streamed at least one token"
    assert len(end_events) == 1, "Should have exactly one end event"

    # Should have tool call events (agent uses RAG for style questions)
    tool_events = [e for e in events if e["type"] in ("tool_call", "tool_result")]
    assert len(tool_events) > 0, "Should have tool call/result events"

    # Reassemble the full response from tokens
    full_response = "".join(e["content"] for e in token_events)
    assert len(full_response) > 50
