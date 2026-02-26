# tests/test_api.py
"""Tests for the FastAPI endpoints."""

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
async def test_chat_endpoint():
    """POST /chat should return agent response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "message": "What is business casual?",
                "user_id": "test_user",
                "thread_id": "test-api-1",
            },
            timeout=120.0,
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["response"]) > 50
    assert isinstance(data["tool_calls"], list)
