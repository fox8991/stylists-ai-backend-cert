# Phase 1: RAG + Agent + Endpoint — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working RAG-powered styling agent with FastAPI endpoint — the core cert deliverable.

**Architecture:** LangGraph ReAct agent with `search_style_knowledge` tool backed by Qdrant in-memory vector store. 24 fashion markdown files chunked with RecursiveCharacterTextSplitter, embedded with text-embedding-3-small. FastAPI serves `/chat` and `/health`. Memory system (load/save nodes) wraps the agent loop. Sample wardrobe, additional tools, and eval come in later phases.

**Tech Stack:** Python 3.11+, uv, FastAPI, LangGraph, langchain, langchain-openai, langchain-qdrant, qdrant-client, Tavily (Phase 3), RAGAS (Phase 2)

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env` (user provides API keys)
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Initialize uv project**

```bash
cd /Users/yingzheli/aimakerspace/stylists-ai-backend-cert
uv init --no-readme
```

**Step 2: Add core dependencies**

```bash
uv add fastapi uvicorn langchain langchain-openai langchain-qdrant langgraph qdrant-client python-dotenv pydantic
```

**Step 3: Add dev dependencies**

```bash
uv add --dev pytest pytest-asyncio httpx
```

**Step 4: Create app/config.py**

```python
"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """App settings from environment variables."""

    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.2")
    LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "stylists-ai")


settings = Settings()
```

**Step 5: Create .env template**

```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-5.2
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=stylists-ai
```

User fills in their actual keys.

**Step 6: Create app/__init__.py, tests/__init__.py, tests/conftest.py**

`app/__init__.py` and `tests/__init__.py` are empty files.

```python
# tests/conftest.py
"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_query():
    return "What colors look best for a Deep Autumn color season?"
```

**Step 7: Verify setup**

```bash
uv run python -c "from app.config import settings; print(f'LLM: {settings.LLM_MODEL}')"
```

Expected: `LLM: gpt-5.2`

**Step 8: Commit**

```bash
git add pyproject.toml uv.lock app/ tests/ .python-version
git commit -m "feat: project scaffold with uv, config, test setup"
```

Note: Do NOT commit `.env` — add it to `.gitignore` if not already there.

---

### Task 2: RAG Ingestion Pipeline

**Files:**
- Create: `app/rag/__init__.py`
- Create: `app/rag/ingest.py`
- Create: `tests/test_rag_ingest.py`

**Step 1: Write the failing test**

```python
# tests/test_rag_ingest.py
"""Tests for RAG ingestion pipeline."""

import pytest
from app.rag.ingest import load_knowledge_files, chunk_documents, create_vector_store


def test_load_knowledge_files():
    """Should load all 24 markdown files from knowledge/ directory."""
    docs = load_knowledge_files()
    assert len(docs) == 24
    assert all(doc.metadata.get("domain") for doc in docs)


def test_chunk_documents():
    """Should split documents into chunks with metadata."""
    docs = load_knowledge_files()
    chunks = chunk_documents(docs)
    assert len(chunks) > 50  # 24 docs should produce many chunks
    # Each chunk should preserve domain metadata
    assert all(chunk.metadata.get("domain") for chunk in chunks)
    assert all(chunk.metadata.get("source") for chunk in chunks)


def test_create_vector_store():
    """Should create Qdrant vector store and return retriever."""
    docs = load_knowledge_files()
    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks)
    # Test basic retrieval
    results = vector_store.similarity_search("Deep Autumn color palette", k=3)
    assert len(results) == 3
    assert any("autumn" in r.page_content.lower() for r in results)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_rag_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag'`

**Step 3: Write the implementation**

```python
# app/rag/__init__.py
```

```python
# app/rag/ingest.py
"""Ingest fashion knowledge files: load, chunk, embed, store in Qdrant."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.config import settings

KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"

# 6 domains matching the knowledge/ subdirectories
DOMAINS = [
    "body_shapes",
    "color_theory",
    "fundamentals",
    "occasion_dressing",
    "style_archetypes",
    "wardrobe_building",
]


def load_knowledge_files() -> list[Document]:
    """Load all markdown files from knowledge/ directory with domain metadata.

    Returns:
        List of Documents with metadata: {source, domain}.
    """
    documents = []
    for domain in DOMAINS:
        domain_dir = KNOWLEDGE_DIR / domain
        if not domain_dir.exists():
            continue
        for md_file in sorted(domain_dir.glob("*.md")):
            loader = TextLoader(str(md_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["domain"] = domain
                doc.metadata["source"] = f"{domain}/{md_file.name}"
            documents.extend(docs)
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 30,
) -> list[Document]:
    """Split documents into chunks, preserving metadata.

    Args:
        documents: Raw documents from load_knowledge_files.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunked Documents with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vector_store(
    chunks: list[Document],
    collection_name: str = "fashion_knowledge",
) -> QdrantVectorStore:
    """Embed chunks and store in Qdrant in-memory vector store.

    Args:
        chunks: Chunked documents to embed and store.
        collection_name: Name for the Qdrant collection.

    Returns:
        QdrantVectorStore ready for similarity search.
    """
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        location=":memory:",
    )
    return vector_store


def build_retriever(vector_store: QdrantVectorStore, k: int = 5):
    """Create a retriever from the vector store.

    Args:
        vector_store: Qdrant vector store with embedded chunks.
        k: Number of results to retrieve.

    Returns:
        LangChain retriever.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_rag_ingest.py -v
```

Expected: 3 PASS (note: `test_create_vector_store` makes real OpenAI embedding calls — needs API key in `.env`)

**Step 5: Commit**

```bash
git add app/rag/ tests/test_rag_ingest.py
git commit -m "feat: RAG ingestion pipeline — load, chunk, embed 24 knowledge files"
```

---

### Task 3: search_style_knowledge Tool

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/style_knowledge.py`
- Create: `tests/test_tools.py`

**Step 1: Write the failing test**

```python
# tests/test_tools.py
"""Tests for agent tools."""

import pytest
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_tools.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# app/tools/__init__.py
```

```python
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

# Module-level vector store — initialized once on import
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
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_tools.py -v
```

Expected: 2 PASS

**Step 5: Commit**

```bash
git add app/tools/ tests/test_tools.py
git commit -m "feat: search_style_knowledge RAG tool with domain filtering"
```

---

### Task 4: Agent State + Prompts

**Files:**
- Create: `app/agent/__init__.py`
- Create: `app/agent/state.py`
- Create: `app/agent/prompts.py`

**Step 1: Create agent state**

```python
# app/agent/__init__.py
```

```python
# app/agent/state.py
"""Agent state definition for the LangGraph graph."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State that flows through the agent graph.

    Attributes:
        messages: Conversation message history (uses add_messages reducer).
        user_id: Unique user identifier for memory namespacing.
        user_profile: Style profile loaded from memory store.
        observations: Relevant observations loaded from memory store.
    """

    messages: Annotated[list, add_messages]
    user_id: str
    user_profile: dict
    observations: list[str]
```

```python
# app/agent/prompts.py
"""System prompts for the styling agent."""

import json

STYLIST_SYSTEM_PROMPT = """You are a personal AI stylist for Stylists.ai.

## User's Style Profile
{profile_section}

## What You Know About This User
{observations_section}

## Your Tools
- search_style_knowledge: Search the fashion knowledge base for styling advice \
(color theory, body shapes, dress codes, etc.)

## Guidelines
- Ground your advice in retrieved fashion knowledge — use search_style_knowledge \
for any styling questions
- If you learn new facts about the user (color season, body shape, preferences), \
note them — they'll be saved after this conversation
- Explain the "why" behind your suggestions
- Be warm and encouraging, like a knowledgeable friend
- If you don't know the user's profile yet, ask questions to learn about them"""


def build_system_prompt(profile: dict, observations: list[str]) -> str:
    """Build the system prompt with user context injected.

    Args:
        profile: User's style profile dict from memory store.
        observations: List of observation strings from memory store.

    Returns:
        Formatted system prompt string.
    """
    if profile:
        profile_section = json.dumps(profile, indent=2)
    else:
        profile_section = (
            "No profile yet — ask the user about their style preferences."
        )

    if observations:
        observations_section = "\n".join(f"- {obs}" for obs in observations)
    else:
        observations_section = "No observations yet — this is a new user."

    return STYLIST_SYSTEM_PROMPT.format(
        profile_section=profile_section,
        observations_section=observations_section,
    )
```

**Step 2: Verify imports work**

```bash
uv run python -c "from app.agent.state import AgentState; from app.agent.prompts import build_system_prompt; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add app/agent/
git commit -m "feat: agent state and system prompt builder"
```

---

### Task 5: LangGraph Agent Graph (without memory nodes)

**Files:**
- Create: `app/agent/graph.py`
- Create: `tests/test_agent.py`

This is the core ReAct agent. We build it first WITHOUT the memory load/save nodes — those come in Phase 3. This keeps the graph simple and testable now.

**Step 1: Write the failing test**

```python
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
    # Should mention autumn-related colors
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
    # Check that tool was called by looking at message history
    tool_messages = [
        m for m in result["messages"] if m.type == "tool"
    ]
    assert len(tool_messages) > 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_agent.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.graph'`

**Step 3: Write the implementation**

```python
# app/agent/graph.py
"""LangGraph ReAct agent graph for the AI stylist."""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.config import settings
from app.tools.style_knowledge import search_style_knowledge


def _get_tools() -> list:
    """Return the list of tools available to the agent."""
    return [search_style_knowledge]


def agent_node(state: AgentState) -> dict:
    """ReAct agent node — reasons and calls tools.

    Args:
        state: Current agent state with messages and user context.

    Returns:
        Updated messages with the agent's response.
    """
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0.7)
    tools = _get_tools()
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = build_system_prompt(
        profile=state.get("user_profile", {}),
        observations=state.get("observations", []),
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools if the agent made tool calls, otherwise end.

    Args:
        state: Current agent state.

    Returns:
        "tools" if tool calls present, "end" otherwise.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def create_graph() -> StateGraph:
    """Create and compile the agent graph.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    tools = _get_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_agent.py -v --timeout=60
```

Expected: 2 PASS (these make real LLM + embedding calls — may take 15-30s)

**Step 5: Commit**

```bash
git add app/agent/graph.py tests/test_agent.py
git commit -m "feat: LangGraph ReAct agent with search_style_knowledge tool"
```

---

### Task 6: FastAPI Endpoint

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing test**

```python
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
            timeout=60.0,
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["response"]) > 50
    assert isinstance(data["tool_calls"], list)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_api.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

**Step 3: Write the implementation**

```python
# app/main.py
"""FastAPI application for the Stylists.ai agent."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.agent.graph import create_graph
from app.config import settings

# Module-level graph instance
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent graph on startup."""
    global _graph
    _graph = create_graph()
    yield


app = FastAPI(title="Stylists.ai Agent API", lifespan=lifespan)


class ChatRequest(BaseModel):
    """Chat endpoint request body."""

    message: str
    user_id: str = "demo_user"
    thread_id: str = "default"


class ChatResponse(BaseModel):
    """Chat endpoint response body."""

    response: str
    tool_calls: list[dict] = []
    observations_stored: list[str] = []


def _extract_tool_calls(messages: list) -> list[dict]:
    """Extract tool call info from message history."""
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {"name": tc["name"], "args": tc["args"]}
                )
    return tool_calls


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the Stylist Agent.

    Args:
        request: Chat request with message, user_id, and thread_id.

    Returns:
        Agent response with tool call history.
    """
    config = {
        "configurable": {
            "thread_id": request.thread_id,
        }
    }

    result = await _graph.ainvoke(
        {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id,
            "user_profile": {},
            "observations": [],
        },
        config=config,
    )

    return ChatResponse(
        response=result["messages"][-1].content,
        tool_calls=_extract_tool_calls(result["messages"]),
    )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "model": settings.LLM_MODEL}
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_api.py -v --timeout=120
```

Expected: 2 PASS

**Step 5: Smoke test — run the server manually**

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Then in another terminal:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What colors look best on a Deep Autumn?", "user_id": "demo", "thread_id": "smoke-1"}'
```

Expected: JSON response with styling advice grounded in the knowledge base.

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "healthy", "model": "gpt-5.2"}`

**Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: FastAPI /chat and /health endpoints"
```

---

### Task 7: Add .gitignore + Final Cleanup

**Files:**
- Create or update: `.gitignore`

**Step 1: Ensure .gitignore covers essentials**

```
# Environment
.env
.env.*

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# IDE
.vscode/
.idea/
.cursor/

# OS
.DS_Store

# Notebooks
.ipynb_checkpoints/

# uv
.venv/
```

**Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --timeout=120
```

Expected: All tests pass (5 tests total: 2 ingest, 2 tools, 1 health, 2 agent, 2 api — some overlap, adjust as needed).

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

---

## Phase 1 Complete Checklist

After all 7 tasks:

- [x] `uv` project with all dependencies
- [x] 24 knowledge files loaded, chunked, embedded in Qdrant in-memory
- [x] `search_style_knowledge` tool with domain filtering
- [x] LangGraph ReAct agent that reasons and calls the tool
- [x] FastAPI `/chat` endpoint serving the agent
- [x] FastAPI `/health` endpoint
- [x] Tests passing for ingest, tools, agent, and API
- [x] Clean git history with working increments

**Next:** Phase 2 — RAGAS baseline eval in Jupyter notebook.
