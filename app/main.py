"""FastAPI application for the Stylists.ai agent."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.graph import create_graph
from app.tools.style_knowledge import init_style_tool
from app.utils.streaming import build_input_state, extract_tool_calls, stream_agent_response
from config import settings
from rag.chunking import chunk_documents
from rag.loader import load_knowledge_files
from rag.vectorstore import create_vector_store

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the vector store and agent graph on startup."""
    global _graph
    docs = load_knowledge_files()
    chunks = chunk_documents(docs)
    vs = create_vector_store(chunks)
    init_style_tool(vs)
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


def _get_graph():
    """Get the graph, lazily initializing if needed."""
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph


@app.post("/chat")
async def chat(request: ChatRequest, stream: bool = Query(default=True)):
    """Send a message to the Stylist Agent.

    Args:
        request: Chat request with message, user_id, and thread_id.
        stream: If True (default), return SSE stream. If False, return JSON.

    Returns:
        SSE stream or JSON response depending on stream parameter.
    """
    graph = _get_graph()

    if stream:
        return StreamingResponse(
            stream_agent_response(graph, request.message, request.user_id, request.thread_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    config = {"configurable": {"thread_id": request.thread_id}}
    result = await graph.ainvoke(
        build_input_state(request.message, request.user_id),
        config=config,
    )

    return ChatResponse(
        response=result["messages"][-1].content,
        tool_calls=extract_tool_calls(result["messages"]),
    )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "model": settings.LLM_MODEL}
