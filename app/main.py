"""FastAPI application for the Stylists.ai agent."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.graph import create_graph
from app.registry import app_registry
from app.utils.streaming import build_input_state, extract_tool_calls, stream_agent_response
from config import settings
from rag.registry import rag_registry
from rag.retrieval import create_naive_retriever
from rag.vectorstore import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the vector store and agent graph on startup."""
    rag_registry.vector_store = get_vector_store()
    rag_registry.retriever = create_naive_retriever(rag_registry.vector_store, k=10)
    app_registry.graph = create_graph()
    yield


app = FastAPI(title="Stylists.ai Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://stylists-ai-frontend-cert.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    if app_registry.graph is None:
        app_registry.graph = create_graph()
    return app_registry.graph


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
