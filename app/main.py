"""FastAPI application for the Stylists.ai agent."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

from app.agent.graph import create_graph
from app.config import settings

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
                tool_calls.append({"name": tc["name"], "args": tc["args"]})
    return tool_calls


def _get_graph():
    """Get the graph, lazily initializing if needed."""
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph


def _build_input_state(request: ChatRequest) -> dict:
    """Build the input state dict for graph invocation."""
    return {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "user_profile": {},
        "observations": [],
    }


async def _stream_agent_response(request: ChatRequest):
    """Stream agent response as Server-Sent Events.

    Uses two stream modes simultaneously:
        - "messages": streams LLM tokens for real-time text display
        - "updates": provides complete node outputs with full tool_calls

    Yields SSE events:
        - {"type": "token", "content": "..."} for each LLM token
        - {"type": "tool_call", "name": "...", "args": {...}} when agent calls a tool
        - {"type": "tool_result", "name": "..."} when a tool finishes
        - {"type": "end"} when the response is complete

    Args:
        request: Chat request with message, user_id, and thread_id.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": request.thread_id}}
    input_state = _build_input_state(request)

    async for mode, chunk in graph.astream(
        input_state, config=config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            msg_chunk, metadata = chunk
            # Only stream text tokens from AIMessageChunk (the LLM output)
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                event = {"type": "token", "content": msg_chunk.content}
                yield f"data: {json.dumps(event)}\n\n"

        elif mode == "updates":
            # "updates" gives complete node outputs: {"node_name": state_update}
            for node_name, state_update in chunk.items():
                messages = state_update.get("messages", [])
                for msg in messages:
                    # Extract complete tool_calls from agent node output
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            event = {
                                "type": "tool_call",
                                "name": tc["name"],
                                "args": tc["args"],
                            }
                            yield f"data: {json.dumps(event)}\n\n"
                    # Extract tool results from tools node output
                    if hasattr(msg, "name") and node_name == "tools":
                        event = {
                            "type": "tool_result",
                            "name": msg.name,
                            "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                        }
                        yield f"data: {json.dumps(event)}\n\n"

    yield f"data: {json.dumps({'type': 'end'})}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest, stream: bool = Query(default=True)):
    """Send a message to the Stylist Agent.

    Args:
        request: Chat request with message, user_id, and thread_id.
        stream: If True (default), return SSE stream. If False, return JSON.

    Returns:
        SSE stream or JSON response depending on stream parameter.
    """
    if stream:
        return StreamingResponse(
            _stream_agent_response(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: invoke and return full JSON response
    graph = _get_graph()
    config = {"configurable": {"thread_id": request.thread_id}}

    result = await graph.ainvoke(
        _build_input_state(request),
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
