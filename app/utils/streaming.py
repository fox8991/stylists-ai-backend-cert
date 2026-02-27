"""SSE streaming and chat helper functions."""

import json

from langchain_core.messages import AIMessageChunk, HumanMessage


def build_input_state(message: str, user_id: str) -> dict:
    """Build the input state dict for graph invocation."""
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "user_profile": {},
        "observations": [],
    }


def extract_tool_calls(messages: list) -> list[dict]:
    """Extract tool call info from message history."""
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc["name"], "args": tc["args"]})
    return tool_calls


async def stream_agent_response(graph, message: str, user_id: str, thread_id: str):
    """Stream agent response as Server-Sent Events.

    Uses two stream modes simultaneously:
        - "messages": streams LLM tokens for real-time text display
        - "updates": provides complete node outputs with full tool_calls

    Yields SSE events:
        - {"type": "token", "content": "..."} for each LLM token
        - {"type": "tool_call", "name": "...", "args": {...}} when agent calls a tool
        - {"type": "tool_result", "name": "...", "content": "..."} when a tool finishes
        - {"type": "end"} when the response is complete

    Args:
        graph: Compiled LangGraph graph instance.
        message: The user's chat message.
        user_id: User identifier.
        thread_id: Conversation thread identifier.
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_state = build_input_state(message, user_id)

    async for mode, chunk in graph.astream(
        input_state, config=config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            msg_chunk, _metadata = chunk
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                event = {"type": "token", "content": msg_chunk.content}
                yield f"data: {json.dumps(event)}\n\n"

        elif mode == "updates":
            for node_name, state_update in chunk.items():
                messages = state_update.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            event = {
                                "type": "tool_call",
                                "name": tc["name"],
                                "args": tc["args"],
                            }
                            yield f"data: {json.dumps(event)}\n\n"
                    if hasattr(msg, "name") and node_name == "tools":
                        event = {
                            "type": "tool_result",
                            "name": msg.name,
                            "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                        }
                        yield f"data: {json.dumps(event)}\n\n"

    yield f"data: {json.dumps({'type': 'end'})}\n\n"
