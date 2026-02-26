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
