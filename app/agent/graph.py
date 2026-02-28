# app/agent/graph.py
"""LangGraph ReAct agent graph for the AI stylist."""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from config import settings
from app.tools.search_trends import search_trends
from app.tools.style_knowledge import search_style_knowledge


def _get_tools() -> list:
    """Return the list of tools available to the agent."""
    return [search_style_knowledge, search_trends]


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


def create_graph() -> CompiledStateGraph:
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
