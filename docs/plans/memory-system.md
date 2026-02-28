# Memory System — Implementation Plan

## Overview

LangGraph separates memory into two layers:

1. **Short-term memory (Checkpointer)** — conversation history within a single thread
2. **Long-term memory (Store)** — user data that persists across all conversations

Both use the same Postgres DB in production (e.g., Supabase).

## Short-term Memory: Checkpointer

Stores message history, tool calls, and agent state for a conversation thread.

| Environment | Implementation | Package |
|---|---|---|
| Dev/Test | `MemorySaver` (in-memory) | `langgraph` (built-in) |
| Production | `AsyncPostgresSaver` | `langgraph-checkpoint-postgres` |

Keyed by `thread_id` in config:
```python
config = {"configurable": {"thread_id": "user-abc-session-1"}}
```

**Current state:** Using `MemorySaver`. Swap to `AsyncPostgresSaver` for prod.

## Long-term Memory: Store

Stores user profile, preferences, and observations that persist across threads.
Supports semantic search (find relevant memories by query).

| Environment | Implementation | Package |
|---|---|---|
| Dev/Test | `InMemoryStore` (with embeddings for semantic search) | `langgraph` (built-in) |
| Production | `AsyncPostgresStore` | `langgraph-store-postgres` |

Keyed by namespace tuples:
```python
# Store a memory
store.put(("memories", user_id), "memory-uuid", {"data": "User is Deep Autumn"})

# Semantic search
memories = store.search(("memories", user_id), query="color preferences", limit=5)
```

**Current state:** Not implemented yet.

## Multi-user: Runtime Context

LangGraph's `Runtime` pattern passes `user_id` to graph nodes without polluting the state:

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

# Graph definition
builder = StateGraph(MessagesState, context_schema=Context)

# At invocation time
graph.astream(
    {"messages": [...]},
    config={"configurable": {"thread_id": "thread-1"}},
    context=Context(user_id="user-abc"),
)

# Inside a node
async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    memories = await runtime.store.asearch(("memories", user_id), query=...)
```

## Production Setup (Supabase Postgres)

```python
DB_URI = "postgresql://..."

async with (
    AsyncPostgresStore.from_conn_string(DB_URI) as store,
    AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    graph = builder.compile(checkpointer=checkpointer, store=store)
```

Both short-term and long-term memory use the same DB connection.

## User Profile: Cert vs Production

### How user_profile flows through the system

The `AgentState` has `user_profile: dict` and `observations: list[str]`. The `agent_node` reads these from state and injects them into the system prompt via `build_system_prompt()`. The agent then has the user's context for personalized responses.

### Cert Challenge (current)

No memory nodes, no store. Hardcode a demo profile in `build_input_state()`:

```python
# app/utils/streaming.py
DEMO_PROFILE = {
    "body_shape": "inverted_triangle",
    "color_season": "deep_autumn",
    "style_archetype": "classic_natural",
    "preferences": {
        "loves": ["earth tones", "structured pieces"],
        "avoids": ["bright neons", "heavy patterns"],
    },
}

def build_input_state(message: str, user_id: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "user_profile": DEMO_PROFILE,
        "observations": [],
    }
```

Graph flow: `START → agent ⇄ tools → END`

Profile is static — every request gets the same demo profile. No `load_memory` or `save_memory` nodes.

### Production

Add `load_memory` and `save_memory` nodes to the graph. Profile comes from the Store, not hardcoded.

Graph flow: `START → load_memory → agent ⇄ tools → save_memory → END`

```python
def load_memory_node(state, *, store):
    """Reads profile + observations from store. Only updates state fields, no messages."""
    user_id = state["user_id"]
    profile_items = store.search(("users", user_id, "profile"))
    profile = profile_items[0].value if profile_items else {}
    observations = store.search(
        ("users", user_id, "observations"),
        query=state["messages"][-1].content,
        limit=5,
    )
    return {
        "user_profile": profile,
        "observations": [o.value["content"] for o in observations],
    }
```

Key difference: `load_memory` only updates `user_profile` and `observations` in state — it does NOT create any messages. The agent node is the first thing that touches messages.

## Implementation Steps

1. **Cert (now):** Hardcode `DEMO_PROFILE` in `build_input_state()`, skip memory nodes
2. **Post-cert:** Add `InMemoryStore` + `load_memory`/`save_memory` nodes
3. **Production:** Swap to persistent storage:
   - Add `langgraph-checkpoint-postgres` and `langgraph-store-postgres` to dependencies
   - Add `context_schema=Context` to graph builder in `app/agent/graph.py`
   - Swap `MemorySaver` → `AsyncPostgresSaver`, add `AsyncPostgresStore`
   - Update `app/main.py` to pass `context=Context(user_id=...)` at invocation
   - Add `SUPABASE_DB_URI` to config

## References

- [LangGraph Memory Docs](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- PRD memory section: `docs/PRD.md` (namespace structure, schemas)
