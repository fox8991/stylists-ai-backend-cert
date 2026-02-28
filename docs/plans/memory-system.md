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

## Implementation Steps

1. Add `langgraph-checkpoint-postgres` and `langgraph-store-postgres` to dependencies
2. Add `context_schema=Context` to graph builder in `app/agent/graph.py`
3. Add `load_memory` node — reads user profile + observations from store, injects into system prompt
4. Add `save_memory` node — LLM extracts new facts from conversation, stores to memory
5. Update `app/main.py` to pass `context=Context(user_id=...)` at invocation
6. For prod: swap `MemorySaver` → `AsyncPostgresSaver`, add `AsyncPostgresStore`
7. Add `SUPABASE_DB_URI` to config

## References

- [LangGraph Memory Docs](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- PRD memory section: `docs/PRD.md` (namespace structure, schemas)
