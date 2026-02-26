# Frontend Handoff — Backend API Reference

## What the Backend Does

Stylists.ai backend is a FastAPI server running a LangGraph ReAct agent that acts as an AI personal stylist. It has:

- **RAG**: 24 curated fashion knowledge files (color theory, body shapes, style archetypes, occasion dressing, wardrobe building, fundamentals) embedded in Qdrant
- **Agent**: LangGraph ReAct agent with tool-calling loop
- **Memory** (coming soon): Learns user preferences over time via InMemoryStore
- **Tools** (currently 1, 3 more coming):
  - `search_style_knowledge` — RAG retrieval over fashion knowledge base
  - `query_wardrobe` — filter user's wardrobe items (Phase 3)
  - `generate_outfit` — LLM-powered outfit composition (Phase 3)
  - `search_trends` — Tavily web search for fashion trends (Phase 3)

## API Endpoints

### POST /chat

Send a message to the styling agent.

```json
// Request
{
  "message": "What colors look best on a Deep Autumn?",
  "user_id": "demo_user",       // optional, defaults to "demo_user"
  "thread_id": "default"         // optional, defaults to "default"
}

// Response
{
  "response": "Deep Autumn looks best in deep, warm, rich colors...",
  "tool_calls": [
    {"name": "search_style_knowledge", "args": {"query": "Deep Autumn colors", "domain": "color_theory"}}
  ],
  "observations_stored": []      // will populate once memory system is added
}
```

- `user_id`: Identifies the user for memory/personalization. Use a consistent ID per user.
- `thread_id`: Identifies the conversation thread. Same thread_id = conversation continues with context. New thread_id = fresh conversation.
- `tool_calls`: Shows which tools the agent used during reasoning (useful for UI indicators like "Searching knowledge base...")
- `observations_stored`: Facts the agent learned about the user in this interaction

### GET /health

```json
{"status": "healthy", "model": "gpt-5.2"}
```

### GET /docs

Auto-generated Swagger UI — useful for testing during development.

## Running the Backend

```bash
cd stylists-ai-backend-cert
uv run uvicorn app.main:app --reload --port 8000
```

Server runs at `http://localhost:8000`.

## Key UX Considerations

1. **Responses can take 5-15 seconds** — the agent may call tools (RAG retrieval + LLM reasoning). Show a loading state.
2. **tool_calls in response** — can be used to show what the agent is doing ("Searching style knowledge...", "Looking at your wardrobe...")
3. **thread_id for conversation continuity** — generate a UUID per chat session so multi-turn conversations work
4. **Streaming is not yet implemented** — responses come as a single JSON payload. Streaming can be added later.

## Product Context

See these docs in the backend repo for full product direction:
- `docs/PRD.md` — product requirements, architecture, schemas
- `docs/build_plan.md` — SEO → user queries → features → architecture
- `docs/tech_architecture.md` — detailed implementation specs

Key user query types the frontend should support:
- **Style education**: "What colors look good on me?", "How should I dress for my body type?"
- **Outfit generation**: "What should I wear to a job interview?" (coming Phase 3)
- **Wardrobe queries**: "Show me all my blue tops" (coming Phase 3)
- **Trend lookups**: "What's trending for spring 2026?" (coming Phase 3)
- **Profile building**: "I'm a Deep Autumn" — agent remembers this (coming Phase 3)
