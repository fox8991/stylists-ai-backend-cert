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

Send a message to the styling agent. **Streams by default** via Server-Sent Events (SSE).

**Request body:**
```json
{
  "message": "What colors look best on a Deep Autumn?",
  "user_id": "demo_user",       // optional, defaults to "demo_user"
  "thread_id": "default"         // optional, defaults to "default"
}
```

**Query params:**
- `stream` (bool, default `true`): Set to `false` for a full JSON response instead of SSE.

#### Streaming mode (default): `POST /chat`

Returns `text/event-stream` with SSE events:

```
data: {"type": "token", "content": "Deep"}
data: {"type": "token", "content": " Autumn"}
data: {"type": "token", "content": " looks"}
data: {"type": "token", "content": " best"}
...
data: {"type": "tool_call", "name": "search_style_knowledge", "args": "..."}
...
data: {"type": "end"}
```

Event types:
- `token` — a chunk of the agent's response text. Append `content` to build the full message.
- `tool_call` — the agent is calling a tool. Use `name` to show UI indicators (e.g., "Searching knowledge base...").
- `end` — the response is complete.

**Frontend usage (JavaScript):**
```javascript
const response = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "What colors suit me?", user_id: "user1", thread_id: "thread1" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const event = JSON.parse(line.slice(6));
      if (event.type === "token") {
        // Append to chat message
      } else if (event.type === "tool_call") {
        // Show "Searching..." indicator
      } else if (event.type === "end") {
        // Done
      }
    }
  }
}
```

#### Non-streaming mode: `POST /chat?stream=false`

Returns full JSON response:
```json
{
  "response": "Deep Autumn looks best in deep, warm, rich colors...",
  "tool_calls": [
    {"name": "search_style_knowledge", "args": {"query": "Deep Autumn colors", "domain": "color_theory"}}
  ],
  "observations_stored": []
}
```

#### Field reference

- `user_id`: Identifies the user for memory/personalization. Use a consistent ID per user.
- `thread_id`: Identifies the conversation thread. Same thread_id = conversation continues with context. New thread_id = fresh conversation.
- `tool_calls`: Shows which tools the agent used during reasoning.
- `observations_stored`: Facts the agent learned about the user in this interaction (populates once memory system is added).

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

1. **Streaming is the default** — tokens arrive incrementally via SSE. Render them as they come for a responsive chat experience.
2. **tool_call events** — show what the agent is doing ("Searching style knowledge...", "Looking at your wardrobe...") while the user waits for content tokens.
3. **thread_id for conversation continuity** — generate a UUID per chat session so multi-turn conversations work.
4. **First token may take a few seconds** — the agent does RAG retrieval before generating. Show a typing indicator until the first token arrives.

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
