# Stylists.ai Backend — Product Requirements Document

## Overview

Stylists.ai is an AI-powered personal styling platform. This backend serves a LangGraph ReAct agent that acts as an AI personal stylist, with RAG-grounded fashion knowledge, wardrobe intelligence, and long-term memory.

This is for the **AIE9 certification challenge**. The cert has specific requirements listed below.

## Cert Requirements Checklist

| Requirement | How We Satisfy It |
|---|---|
| **RAG** | Fashion knowledge base (24 markdown files) → chunked by H2 headers → embedded with text-embedding-3-small → retrieved via `search_style_knowledge` tool |
| **Agent** | LangGraph ReAct agent with 4 tools + memory read/write layer |
| **RAGAS evaluation** | Evaluate style education queries: faithfulness, context precision, context recall |
| **External API** | Tavily for trend searches (`search_trends` tool) |
| **Own data uploaded** | Curated fashion knowledge corpus in `knowledge/` folder |
| **Local endpoint** | FastAPI `/chat` endpoint serving the agent |
| **5-min Loom** | Demo showing agent learning user's style profile over multiple interactions |

## Architecture

```
User message
    │
    ▼
┌─────────────────────────────┐
│  MEMORY READ                │
│  Load user profile +        │
│  relevant observations      │
│  from InMemoryStore         │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  ReAct AGENT (LangGraph)    │
│                             │
│  System prompt includes:    │
│  - User's style profile     │
│  - Recent observations      │
│  - Fashion stylist persona  │
│                             │
│  Tools:                     │
│  ├── search_style_knowledge │ ← RAG retrieval over fashion corpus
│  ├── query_wardrobe         │ ← Filter/retrieve wardrobe items
│  ├── generate_outfit        │ ← Wardrobe + context → outfit combo
│  └── search_trends          │ ← Tavily API (external API req)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  MEMORY WRITE               │
│  Extract new observations:  │
│  - Profile facts learned    │
│  - Preferences expressed    │
│  - Feedback on suggestions  │
│  Store to InMemoryStore     │
└─────────────────────────────┘
    │
    ▼
Response to user
```

### LangGraph Graph Structure

```
START → load_memory → agent (ReAct loop) ⇄ tools → save_memory → END
```

- `load_memory`: Read user profile + observations from InMemoryStore, inject into system prompt
- `agent`: ReAct loop — reason → tool call → observe → repeat until done
- `tools`: ToolNode with the 4 tools
- `save_memory`: LLM extracts new observations from conversation, stores to InMemoryStore

## Tech Stack

| Component | Technology |
|---|---|
| LLM | GPT-4o (or Claude 3.5 Sonnet) |
| Agent framework | LangGraph (StateGraph with ReAct pattern) |
| Embedding | OpenAI text-embedding-3-small (1536 dims) |
| Vector store | Qdrant `:memory:` (in-memory for cert) |
| Memory store | LangGraph InMemoryStore |
| Checkpointer | MemorySaver |
| External API | Tavily (web search for trends) |
| Evaluation | RAGAS (faithfulness, context precision, context recall) |
| Monitoring | LangSmith |
| Endpoint | FastAPI |

## Knowledge Base (RAG Corpus)

The `knowledge/` folder contains 24 curated markdown files across 6 styling domains, sourced from deep research reports (Claude, OpenAI, Gemini) plus web-search summaries (Cowork):

```
knowledge/
├── color_theory/         (4 reports — seasonal color analysis, undertones, palettes)
├── body_shapes/          (4 reports — proportion, balance, silhouette, inclusive styling)
├── style_archetypes/     (4 reports — classic, romantic, dramatic, natural, creative, modern aesthetics)
├── occasion_dressing/    (4 reports — dress codes, business casual, formal, casual)
├── wardrobe_building/    (4 reports — capsule wardrobes, essentials, mix-and-match)
└── fundamentals/         (4 reports — fabric, fit, proportion, color matching)
```

### Chunking Strategy
- Split by H2 headers (natural topic boundaries)
- Target ~300-500 tokens per chunk
- Overlap: 50 tokens between chunks
- Metadata per chunk: `{domain, subtopic, specificity, source_provider}`

### Retrieval Strategy
- **Baseline** (for RAGAS comparison): naive dense retrieval, k=5
- **Improved**: metadata filter by domain + retrieve k=15 → Cohere rerank to top-5
- Expected improvement: 10-15% on context precision

## Agent Tools

### search_style_knowledge (RAG)
- Args: `query: str`, `domain: str = None`
- Searches the fashion vector store
- Optional domain filter: color_theory, body_shapes, style_archetypes, occasion_dressing, wardrobe_building, fundamentals
- Returns top-5 relevant chunks with source metadata

### query_wardrobe
- Args: `category`, `color`, `occasion`, `season`, `formality` (all optional), `limit: int = 20`
- Filters in-memory wardrobe item list
- For cert: uses preloaded sample wardrobe (15-20 items)

### generate_outfit
- Args: `occasion: str`, `weather: str = None`, `mood: str = None`, `constraints: str = None`
- Fetches full wardrobe → LLM selects outfit combination
- Returns: selected items with reasoning and styling tips

### search_trends (External API — Tavily)
- Args: `query: str`
- Searches current fashion trends via Tavily web search
- Appends "fashion trends 2026" to query for relevance

## Memory System

### Namespace Structure
```
("users", user_id, "profile")          → style profile JSON (single doc, updated in place)
("users", user_id, "observations")     → learned preferences (collection, accumulates)
```

### Style Profile Schema
```json
{
    "body_shape": "inverted_triangle",
    "color_season": "deep_autumn",
    "style_archetype": "classic_natural",
    "preferences": {
        "loves": ["earth tones", "structured pieces"],
        "avoids": ["bright neons", "heavy patterns"]
    }
}
```

### Wardrobe Item Schema
```json
{
    "id": "item_001",
    "name": "Navy Wool Blazer",
    "category": "outerwear",
    "sub_category": "blazer",
    "color": {"primary": "navy", "hex": "#1b3a5c"},
    "fabric": "wool blend",
    "pattern": "solid",
    "fit": "structured",
    "formality": "business_casual",
    "season": ["fall", "winter", "spring"],
    "occasions": ["business", "smart_casual", "date_night"],
    "style_tags": ["classic", "versatile"]
}
```

### Observation Schema
```json
{
    "content": "User consistently prefers structured blazers over cardigans for formal events.",
    "created_at": "2026-02-18T14:30:00Z",
    "context": "outfit_suggestion_feedback",
    "confidence": "high"
}
```

### Memory Flow
1. `load_memory` reads profile + semantically searches observations relevant to current query
2. Both injected into agent system prompt
3. After agent responds, `save_memory` uses LLM to extract new observations
4. Profile facts (body_shape, color_season) update profile doc; preferences become observation entries

## Sample Wardrobe (Cert)

Preload 15-20 items designed around a Deep Autumn color palette. Include:
- In-palette items: rust, olive, cream, burgundy, forest green, chocolate brown
- Out-of-palette items: bright red, neon, icy blue (so agent can demonstrate color awareness)
- Mix of categories: tops, bottoms, dresses, outerwear, shoes, accessories
- Mix of formality: casual through formal

## RAGAS Evaluation

Test set: 15-20 style education questions with ground truth answers.
Metrics: faithfulness (>0.8), context precision (>0.7), context recall (>0.7).
Compare baseline retriever vs improved (metadata filter + rerank).

## API Contract

```
POST /chat
Request:  { "message": str, "user_id": str, "thread_id": str }
Response: { "response": str, "tool_calls": list[dict], "observations_stored": list[str] }

GET /health
Response: { "status": "healthy", "model": "gpt-4o" }
```

## Build Sequence (Cert — 3 Weeks)

| Week | Focus |
|---|---|
| 1 | RAG pipeline (chunk, embed, retrieve) + basic ReAct agent with `search_style_knowledge` + FastAPI endpoint |
| 2 | Wardrobe tools (`query_wardrobe`, `generate_outfit`) + Tavily `search_trends` + sample wardrobe data |
| 3 | Memory system (load/save nodes) + RAGAS evaluation + polish + Loom recording |

## Deployment

- **Cert**: Local FastAPI server
- **Demo Day**: Railway (backend) + Qdrant Cloud (vectors) + Supabase (memory/auth)

## Key Design Decisions

- **Single ReAct agent for cert** (not multi-agent) — simpler to build, debug, eval. Memory layer makes it non-trivial.
- **RAG over fine-tuning** — fashion knowledge changes; RAG is updatable
- **InMemoryStore for cert** — no database needed for single user. Swap to PostgresStore for demo day.
- **Qdrant for RAG vectors** — HNSW + payload filtering, good for fashion domain metadata
- **Semantic memory only for cert** — demonstrates learning without the complexity of episodic/procedural

## Demo Day Evolution

For demo day, the cert architecture evolves into:
- Supervisor + 4 specialist agents (Outfit Specialist, Style Advisor, Wardrobe Analyst, Profile Builder)
- Three memory types (semantic + episodic + procedural)
- Persistent storage (PostgresStore + pgvector)
- See `docs/tech_architecture.md` for full demo day specs
