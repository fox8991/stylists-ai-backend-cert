# Stylists.ai Cert Build — Design Document

**Date:** 2026-02-26
**Timeline:** 3 days
**Goal:** Hit all AIE9 cert requirements: RAG, Agent, RAGAS eval, external API, own data, local endpoint, Loom demo

## Build Sequence

### Phase 1 — RAG + Agent + Endpoint
1. Project scaffold (uv, dependencies, project structure)
2. RAG pipeline: chunk 24 knowledge files with RecursiveCharacterTextSplitter → embed with text-embedding-3-small → Qdrant in-memory
3. `search_style_knowledge` tool (RAG retrieval)
4. LangGraph ReAct agent with this one tool
5. FastAPI `/chat` + `/health` endpoints
6. Smoke test end-to-end

### Phase 2 — RAGAS Baseline Eval (Jupyter)
- Generate synthetic test set via RAGAS TestsetGenerator (or curate 15-20 Q&A pairs)
- Run RAGAS: faithfulness, context recall, factual correctness, response relevancy, context entity recall, noise sensitivity
- Pure retrieval quality — no wardrobe/persona needed yet

### Phase 3 — Tools + Memory + Wardrobe
- `query_wardrobe` tool + sample wardrobe data (15-20 items, Deep Autumn palette)
- `generate_outfit` tool (LLM-powered outfit composition)
- `search_trends` tool (Tavily)
- Memory system: `load_memory` + `save_memory` graph nodes with InMemoryStore
- Register all tools in graph, add basic tests

### Phase 4 — Advanced Retriever + Re-eval (Jupyter)
- Improved retriever: Cohere rerank (rerank-v3.5) via ContextualCompressionRetriever
- Optionally: metadata domain filter, multi-query, ensemble
- Re-run RAGAS, compare baseline vs improved in table
- Agent-level eval: tool call accuracy, goal accuracy

### Phase 5 — Deploy + Persistence (post-cert)
- Deploy to Railway (still in-memory)
- Swap to Qdrant Cloud + Supabase PostgresStore for persistence

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| LLM | GPT-5.2 (agent) | Current OpenAI flagship |
| Eval LLM | GPT-4.1-mini | Cost-effective for judging |
| Embeddings | text-embedding-3-small (1536 dims) | Tunable hyperparameter |
| Agent framework | LangGraph StateGraph + ToolNode | Custom memory nodes around ReAct loop |
| Vector store | Qdrant in-memory (langchain_qdrant) | Same API as Qdrant Cloud |
| Memory | InMemoryStore (semantic indexing) + MemorySaver (checkpointer) | Swap to PostgresStore later |
| External API | Tavily (langchain_tavily) | Trend search |
| Reranking | Cohere rerank-v3.5 | For improved retriever |
| Evaluation | RAGAS (6 RAG metrics + 3 agent metrics) | Jupyter notebooks |
| Monitoring | LangSmith | Native LangGraph tracing |
| Endpoint | FastAPI | Cert requirement |
| Package manager | uv | Course standard |

## Project Structure

```
stylists-ai-backend-cert/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   ├── config.py              # Settings from env vars
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py           # LangGraph StateGraph
│   │   ├── nodes.py           # load_memory, agent, save_memory
│   │   ├── state.py           # AgentState TypedDict
│   │   └── prompts.py         # System prompts
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── style_knowledge.py # search_style_knowledge (RAG)
│   │   ├── wardrobe.py        # query_wardrobe
│   │   ├── outfit.py          # generate_outfit
│   │   └── trends.py          # search_trends (Tavily)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py          # Chunk + embed + load to Qdrant
│   │   └── retriever.py       # Baseline + improved retrievers
│   ├── memory/
│   │   ├── __init__.py
│   │   └── store.py           # InMemoryStore setup + helpers
│   └── data/
│       └── wardrobe.py        # Sample wardrobe items
├── notebooks/
│   ├── 01_rag_baseline_eval.ipynb
│   ├── 02_advanced_retriever_eval.ipynb
│   └── 03_agent_eval.ipynb
├── knowledge/                 # 24 curated markdown files (exists)
├── tests/
├── docs/
├── .env
└── pyproject.toml
```

## Chunking Strategy

- **Baseline:** RecursiveCharacterTextSplitter, chunk_size=500, chunk_overlap=30
- Chunk size is a tunable hyperparameter for eval phase
- Metadata per chunk: `{source, domain}` extracted from file path

## Memory Architecture

- **Short-term:** MemorySaver checkpointer (conversation persistence via thread_id)
- **Long-term semantic:** InMemoryStore with OpenAI embeddings indexing
  - Namespace: `(user_id, "profile")` — style profile (updated in place)
  - Namespace: `(user_id, "observations")` — learned preferences (accumulates)

## Graph Structure

```
START → load_memory → agent (ReAct) ⇄ tools → save_memory → END
```

- `load_memory`: Read profile + semantically search observations → inject into system prompt
- `agent`: LLM with tools bound, loops until no more tool calls
- `tools`: ToolNode with all 4 tools
- `save_memory`: LLM extracts new observations from conversation, stores to InMemoryStore

## API Contract

```
POST /chat
Request:  { "message": str, "user_id": str, "thread_id": str }
Response: { "response": str, "tool_calls": list[dict], "observations_stored": list[str] }

GET /health
Response: { "status": "healthy", "model": "gpt-5.2" }
```

## Key Decisions

- RecursiveCharacterTextSplitter as baseline (course standard), not H2-header split
- GPT-5.2 over GPT-4o (current flagship, better tool calling)
- Embedding model as tunable hyperparameter
- Eval in Jupyter notebooks (course pattern), backend as Python package
- Details adjustable during implementation
