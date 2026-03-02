# Cert Challenge → Demo Day Handoff

What we built for cert, what's hardcoded or in-memory, and what needs to change for a multi-user production system.

**Last Updated:** March 2026

---

## 1. What We Built (Cert Challenge)

### RAG Pipeline
- **24 curated markdown files** across 6 domains (color_theory, body_shapes, style_archetypes, occasion_dressing, wardrobe_building, fundamentals)
- **Chunking:** RecursiveCharacterTextSplitter, 1000 chars / 100 overlap (validated via RAGAS eval — beat 250/50 and 500/50)
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dims)
- **Vector store:** Qdrant Cloud (collection: `fashion_knowledge`, 1365 chunks). Falls back to in-memory Qdrant for local dev.
- **Retrieval:** Naive dense similarity, k=10 (validated as best cost/latency tradeoff — ensemble scored higher but 2.4x slower and 5x more expensive)

### Agent
- **LangGraph ReAct agent** with tool-calling loop
- **2 tools:** `search_style_knowledge` (RAG), `search_trends` (Tavily)
- **System prompt** injects user profile + observations via `build_system_prompt()`
- **Checkpointer:** `MemorySaver` (in-memory, conversation history persists within a thread)

### API
- **FastAPI** with streaming SSE (default) and non-streaming JSON modes
- **POST /chat** — message → agent → streaming response with tool_call/tool_result events
- **GET /health** — server status
- **CORS** configured for localhost:3000 and Vercel frontend

### RAGAS Evaluation
- **Synthetic test set:** 21 questions (7 single-hop, 7 multi-hop abstract, 7 multi-hop specific)
- **Experiments run:** 3 chunking configs, 3 k-values, 5 retrieval strategies
- **LangSmith integration:** Cost/latency comparison across all strategies
- **Results:** Ensemble best composite (0.942), naive second (0.893). Naive chosen for production.
- **Notebook:** `evals/ragas_eval.ipynb` with full analysis and conclusion

### Tests
- 12 tests passing: agent reasoning, tool usage, API endpoints (streaming + non-streaming), RAG pipeline (load, chunk, vectorstore), tool functions

---

## 2. What's Hardcoded or In-Memory

These are the shortcuts we took for a single-user cert demo that need to change for production.

### 2.1 Demo Profile (Hardcoded)

**File:** `app/utils/streaming.py` — `DEMO_PROFILE` dict

```python
DEMO_PROFILE = {
    "gender": "male",
    "height": "5'10\"",
    "weight": "170 lbs",
    "body_shape": "inverted_triangle",
    "color_season": "deep_autumn",
    "skin_tone": "warm, dark",
    "style_archetype": "classic_natural",
    "preferences": {
        "loves": ["earth tones", "structured pieces", "clean lines"],
        "avoids": ["bright neons", "heavy patterns", "oversized fits"],
    },
    "lifestyle": {
        "work": "business_casual_office",
        "social": "casual_dinners",
        "active": "hiking",
    },
}
```

**What happens now:** Every request gets this exact profile injected into `build_input_state()`. The agent always "knows" this user.

**Production change:** Replace with `load_memory` graph node that reads from Store. Profile starts empty for new users and gets populated through onboarding or conversation.

### 2.2 No Memory Nodes

**Current graph:** `START → agent ⇄ tools → END`

**Production graph:** `START → load_memory → agent ⇄ tools → save_memory → END`

Missing pieces:
- `load_memory` node — reads profile + observations from Store, injects into state
- `save_memory` node — uses LLM to extract new facts/preferences from conversation, writes to Store
- `InMemoryStore` (cert) or `AsyncPostgresStore` (production) instance passed to graph

See `docs/plans/memory-system.md` for full implementation plan.

### 2.3 Wardrobe Data (Hardcoded Sample)

**File:** `app/data/wardrobe.py` — `SAMPLE_WARDROBE` list of 19 items

**What happens now:** Every user gets the same 19-item wardrobe designed for the Deep Autumn SV software engineer demo profile. The `query_wardrobe` tool filters this in-memory list.

**Production change:** Wardrobe items stored in database (Supabase Postgres). Users upload items via frontend — either manual entry or photo upload with vision model auto-extraction of attributes (see Section 9).

### 2.4 Observations Always Empty

**File:** `app/utils/streaming.py` — `build_input_state()` sets `observations: []`

**What happens now:** The agent never has learned observations about the user. The system prompt section says "No observations yet — this is a new user." every time.

**Production change:** `load_memory` node does semantic search over stored observations, returns the most relevant ones for the current query.

### 2.5 Conversation Memory (In-Memory Only)

**Current:** `MemorySaver` — conversation history lives in process memory. Server restart = all history lost.

**Production:** `AsyncPostgresSaver` backed by Supabase Postgres. History persists across restarts and deployments.

### 2.6 Vector Store Fallback

**File:** `rag/vectorstore.py` — `get_vector_store()`

**Current behavior:**
- If `QDRANT_URL` is set → connects to Qdrant Cloud (production-ready)
- If not set → builds in-memory vector store from knowledge files on every startup (~5-10s)

**Production:** Always use Qdrant Cloud. The in-memory fallback is for local dev only.

---

## 3. Profile Attributes — Onboarding Design

Not all profile fields are equal. Some are instant (users know them), others require guided discovery.

### Tier 1: Ask at Onboarding (Low Friction)
| Field | Why It's Easy |
|-------|--------------|
| `gender` | User knows immediately |
| `height` | User knows immediately |
| `weight` | User knows immediately (optional — can skip) |
| `lifestyle` (work/social/active) | Simple multiple choice |
| `preferences.loves` / `preferences.avoids` | "Pick styles you like" visual quiz |

### Tier 2: Progressive Discovery (Dashboard Features)
| Field | How to Discover | Product Feature |
|-------|----------------|-----------------|
| `color_season` | Guided quiz ("Is your skin warm or cool?") or selfie + vision model | Color Analysis Quiz (`/tools/color-analysis-quiz`) |
| `body_shape` | Measurement-based quiz or photo analysis | Body Type Calculator (`/tools/body-type-quiz`) |
| `style_archetype` | "Pick outfit A or B" preference quiz (10 rounds) | Style Quiz (`/tools/style-quiz`) |
| `skin_tone` | Derived from color season analysis | Part of Color Analysis |

### Design Principle

The system should work from day 1 with just Tier 1 fields, giving decent generic advice from RAG. Each Tier 2 field unlocked visibly improves personalization — that's the retention hook. The agent can prompt discovery: *"I notice you haven't determined your color season yet — want to take a quick quiz?"*

---

## 4. Wardrobe Item Schema

The schema is designed to work for both cert (hardcoded) and production (vision-extracted). Production-only fields are present with `null` values so no migration is needed.

```json
{
    "id": "item_001",
    "name": "Olive Cotton T-Shirt",

    "category": "tops",
    "sub_category": "t-shirt",

    "color": {"primary": "olive", "hex": "#708238"},
    "pattern": "solid",
    "fabric": "cotton",
    "fit": "regular",
    "style_tags": ["casual", "minimalist"],

    "formality": "casual",
    "seasons": ["spring", "summer", "fall"],
    "occasions": ["everyday", "wfh"],

    "image_url": null,
    "brand": null,
    "purchase_date": null,
    "wear_count": 0,
    "last_worn": null,
    "notes": null
}
```

### Field Categories

| Fields | Purpose | Source |
|--------|---------|--------|
| `id`, `name` | Identity | Auto-generated / vision model suggests, user confirms |
| `category`, `sub_category` | Classification | Vision model (high confidence) |
| `color`, `pattern`, `fabric`, `fit` | Visual attributes | Vision model (medium-high confidence) |
| `style_tags` | Style descriptors | Vision model (medium confidence) |
| `formality`, `seasons`, `occasions` | Context | Vision model suggests, user confirms |
| `image_url`, `brand` | Production metadata | User upload / OCR from labels |
| `wear_count`, `last_worn`, `notes` | Behavioral data | Tracked by app over time |

---

## 5. What's NOT Needed for Cert (Confirmed Cuts)

These were in the original PRD/architecture docs but are explicitly deferred:

| Feature | Why Cut | When to Build |
|---------|---------|---------------|
| `generate_outfit` tool | Agent can reason about outfits without a dedicated tool — just use `query_wardrobe` + agent reasoning | Demo day (if needed) |
| `load_memory` / `save_memory` nodes | Hardcoded demo profile is sufficient for cert demo | Demo day |
| Persistent storage (Postgres) | Single-user cert doesn't need persistence | Demo day |
| Multi-agent supervisor | Single ReAct agent handles all query types fine | Demo day |
| Episodic memory | Not needed without real multi-session users | Demo day |
| Procedural memory | Requires episodic first | Demo day |
| User auth | Single demo user | Demo day |
| Image upload pipeline | No images for cert wardrobe items | Demo day |

---

## 6. Existing Docs — What's Accurate vs. Outdated

| Document | Status | Notes |
|----------|--------|-------|
| `docs/PRD.md` | Partially outdated | Lists 4 tools (we have 3 — `search_style_knowledge`, `search_trends`, `query_wardrobe`). Lists `generate_outfit` (cut). Chunking strategy says 500/50 (now 1000/100). Retrieval says "metadata filter + rerank" as improved (we evaluated this — naive won for production). Memory section is still aspirational. |
| `docs/build_plan.md` | Still accurate for direction | SEO → queries → features mapping is valid. Week-by-week build sequence is outdated (we're past it). |
| `docs/tech_architecture.md` | Partially outdated | Cert section lists 4 tools, H2-header chunking (we use recursive), FAISS/Chroma (we use Qdrant). Demo day section is still valid as aspirational direction. |
| `docs/frontend_handoff.md` | Mostly accurate | API contract is correct. Lists tools as "coming Phase 3" — `search_trends` is now live. CORS section says "not yet configured" but it is. |
| `docs/plans/memory-system.md` | Accurate and ready | Implementation plan for adding Store + memory nodes. Follows exactly the path we'd take post-cert. |

---

## 7. Demo Day Upgrade Path

### Phase 1: Wardrobe + `query_wardrobe` Tool — DONE
- `app/data/wardrobe.py` — 19 sample items for Deep Autumn SV engineer
- `app/tools/query_wardrobe.py` — filters by category, color, occasion, season, formality
- Registered in agent graph, system prompt updated, 6 tests passing

### Phase 2: Memory System
1. Add `InMemoryStore` to graph compilation (see `docs/plans/memory-system.md`)
2. Implement `load_memory` node — reads profile + semantic search on observations
3. Implement `save_memory` node — LLM extracts new facts, stores as observations
4. Remove hardcoded `DEMO_PROFILE` from `streaming.py`
5. Update `build_input_state()` to pass empty profile (load_memory fills it)

### Phase 3: Persistent Storage
1. Add `langgraph-checkpoint-postgres` and `langgraph-store-postgres`
2. Swap `MemorySaver` → `AsyncPostgresSaver`
3. Swap `InMemoryStore` → `AsyncPostgresStore`
4. Add `SUPABASE_DB_URI` to config
5. Deploy to Railway/Render with Supabase Postgres

### Phase 4: Multi-Agent (If Going Deeper)
1. Add supervisor node for intent classification
2. Split single agent into 4 specialists (Outfit, Style, Wardrobe, Profile)
3. Each specialist gets focused system prompt + subset of tools
4. Add episodic memory (conversation summaries)
5. Add procedural memory (per-user agent instructions)

---

## 8. Key Files Reference

| File | Purpose | Hardcoded? |
|------|---------|-----------|
| `app/main.py` | FastAPI endpoints, lifespan init | No |
| `app/utils/streaming.py` | SSE streaming, `DEMO_PROFILE`, `build_input_state()` | **Yes** — demo profile |
| `app/agent/graph.py` | LangGraph ReAct agent | No |
| `app/agent/prompts.py` | System prompt with profile/observation injection | No |
| `app/agent/state.py` | AgentState TypedDict | No |
| `app/tools/style_knowledge.py` | RAG retrieval tool | No |
| `app/tools/search_trends.py` | Tavily trend search tool | No |
| `app/tools/query_wardrobe.py` | Wardrobe filter tool | No — but queries hardcoded sample data |
| `app/data/wardrobe.py` | 19 sample wardrobe items | **Yes** — hardcoded for demo profile |
| `rag/chunking.py` | Document chunking (1000/100 defaults) | No |
| `rag/vectorstore.py` | Qdrant vector store creation/connection | No |
| `rag/retrieval.py` | Retriever factory functions (naive, BM25, rerank, etc.) | No |
| `rag/loader.py` | Knowledge file loading with domain metadata | No |
| `config.py` | Environment-based settings | No |
| `scripts/ingest_to_qdrant.py` | One-time Qdrant Cloud ingestion | No |
| `evals/ragas_eval.ipynb` | Full evaluation notebook with conclusions | No |

---

## 9. Production Wardrobe Upload Pipeline (Vision)

For production, users upload photos of their closet. A vision pipeline extracts clothing items and their attributes automatically.

### Pipeline Flow

```
User uploads closet photo(s)
    │
    ▼
SAM3 (Segment Anything Model 3)
    → Segments individual clothing items from the photo
    → Outputs cropped images per item
    │
    ▼
Vision Model (GPT-4o / Gemini)
    → Extracts attributes from each cropped item image
    → Returns structured JSON matching our wardrobe schema
    │
    ▼
User Confirmation UI
    → Shows extracted attributes on a card per item
    → User confirms, edits, or discards
    → Confirmed items saved to wardrobe database
```

### What the Vision Model Can Extract

| Attribute | Confidence | Notes |
|-----------|-----------|-------|
| `category` | High | "jacket" vs "pants" — trivial for vision models |
| `sub_category` | High | "blazer" vs "hoodie" vs "puffer" |
| `color.primary` | High | Dominant color name |
| `color.hex` | Medium | Approximate hex from image pixels |
| `pattern` | High | solid, striped, plaid, floral, graphic |
| `fabric` | Medium | Cotton vs denim vs leather vs wool are visual. Silk vs polyester is hard. |
| `fit` | Medium | Slim vs regular vs oversized from garment shape |
| `style_tags` | Medium | "minimalist", "streetwear", "preppy" — models are decent |
| `formality` | Medium | Hoodie = casual, blazer = business_casual. Context-dependent. |
| `name` | Medium | Model generates reasonable name ("Navy Zip-Up Hoodie"), user can edit |

### What the Vision Model Should Suggest (User Confirms)

| Attribute | Why It Needs Confirmation |
|-----------|--------------------------|
| `seasons` | A flannel *looks* fall/winter but Bay Area people wear them year-round |
| `occasions` | Highly personal — one person's "office" shirt is another's "date_night" shirt |
| `brand` | Sometimes visible on logos/tags, often not |

### What Only the User Can Provide

| Attribute | Why |
|-----------|-----|
| `purchase_date` | Not visible in image |
| `wear_count` / `last_worn` | Behavioral data tracked over time |
| `notes` | Personal context ("runs small", "gift from mom") |

### Vision Model Prompt (Example)

```
Analyze this clothing item image and extract the following attributes
as JSON:

{
    "name": "descriptive name",
    "category": "tops|bottoms|outerwear|shoes|accessories",
    "sub_category": "specific type (e.g., t-shirt, jeans, sneakers)",
    "color": {"primary": "color name", "hex": "#hexcode"},
    "pattern": "solid|striped|plaid|floral|graphic|camo|other",
    "fabric": "best guess of material",
    "fit": "slim|regular|relaxed|oversized",
    "style_tags": ["2-4 style descriptors"],
    "formality": "casual|smart_casual|business_casual|formal",
    "seasons_suggestion": ["likely wearable seasons"],
    "occasions_suggestion": ["likely occasion tags"]
}
```

### Key Design Decisions for Production

1. **SAM3 first, then vision model** — segmenting before classifying lets users upload full closet photos instead of individual item photos. Better UX.
2. **Confirm-then-save** — never auto-save extracted items. Users see a review screen and can edit any field. This builds trust and catches vision model mistakes.
3. **Same schema, different source** — the `query_wardrobe` tool doesn't care whether items came from hardcoded data or vision extraction. Same filtering logic works on both.
4. **Incremental enrichment** — `wear_count` and `last_worn` start at 0/null and get updated as users log outfits over time.
