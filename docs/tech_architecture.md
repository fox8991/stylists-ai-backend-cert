# Stylists.ai — Technical Architecture

**Purpose:** Detailed technical architecture for the AI Stylist agent — tools, memory, RAG, LangGraph graph structure, and data schemas. Covers cert challenge implementation and demo day evolution.

**Companion files:**
- `stylists_ai_build_plan.md` — SEO → Queries → Features → Architecture (the "what")
- `stylists_ai_tech_architecture.html` — interactive visual diagrams
- This document — the "how" (implementation details)

**Last Updated:** February 2026

---

## 1. Architecture Overview

### The Core Pattern

Both cert and demo day follow the same fundamental pattern:

```
Memory Read → Agent Reasoning → Tool Calls → Response → Memory Write
```

The difference is complexity at each step:

| Step | Cert Challenge | Demo Day |
|------|---------------|----------|
| Memory Read | Load semantic observations from InMemoryStore | Load semantic + episodic + procedural from persistent store |
| Agent Reasoning | Single ReAct agent with all tools | Supervisor classifies intent → routes to specialist |
| Tool Calls | 4 tools (RAG, wardrobe, outfit gen, trends) | Same tools distributed across 4 specialists |
| Response | Text response | Text + potentially visual outfit cards |
| Memory Write | Extract and store new observations | Update semantic + summarize to episodic + refine procedural |

### What Stays The Same

Regardless of cert or demo day:
- **The same tools** — `search_style_knowledge`, `query_wardrobe`, `generate_outfit`, `search_trends` exist in both
- **The same RAG corpus** — fashion knowledge base, chunked and embedded the same way
- **The same data schemas** — wardrobe items, style profile, observations have the same JSON structure
- **The same memory store interface** — whether backed by InMemoryStore or PostgresStore, the tools use the same API

The agent code from cert doesn't get thrown away for demo day — it gets wrapped in a supervisor and split into specialists.

---

## 2. Cert Challenge Architecture

### 2.1 LangGraph Graph Structure

```
                    ┌─────────────┐
                    │  START       │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ load_memory │  ← Read user profile + observations
                    └──────┬──────┘     from InMemoryStore
                           │
                           ▼
                    ┌─────────────┐
                    │   agent     │  ← ReAct loop: reason → tool call → observe
                    │  (ReAct)    │     System prompt includes user profile
                    └──────┬──────┘
                           │
                     ┌─────┴─────┐
                     │           │
              tool_call?    no_tool_call
                     │           │
                     ▼           │
              ┌─────────────┐   │
              │    tools     │   │  ← search_style_knowledge
              │              │   │     query_wardrobe
              └──────┬──────┘   │     generate_outfit
                     │           │     search_trends
                     └─────┬─────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ save_memory  │  ← Extract new observations
                    └──────┬──────┘     Store to InMemoryStore
                           │
                           ▼
                    ┌─────────────┐
                    │    END      │
                    └─────────────┘
```

### 2.2 Graph Implementation (Pseudocode)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore

# State includes messages + user_id + profile
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_profile: dict         # loaded from memory
    observations: list[str]    # loaded from memory

# Initialize memory store
store = InMemoryStore()

# Define graph
graph = StateGraph(AgentState)

# Nodes
graph.add_node("load_memory", load_memory_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools=[
    search_style_knowledge,
    query_wardrobe,
    generate_outfit,
    search_trends,
]))
graph.add_node("save_memory", save_memory_node)

# Edges
graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "save_memory": "save_memory",
})
graph.add_edge("tools", "agent")  # loop back after tool call
graph.add_edge("save_memory", END)

app = graph.compile(checkpointer=MemorySaver(), store=store)
```

### 2.3 Node Implementations

**load_memory_node** — Runs before the agent. Reads the user's profile and recent observations from the Store, then injects them into the agent's system prompt.

```python
def load_memory_node(state: AgentState, *, store: BaseStore):
    user_id = state["user_id"]

    # Read profile (profile pattern — single document)
    profile_items = store.search(("users", user_id, "profile"))
    profile = profile_items[0].value if profile_items else {}

    # Read recent observations (collection pattern — multiple docs)
    observations = store.search(
        ("users", user_id, "observations"),
        query=state["messages"][-1].content,  # semantic search
        limit=5
    )
    obs_texts = [o.value["content"] for o in observations]

    return {
        "user_profile": profile,
        "observations": obs_texts,
    }
```

**agent_node** — The ReAct agent. Uses a system prompt that includes the user's profile and observations.

```python
def agent_node(state: AgentState):
    profile = state.get("user_profile", {})
    observations = state.get("observations", [])

    system_prompt = f"""You are a personal AI stylist for Stylists.ai.

## User's Style Profile
{json.dumps(profile, indent=2) if profile else "No profile yet — ask the user about their style preferences."}

## What You Know About This User
{chr(10).join(f"- {obs}" for obs in observations) if observations else "No observations yet — this is a new user."}

## Your Tools
- search_style_knowledge: Search the fashion knowledge base for styling advice (color theory, body shapes, dress codes, etc.)
- query_wardrobe: Query the user's wardrobe items by category, color, occasion, etc.
- generate_outfit: Generate an outfit combination from the user's wardrobe for a specific context
- search_trends: Search current fashion trends using Tavily

## Guidelines
- Ground your advice in retrieved fashion knowledge (cite the principles)
- Always suggest from the user's OWN wardrobe first
- If you learn new facts about the user (color season, body shape, preferences), note them — they'll be saved after this conversation
- Explain the "why" behind your suggestions"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    return {"messages": [response]}
```

**save_memory_node** — Runs after the agent responds. Extracts new observations from the conversation and stores them.

```python
def save_memory_node(state: AgentState, *, store: BaseStore):
    user_id = state["user_id"]

    # Use LLM to extract observations from the conversation
    extraction_prompt = """Review this conversation and extract any new facts
    or preferences learned about the user. Return as JSON array of strings.
    Examples: "User is a Deep Autumn color season", "User prefers earth tones",
    "User dislikes bright patterns". Return [] if nothing new was learned."""

    conversation = format_messages(state["messages"])
    result = llm.invoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(content=conversation)
    ])

    new_observations = json.loads(result.content)

    for obs in new_observations:
        store.put(
            ("users", user_id, "observations"),
            str(uuid4()),  # unique key for each observation
            {"content": obs, "created_at": datetime.now().isoformat()}
        )

    # Update profile if structured facts were learned
    # (e.g., color_season, body_shape — these go in the profile, not observations)
    profile_update = extract_profile_facts(state["messages"])
    if profile_update:
        existing = store.search(("users", user_id, "profile"))
        current_profile = existing[0].value if existing else {}
        current_profile.update(profile_update)
        store.put(("users", user_id, "profile"), "profile", current_profile)

    return state
```

### 2.4 Tools — Detailed Specifications

#### search_style_knowledge (RAG Retrieval)

```python
@tool
def search_style_knowledge(query: str, domain: str = None) -> str:
    """Search the fashion knowledge base for styling advice.

    Args:
        query: What styling knowledge to search for (e.g., "Deep Autumn color palette",
               "inverted triangle body shape styling tips", "business casual dress code")
        domain: Optional filter — one of: color_theory, body_shapes, style_archetypes,
                occasion_dressing, wardrobe_building, fundamentals

    Returns:
        Retrieved styling knowledge relevant to the query.
    """
    search_kwargs = {"k": 5}
    if domain:
        search_kwargs["filter"] = {"domain": domain}

    docs = vector_store.similarity_search(query, **search_kwargs)
    return "\n\n".join([
        f"[{doc.metadata.get('source', 'fashion_knowledge')}]\n{doc.page_content}"
        for doc in docs
    ])
```

**RAG corpus details:**

| Domain | # Documents | Example Topics | Chunk Size |
|--------|-------------|----------------|------------|
| color_theory | 6-8 | 12 seasons, undertones, color combinations | ~400 tokens |
| body_shapes | 6-8 | Inverted triangle, rectangle, hourglass, pear, petite, styling principles | ~400 tokens |
| style_archetypes | 6-8 | Classic, romantic, dramatic, natural, creative, aesthetic guide | ~400 tokens |
| occasion_dressing | 5-6 | Business casual (91K keyword), smart casual, formal, date night (60K keyword), casual | ~400 tokens |
| wardrobe_building | 4-5 | Capsule wardrobe (880/mo keyword), essentials, mix-and-match, seasonal rotation | ~400 tokens |
| fundamentals | 4-5 | Fabric guide, fit principles, proportion/balance, color matching clothes | ~400 tokens |

**Total:** ~35-40 source documents → ~100-150 chunks after splitting

**Embedding model:** OpenAI `text-embedding-3-small` (1536 dimensions)

**Vector store:** Qdrant `:memory:` for cert, Qdrant Cloud for demo day

**Metadata per chunk:**
```json
{
    "source": "color_theory/seasonal_color_analysis.md",
    "domain": "color_theory",
    "subtopic": "deep_autumn",
    "specificity": "specific_type",
    "related_keywords": ["deep autumn", "warm colors", "earth tones"]
}
```

#### query_wardrobe

```python
@tool
def query_wardrobe(
    category: str = None,
    color: str = None,
    occasion: str = None,
    season: str = None,
    formality: str = None,
    limit: int = 20
) -> str:
    """Query the user's wardrobe items with optional filters.

    Args:
        category: Filter by category (tops, bottoms, dresses, outerwear, shoes, accessories)
        color: Filter by primary color
        occasion: Filter by occasion tag (business, casual, formal, date_night, etc.)
        season: Filter by season (spring, summer, fall, winter)
        formality: Filter by formality (casual, smart_casual, business_casual, formal)
        limit: Max items to return

    Returns:
        JSON list of matching wardrobe items with attributes.
    """
    # For cert: filter in-memory list of dicts
    # For demo day: SQL query against Supabase
    items = wardrobe_data  # list of dicts

    if category:
        items = [i for i in items if i["category"] == category]
    if color:
        items = [i for i in items if i["color"]["primary"].lower() == color.lower()]
    if occasion:
        items = [i for i in items if occasion in i.get("occasions", [])]
    if season:
        items = [i for i in items if season in i.get("season", [])]
    if formality:
        items = [i for i in items if i.get("formality") == formality]

    return json.dumps(items[:limit], indent=2)
```

#### generate_outfit

```python
@tool
def generate_outfit(
    occasion: str,
    weather: str = None,
    mood: str = None,
    constraints: str = None
) -> str:
    """Generate an outfit recommendation from the user's wardrobe.

    Args:
        occasion: What the outfit is for (e.g., "job interview", "casual dinner", "weekend brunch")
        weather: Current weather conditions (e.g., "65°F, partly cloudy")
        mood: How the user wants to feel (e.g., "confident", "comfortable", "creative")
        constraints: Any constraints (e.g., "no heels", "must include the new blazer")

    Returns:
        JSON with selected items, reasoning, and styling tips.
    """
    # This tool fetches the wardrobe internally and uses LLM to compose
    items = get_all_wardrobe_items()  # full wardrobe

    prompt = f"""Select an outfit from these wardrobe items for: {occasion}
    Weather: {weather or 'not specified'}
    Mood: {mood or 'not specified'}
    Constraints: {constraints or 'none'}

    Wardrobe: {json.dumps(items)}

    Return JSON with:
    - outfit: list of item_ids with their role (top, bottom, shoes, accessory, outerwear)
    - reasoning: why these items work together
    - styling_tips: 1-2 tips for wearing this outfit"""

    result = llm.invoke(prompt)
    return result.content
```

#### search_trends

```python
@tool
def search_trends(query: str) -> str:
    """Search current fashion trends using Tavily web search.

    Args:
        query: What trend to search for (e.g., "spring 2026 fashion trends",
               "are wide leg pants still in style", "business casual trends")

    Returns:
        Summary of current trend information from the web.
    """
    results = tavily_client.search(
        query=f"fashion trends {query} 2026",
        search_depth="basic",
        max_results=3
    )
    return "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in results['results']
    ])
```

### 2.5 Sample Wardrobe Data (Cert)

For the cert demo, we preload a realistic wardrobe (15-20 items). This simulates what the SAM3 ingestion pipeline would produce in production.

```python
SAMPLE_WARDROBE = [
    {
        "id": "item_001",
        "name": "Navy Wool Blazer",
        "category": "outerwear",
        "color": {"primary": "navy", "hex": "#1b3a5c"},
        "fabric": "wool blend",
        "pattern": "solid",
        "formality": "business_casual",
        "season": ["fall", "winter", "spring"],
        "occasions": ["business", "smart_casual", "date_night"],
        "style_tags": ["classic", "structured", "versatile"],
    },
    {
        "id": "item_002",
        "name": "Cream Silk Blouse",
        "category": "tops",
        "color": {"primary": "cream", "hex": "#f5f0e1"},
        "fabric": "silk",
        "pattern": "solid",
        "formality": "smart_casual",
        "season": ["spring", "summer", "fall"],
        "occasions": ["business", "date_night", "smart_casual"],
        "style_tags": ["elegant", "feminine", "versatile"],
    },
    # ... 13-18 more items covering all categories
]
```

The wardrobe should be designed to work with a specific color season (Deep Autumn) so we can demonstrate personalization. Items should include a mix of colors — some in-palette (rust, olive, cream, burgundy) and some out-of-palette (bright red, neon) — so the agent can show awareness.

### 2.6 RAGAS Evaluation Setup

**What we evaluate:** Style education queries where the agent uses RAG retrieval.

**Test set structure:**

```python
eval_dataset = [
    {
        "question": "What colors look best for a Deep Autumn color season?",
        "ground_truth": "Deep Autumn colors include rich earth tones: olive, rust, burgundy, mustard, chocolate brown, warm cream, forest green, terracotta. Avoid cool pastels, icy blues, and neon colors.",
        "contexts": []  # filled by retriever during eval
    },
    {
        "question": "How should someone with an inverted triangle body shape dress?",
        "ground_truth": "Balance broader shoulders with wider bottoms: A-line skirts, wide-leg pants, bootcut jeans. Avoid shoulder pads, structured shoulders, or boat necks. V-necks and scoop necks draw the eye downward.",
        "contexts": []
    },
    # ... 15-20 more pairs
]
```

**Metrics:**
- **Faithfulness:** Does the agent's answer stay true to retrieved documents? (Target: > 0.8)
- **Context Precision:** Are the top retrieved chunks actually relevant? (Target: > 0.7)
- **Context Recall:** Did we retrieve enough relevant information? (Target: > 0.7)

**Retriever improvement strategy:**
1. **Baseline:** Naive dense retrieval (embed query → cosine similarity → top-5)
2. **Improved:** Metadata filtering by domain + reranking with Cohere reranker
   - When user asks about colors → filter to `domain: color_theory` before similarity search
   - Retrieve top-15 → rerank to top-5 with cross-encoder
3. **Compare RAGAS scores** between baseline and improved

### 2.7 Cert Tech Stack Summary

| Component | Technology | Notes |
|-----------|-----------|-------|
| LLM | GPT-4o (or Claude 3.5 Sonnet) | Primary reasoning + tool calling |
| Agent framework | LangGraph | StateGraph with ReAct pattern |
| Embedding | OpenAI text-embedding-3-small | 1536 dims, good quality/cost |
| Vector store | Qdrant `:memory:` | In-memory for cert, Cloud for prod |
| Memory store | LangGraph InMemoryStore | Namespace-scoped per user |
| Checkpointer | MemorySaver | Conversation persistence |
| External API | Tavily | Web search for trends |
| Evaluation | RAGAS | Faithfulness, precision, recall |
| Monitoring | LangSmith | Trace agent decisions |
| UI | Chainlit (or Streamlit) | Chat interface for demo |
| Endpoint | FastAPI | Local endpoint (cert requirement) |

---

## 3. Demo Day Architecture (Deeper Path)

### 3.1 Multi-Agent LangGraph Structure

```
                        ┌─────────────┐
                        │   START     │
                        └──────┬──────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ load_memory  │  ← Load all 3 memory types
                        │              │     (semantic + episodic + procedural)
                        └──────┬──────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  supervisor  │  ← Classifies intent, routes
                        │   (router)   │
                        └──────┬──────┘
                               │
                ┌──────────────┼──────────────┐──────────────┐
                │              │              │              │
                ▼              ▼              ▼              ▼
         ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
         │   outfit    │ │   style    │ │  wardrobe  │ │  profile   │
         │ specialist  │ │  advisor   │ │  analyst   │ │  builder   │
         │             │ │            │ │            │ │            │
         │ Tools:      │ │ Tools:     │ │ Tools:     │ │ Tools:     │
         │ -query_     │ │ -search_   │ │ -query_    │ │ -analyze_  │
         │  wardrobe   │ │  style_    │ │  wardrobe  │ │  color_    │
         │ -generate_  │ │  knowledge │ │ -analyze_  │ │  season    │
         │  outfit     │ │ -search_   │ │  gaps      │ │ -update_   │
         │ -search_    │ │  trends    │ │            │ │  profile   │
         │  trends     │ │            │ │            │ │            │
         └──────┬─────┘ └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
                │              │              │              │
                └──────────────┼──────────────┘──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ memory_mgr   │  ← Update semantic, episodic,
                        │              │     and procedural memory
                        └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │    END      │
                        └─────────────┘
```

### 3.2 Supervisor Implementation

The supervisor is a lightweight LLM call that classifies intent and returns a routing decision.

```python
def supervisor_node(state: AgentState):
    """Classify user intent and route to specialist."""

    routing_prompt = """Classify this user message into exactly one intent:
    - outfit_request: User wants outfit suggestions or help getting dressed
    - style_question: User wants styling advice or education (colors, body shape, etc.)
    - wardrobe_query: User wants to know about what's in their wardrobe, gaps, analysis
    - profile_update: User is sharing style info (quiz results, preferences, body shape)

    Return JSON: {"intent": "...", "reasoning": "..."}"""

    result = llm.invoke([
        SystemMessage(content=routing_prompt),
        state["messages"][-1]
    ])

    intent = json.loads(result.content)["intent"]
    return {"next_agent": intent}
```

### 3.3 Specialist System Prompts

Each specialist gets a focused persona:

**Outfit Specialist:**
```
You are the Outfit Specialist for Stylists.ai. Your job is to create
outfit combinations from the user's wardrobe. You consider:
- The user's color season and body shape (from their profile)
- The occasion, weather, and mood
- Past outfit feedback (what they liked/disliked)
- Fashion principles from the knowledge base
Always explain WHY items work together.
```

**Style Advisor:**
```
You are the Style Advisor for Stylists.ai. Your job is to educate users
about styling principles. You ground every answer in the fashion
knowledge base. When the user has a known color season or body shape,
personalize your advice. You're teaching them to understand their style,
not just telling them what to wear.
```

**Wardrobe Analyst:**
```
You are the Wardrobe Analyst for Stylists.ai. Your job is to analyze the
user's wardrobe holistically — identify gaps, suggest versatile additions,
compare against capsule wardrobe frameworks. Cross-reference items against
the user's color season to flag out-of-palette pieces.
```

**Profile Builder:**
```
You are the Profile Builder for Stylists.ai. Your job is to build and
maintain the user's style profile. You can conduct mini style assessments
through conversation, process quiz results, and extract style preferences.
Every fact you learn gets stored in the user's profile for other agents to use.
```

### 3.4 Memory Evolution

**Semantic Memory (cert → demo day enhancement):**
Cert already stores observations. Demo day adds semantic search (via pgvector) so the agent can retrieve the most relevant observations for the current query, not just the most recent ones.

**Episodic Memory (new in demo day):**
After each conversation, a summary is stored as an episode.

```python
def store_episode(state, store):
    """Summarize this conversation and store as episodic memory."""
    summary_prompt = """Summarize this conversation in 2-3 sentences.
    Focus on: what the user asked for, what was suggested, and how they reacted.
    Format: {"situation": "...", "suggestion": "...", "outcome": "..."}"""

    summary = llm.invoke([
        SystemMessage(content=summary_prompt),
        *state["messages"]
    ])

    store.put(
        ("users", state["user_id"], "episodes"),
        str(uuid4()),
        {
            **json.loads(summary.content),
            "created_at": datetime.now().isoformat()
        }
    )
```

The Outfit Specialist retrieves relevant episodes as few-shot examples: "Last time you asked for a formal outfit, I suggested the navy blazer + cream blouse and you loved it."

**Procedural Memory (new in demo day):**
After every N interactions (e.g., every 5), the Memory Manager reviews recent episodes and updates per-user instructions.

```python
def update_procedural_memory(state, store):
    """Review recent episodes and refine agent instructions."""
    user_id = state["user_id"]

    # Load current instructions
    current = store.search(("agent", user_id, "instructions"))
    current_instructions = current[0].value["instructions"] if current else DEFAULT_INSTRUCTIONS

    # Load recent episodes
    episodes = store.search(("users", user_id, "episodes"), limit=10)

    refinement_prompt = f"""Current agent instructions for this user:
    {current_instructions}

    Recent interactions:
    {json.dumps([e.value for e in episodes], indent=2)}

    Based on these interactions, should the instructions be updated?
    Look for patterns: repeated preferences, consistent feedback, new constraints.
    Return the updated instructions (keep what works, add new insights)."""

    updated = llm.invoke(refinement_prompt)

    store.put(
        ("agent", user_id, "instructions"),
        "instructions",
        {
            "instructions": updated.content,
            "version": (current[0].value.get("version", 0) + 1) if current else 1,
            "updated_at": datetime.now().isoformat()
        }
    )
```

### 3.5 Demo Day Data Layer

| Component | Cert | Demo Day |
|-----------|------|----------|
| Wardrobe items | In-memory list of dicts | Supabase Postgres table |
| Fashion RAG | Qdrant `:memory:` | Qdrant Cloud (persistent) |
| Semantic memory | InMemoryStore | PostgresStore + pgvector |
| Episodic memory | Not implemented | PostgresStore + pgvector |
| Procedural memory | Not implemented | PostgresStore (profile pattern) |
| Checkpointer | MemorySaver | PostgresSaver (Supabase) |
| User auth | None (single test user) | Better Auth via ShipAny |

**Why dual vector stores in production:**
- **Qdrant** = shared fashion knowledge (read-only, thousands of chunks, benefits from HNSW + payload filtering)
- **pgvector** = per-user memory search (read-write, hundreds per user, already in Postgres with the rest of user data)

---

## 4. Data Schemas

These schemas are the contract between the agent's tools and the storage backend. They're the same whether backed by InMemoryStore (cert) or PostgresStore (production).

### 4.1 Namespace Structure

```
Context (raw data — what the agent works WITH):
  ("users", user_id, "profile")              → style profile JSON (profile pattern)
  ("users", user_id, "wardrobe", item_id)    → wardrobe item (profile pattern)
  ("users", user_id, "history")              → outfit suggestion log (collection)

Semantic Memory (learned facts):
  ("users", user_id, "observations")         → learned preferences (collection + embeddings)

Episodic Memory (past interactions):
  ("users", user_id, "episodes")             → conversation summaries (collection + embeddings)

Procedural Memory (agent behavior):
  ("agent", user_id, "instructions")         → per-user agent rules (profile + versioned)
```

**Profile vs Collection:**
- Profile = updated in place. Body shape, current instructions. New value supersedes old.
- Collection = entries accumulate. "Loves earth tones" doesn't replace "dislikes bright patterns" — both are signals. Semantic search finds the most relevant when the collection grows.

### 4.2 Style Profile Schema

```json
{
    "body_shape": "inverted_triangle",
    "height": "5'10\"",
    "color_season": "deep_autumn",
    "style_archetype": "classic_natural",
    "preferences": {
        "loves": ["earth tones", "structured pieces", "clean lines"],
        "avoids": ["bright neons", "heavy patterns", "oversized fits"],
        "comfort_priority": "high"
    },
    "lifestyle": {
        "work": "business_casual_office",
        "social": "casual_dinners",
        "active": "yoga_hiking"
    }
}
```

Fields like `body_shape` and `height` are raw user inputs (context). Fields like `color_season`, `style_archetype`, and `preferences` are learned — they start empty and get populated through conversations or quiz results. This makes the profile a hybrid of context and memory.

### 4.3 Wardrobe Item Schema

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
    "style_tags": ["classic", "versatile"],
    "wear_count": 12,
    "last_worn": "2026-02-10",
    "image_url": "...",
    "added_date": "2025-11-15"
}
```

For cert, `image_url`, `wear_count`, and `last_worn` can be omitted or mocked. The essential fields for outfit generation are: id, name, category, color, formality, season, occasions, style_tags.

### 4.4 Observation Schema (Semantic Memory)

```json
{
    "content": "User consistently prefers structured blazers over cardigans for formal events. Rejected soft/unstructured suggestions 3 times.",
    "created_at": "2026-02-18T14:30:00Z",
    "context": "outfit_suggestion_feedback",
    "confidence": "high"
}
```

### 4.5 Episode Schema (Episodic Memory — Demo Day)

```json
{
    "situation": "User asked for cocktail party outfit. Winter evening, outdoor venue.",
    "suggestion": "Navy midi dress + gold statement necklace + camel wool coat + ankle boots.",
    "outcome": "positive — user said 'felt confident and comfortable all night'",
    "created_at": "2026-01-20T19:00:00Z"
}
```

### 4.6 Agent Instructions Schema (Procedural Memory — Demo Day)

```json
{
    "version": 3,
    "instructions": "Key rules for this user:\n- Always prioritize comfort\n- Suggest wardrobe items first, purchases as last resort\n- Avoid bright neons and heavy patterns\n- For formal events, default to structured blazers\n- Ask about occasion before suggesting",
    "updated_at": "2026-02-15T10:00:00Z",
    "update_reason": "User feedback: 'Stop suggesting things I need to buy'"
}
```

---

## 5. Three Architecture Decisions (Preserved from System Design)

These three decisions are independent and should be made separately:

### Decision 1: Interface — How does the agent access data?

**Choice: Tier 1 — Filesystem-shaped tool functions**

The agent calls structured tools (`query_wardrobe()`, `search_memory()`, etc.) backed by whatever storage we choose. Each call is a separate tool invocation. The styling agent mostly does reads and lookups — it doesn't need full bash composability (Tier 2) or code execution (Tier 3).

### Decision 2: Deployment — How are users isolated?

**Choice: Application-level isolation**

`WHERE user_id = ?` on every query. Namespace scoping in the Store handles memory isolation. No containers needed unless we add deep agent features with real bash execution.

### Decision 3: Storage — Where does data live?

**Choice: InMemoryStore (cert) → PostgresStore + pgvector (demo day)**

The threshold for needing a database isn't crossed in cert (single user, small data). It IS crossed for demo day (multi-user, semantic search over memories, concurrent sessions).

**The context vs memory distinction:**
- **Context** = raw data the agent works WITH. Wardrobe items, outfit history logs, plans. Records of what exists or happened.
- **Memory** = distilled knowledge the agent has LEARNED. Observations, episodes, behavioral rules. Extracted insights, not raw data.

The outfit history *records* that navy blazer was suggested on Feb 10. Memory *learns* that "user prefers structured blazers for formal events." One is raw data, the other is extracted insight.

---

## 6. RAG Pipeline Details

### 6.1 Corpus Curation

**Source material:** We curate this ourselves from styling knowledge. This is our "own data" for the cert requirement.

```
fashion_knowledge/
├── color_theory/
│   ├── seasonal_color_analysis.md      (12 seasons overview)
│   ├── warm_seasons.md                 (Spring, Autumn — Deep/Warm/Soft)
│   ├── cool_seasons.md                 (Summer, Winter — Deep/Cool/Soft)
│   ├── undertone_guide.md              (warm/cool/neutral identification)
│   └── color_combinations.md           (complementary, analogous, etc.)
│
├── body_shapes/
│   ├── inverted_triangle.md
│   ├── rectangle.md
│   ├── hourglass.md
│   ├── triangle_pear.md
│   ├── petite_styling.md
│   └── general_principles.md           (proportion, balance, line)
│
├── style_archetypes/
│   ├── classic.md
│   ├── romantic.md
│   ├── dramatic.md
│   ├── natural.md
│   ├── creative.md
│   └── modern_aesthetics.md            (Dark Academia, Coastal Grandmother, etc.)
│
├── occasion_dressing/
│   ├── business_casual.md
│   ├── smart_casual.md
│   ├── formal_events.md
│   ├── date_night.md
│   └── casual_everyday.md
│
├── wardrobe_building/
│   ├── capsule_wardrobe.md
│   ├── essential_pieces.md
│   ├── mix_and_match.md
│   └── seasonal_rotation.md
│
└── fundamentals/
    ├── fabric_guide.md
    ├── fit_principles.md
    └── proportion_and_balance.md
```

### 6.2 Chunking Strategy

- Split by H2 headers (natural topic boundaries)
- Target ~300-500 tokens per chunk
- Overlap: 50 tokens between chunks
- Each chunk gets metadata: `{domain, subtopic, specificity, related_keywords}`

**Why header-based splitting:** Fashion content has clear topical sections. A chunk about "Deep Autumn palette" shouldn't bleed into "Light Spring palette." Header-based splitting preserves this boundary naturally.

### 6.3 Retrieval Strategy

**Baseline (cert — for RAGAS comparison):**
```python
# Naive dense retrieval
docs = vector_store.similarity_search(query, k=5)
```

**Improved (cert — show RAGAS improvement):**
```python
# Step 1: Metadata filter (if domain is detectable from query)
domain = classify_query_domain(query)  # simple LLM call or keyword match
filter_kwargs = {"filter": {"domain": domain}} if domain else {}

# Step 2: Retrieve more candidates
docs = vector_store.similarity_search(query, k=15, **filter_kwargs)

# Step 3: Rerank with cross-encoder
reranked = cohere_reranker.rerank(query, docs, top_n=5)
```

Expected improvement: metadata filtering alone should improve context precision by 10-15%. Adding reranking should improve both precision and recall.

---

## 7. FastAPI Endpoint (Cert Requirement)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Stylists.ai Agent API")

class ChatRequest(BaseModel):
    message: str
    user_id: str = "demo_user"
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict] = []
    observations_stored: list[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the Stylist Agent."""
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    result = await agent_graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)],
         "user_id": request.user_id},
        config=config
    )

    return ChatResponse(
        response=result["messages"][-1].content,
        tool_calls=extract_tool_calls(result),
        observations_stored=result.get("new_observations", [])
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "gpt-4o"}
```

---

## 8. Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single vs multi-agent (cert) | Single ReAct agent | Simpler to build, debug, eval. Memory layer makes it non-trivial. |
| Single vs multi-agent (demo) | Supervisor + 4 specialists | Shows multi-agent pattern. Same tools, distributed. |
| Memory for cert | Semantic only (InMemoryStore) | Demonstrates learning without complexity of 3 memory types. |
| Memory for demo | Semantic + Episodic + Procedural | Full CoALA framework. Shows progressive personalization. |
| RAG vs fine-tuning | RAG | Fashion knowledge changes. RAG is updatable. Fine-tuning locks knowledge. |
| Vector store | Qdrant (RAG) + pgvector (memory) | Different data, different access patterns, different scale. |
| LLM | GPT-4o | Vision capability (future), strong tool calling, JSON mode. |
| Embeddings | text-embedding-3-small | Good quality/cost balance for fashion domain. |
| UI (cert) | Chainlit | Fastest to build chat interface with streaming. |
| Endpoint | FastAPI | Standard, lightweight, async support. |
| Evaluation | RAGAS | Required by cert. Faithfulness + precision + recall. |
| Monitoring | LangSmith | Native LangGraph integration. |

---

*Companion file: `stylists_ai_tech_architecture.html` — interactive diagrams for agent graph, tool flows, and memory evolution*
