# Stylists.ai — AI Agent Build Plan

**Purpose:** Map the path from SEO keywords → user queries → core features → technical architecture for both the AIE9 cert challenge and demo day.

**Thought Process:** Every feature we build traces back to what users actually search for. SEO keywords reveal user intent, user intent maps to queries our agent handles, queries define features, and features dictate the technical architecture.

**Last Updated:** February 2026

---

## 1. SEO Keywords We're Targeting

### Cert Challenge Keywords (P0 — Core Product)

These keywords map to the authenticated AI Stylist experience at `/app/*`.

| Keyword Cluster | Volume/mo | Target URL | Maps To |
|-----------------|-----------|------------|---------|
| "outfit generator", "ai outfit generator", "outfit maker" | 8,000 | `/outfit-generator` → `/app/outfits` | Outfit Generation feature |
| "ai personal stylist", "ai stylist", "ai fashion stylist" | 1,200 | `/` homepage | Overall product positioning |
| "smart closet", "digital wardrobe", "closet organizer app" | 700 | `/wardrobe` → `/app/wardrobe` | Wardrobe Intelligence feature |
| "capsule wardrobe checklist", "capsule wardrobe planner" | 880 | `/tools/capsule-wardrobe-builder` | Wardrobe Intelligence queries |

**Total cert-relevant volume:** ~10,800/mo direct, but the agent also uses knowledge from P1 keywords (below).

### Knowledge Keywords (P1 — Powers Agent RAG + Future SEO Tools)

These are high-volume SEO tool keywords. We don't build the tools for cert, but the **knowledge behind them is our RAG corpus** and the **results become user profile data** stored in agent memory.

| Keyword Cluster | Volume/mo | Future Tool URL | Agent Use |
|-----------------|-----------|-----------------|-----------|
| "color analysis quiz", "what colors look good on me", "seasonal color analysis" | 54,000 | `/tools/color-analysis-quiz` | RAG: color theory knowledge. Memory: user's color season stored as profile fact |
| "body type calculator", "body shape calculator", "what is my body type" | 56,000 | `/tools/body-type-quiz` | RAG: body shape styling guides. Memory: user's body shape stored as profile fact |
| "style quiz", "fashion style quiz", "how to find my style" | 53,000 | `/tools/style-quiz` | RAG: style archetype descriptions. Memory: user's style archetype stored as profile fact |
| "face shape detector", "what face shape do I have" | 36,000 | `/tools/face-shape-analyzer` | RAG: face shape + hairstyle/glasses recs. Memory: user's face shape |
| "spring colors", "deep autumn", "color palette" | 75,000 | `/tools/color-palette-explorer` | RAG: seasonal palette details. Used when suggesting outfit colors |

**Total knowledge keyword volume:** ~274,000/mo — this is the traffic the SEO tools will eventually capture, but for now, the knowledge feeds our agent.

### Demo Day Additional Keywords

| Keyword Cluster | Volume/mo | Target URL | Maps To |
|-----------------|-----------|------------|---------|
| All cert keywords | ~10,800 | Same | Same, but enhanced |
| Color analysis (if building quiz demo) | 54,000 | `/tools/color-analysis-quiz` | Color Analysis feature (tool → profile → memory) |
| Body shape guides (blog content knowledge) | 26,000 | `/blog/body-shape-*` | Richer RAG corpus for body shape advice |

---

## 2. User Queries → How People Use the System

### Outfit Generation Queries

**SEO origin:** "outfit generator" (8K/mo), "ai outfit generator" (1.6K/mo), "outfit maker" (4.4K/mo)

| User Query | What They Want | Cert | Demo Day |
|------------|---------------|------|----------|
| "What should I wear to a job interview?" | Outfit from wardrobe for specific occasion | ✅ Basic: picks items matching "formal" | ✅ Enhanced: factors in user's color season, body shape, past preferences |
| "Put together a casual weekend outfit" | Low-effort styling from own clothes | ✅ Filters casual items, suggests combo | ✅ + avoids recently worn combos (episodic memory) |
| "I have a date tonight, what works?" | Contextual outfit suggestion | ✅ Occasion-aware generation | ✅ + knows user's confidence preferences from past feedback |
| "What should I wear this week?" | Multi-day planning | ❌ Not in cert | ✅ Multi-outfit generation, no repeats, occasion-aware per day |

### Style Education Queries

**SEO origin:** "what colors look good on me" (33K/mo), "what is my body type" (60K/mo), "how to find my style" (2.4K/mo)

| User Query | What They Want | Cert | Demo Day |
|------------|---------------|------|----------|
| "What colors look good on me?" | Color season guidance | ✅ RAG retrieval from color theory corpus + asks clarifying Qs | ✅ + if color quiz taken, uses stored result for personalized answer |
| "I have an inverted triangle body shape, what should I wear?" | Body-shape-specific styling | ✅ RAG retrieval from body shape guides | ✅ + cross-references with wardrobe ("here are items you own that work") |
| "What's the difference between business casual and smart casual?" | Style education | ✅ RAG retrieval | ✅ Same |
| "How do I build a capsule wardrobe?" | Wardrobe building advice | ✅ RAG + general advice | ✅ + analyzes current wardrobe against capsule framework |

### Wardrobe Intelligence Queries

**SEO origin:** "smart closet" (590/mo), "closet organizer" (260/mo), "digital wardrobe" (110/mo)

| User Query | What They Want | Cert | Demo Day |
|------------|---------------|------|----------|
| "Show me all my blue tops" | Wardrobe filtering | ✅ query_wardrobe tool with color filter | ✅ Same |
| "Do I have enough business casual options?" | Wardrobe gap analysis | ✅ Basic count/filter | ✅ Enhanced: compares against capsule framework, suggests fills |
| "What's missing from my wardrobe?" | Gap identification | ✅ Basic: checks category balance | ✅ + factors in lifestyle needs from memory |
| "Which of my clothes match my color season?" | Profile-aware wardrobe analysis | ❌ Not in cert | ✅ Cross-references color season with item colors |

### Trend Queries

**SEO origin:** General fashion interest, powered by Tavily external API

| User Query | What They Want | Cert | Demo Day |
|------------|---------------|------|----------|
| "What's trending for spring 2026?" | Current trend info | ✅ Tavily search, summarized | ✅ + "here's what's trending that matches YOUR style" |
| "Are wide leg pants still in style?" | Specific trend check | ✅ Tavily search | ✅ + "you have wide leg pants in your wardrobe, here's how to style them now" |

### Profile-Building Queries (Quiz Results → Memory)

**SEO origin:** Color quiz (54K), body type quiz (56K), style quiz (53K) — these queries happen when users bring quiz results TO the agent

| User Query | What They Want | Cert | Demo Day |
|------------|---------------|------|----------|
| "I just found out I'm a Deep Autumn" | Store color season | ✅ Agent extracts fact, stores in memory, uses going forward | ✅ Same, but richer integration |
| "My body type is inverted triangle" | Store body shape | ✅ Stored in memory, influences outfit suggestions | ✅ + triggers personalized styling guide from RAG |
| "I took your style quiz and got Classic Minimalist" | Store style archetype | ✅ Stored in memory | ✅ + adjusts outfit generation to match aesthetic |
| "What's my color season?" (uploads selfie) | Color analysis via Vision API | ❌ Not in cert (no Vision integration) | ✅ Profile Builder agent analyzes, stores result |

---

## 3. Core Features

### Cert Challenge Features

#### Feature 1: Outfit Generation
- **SEO keywords served:** "outfit generator" (8K/mo), "ai outfit generator" (1.6K/mo)
- **Product page:** `/app/outfits` (authenticated), `/outfit-generator` (marketing)
- **What it does:** Takes user's wardrobe items + context (occasion, weather, preferences) → recommends outfit combination with reasoning
- **Agent tools:** `query_wardrobe`, `generate_outfit`
- **Memory dependency:** Reads user's style profile (color season, body shape, preferences) to personalize. Reads past feedback to avoid disliked patterns.

#### Feature 2: Style Knowledge Q&A (RAG)
- **SEO keywords served:** Knowledge behind "what colors look good on me" (33K/mo), "body shape calculator" (60K/mo), "how to find my style" (2.4K/mo)
- **Product page:** Powers responses across all `/app/*` pages, especially `/app/chat`
- **What it does:** Answers style education questions grounded in curated fashion knowledge base
- **Agent tools:** `search_style_knowledge` (RAG retrieval)
- **Memory dependency:** Personalizes answers based on known profile facts ("since you're a Deep Autumn...")
- **Evaluation:** RAGAS metrics — faithfulness, context precision, context recall

#### Feature 3: Wardrobe Intelligence
- **SEO keywords served:** "smart closet" (590/mo), "digital wardrobe" (110/mo)
- **Product page:** `/app/wardrobe`
- **What it does:** Queries, filters, and reasons about user's wardrobe items
- **Agent tools:** `query_wardrobe`
- **Memory dependency:** Uses stored preferences for smarter filtering

#### Feature 4: Trend Lookup
- **SEO keywords served:** General fashion trend queries
- **What it does:** Searches current fashion trends via Tavily
- **Agent tools:** `search_trends` (Tavily API — satisfies external API cert requirement)

#### Cross-cutting: Long-Term Memory
- **What it stores:** User profile facts (color season, body shape, style archetype), learned preferences ("loves earth tones"), outfit feedback patterns
- **How it evolves:** First interaction — agent knows nothing. Agent asks questions or user volunteers info. Each interaction may add/update observations. By 5th interaction, suggestions are noticeably personalized.
- **Product mapping:** This IS the `/app/style-profile` feature in embryonic form

### Demo Day Features (Everything Above, Plus:)

#### Feature 5: Color Analysis (if going wider)
- **SEO keywords served:** "color analysis quiz" (54K/mo), "seasonal color analysis" (2.9K/mo)
- **Product page:** `/tools/color-analysis-quiz`
- **What it does:** User uploads selfie → Vision API analyzes → determines color season → stores in memory
- **Shows the full flywheel:** SEO tool → profile data → memory → better agent suggestions

#### Feature 6: Style Profile Builder (if going deeper)
- **Product page:** `/app/style-profile`
- **What it does:** Aggregates everything the agent knows — color season, body shape, style archetype, wardrobe composition, preference patterns — into unified profile
- **Can be built through:** Conversation (agent asks targeted questions) OR quiz integrations

#### Enhanced: Multi-Day Outfit Planning
- **User query:** "What should I wear this week?"
- **Technical requirement:** Episodic memory (knows what was worn recently), occasion awareness (calendar context), no-repeat logic

---

## 4. Technical Architecture

### Cert Challenge Architecture

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

**Key components:**

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Agent framework | LangGraph ReAct agent | Tool-calling loop with reasoning |
| RAG corpus | Curated fashion .md files → chunked → embedded | Style education grounding |
| Vector store | FAISS or Chroma (local for cert) | RAG retrieval |
| Memory store | LangGraph InMemoryStore | User profile + observations |
| Wardrobe data | In-memory structured dicts | Preloaded sample wardrobe for cert |
| External API | Tavily | Trend search (cert requirement) |
| Evaluation | RAGAS | Faithfulness, context precision, context recall |
| Endpoint | FastAPI local endpoint | Cert requirement |

### Demo Day Architecture (Deeper Path)

```
User message
    │
    ▼
┌─────────────────────────────┐
│  MEMORY READ                │
│  Load full user profile:    │
│  - Semantic memory (facts)  │
│  - Episodic memory (recent) │
│  - Procedural (learned      │
│    rules about this user)   │
└─────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  SUPERVISOR AGENT (intent classifier + router)       │
│                                                      │
│  Classifies user intent:                             │
│  → "outfit_request" → route to Outfit Specialist     │
│  → "style_question" → route to Style Advisor         │
│  → "wardrobe_query" → route to Wardrobe Analyst      │
│  → "profile_update" → route to Profile Builder       │
└──────────────────────────────────────────────────────┘
    │
    ├─── outfit_request ───▶ ┌──────────────────────────┐
    │                        │  OUTFIT SPECIALIST        │
    │                        │  Tools:                   │
    │                        │  ├── query_wardrobe       │
    │                        │  ├── generate_outfit      │
    │                        │  └── search_trends        │
    │                        │                           │
    │                        │  Has access to:           │
    │                        │  - User's color season    │
    │                        │  - Body shape prefs       │
    │                        │  - Past outfit feedback   │
    │                        │  - Episodic memory        │
    │                        └──────────────────────────┘
    │
    ├─── style_question ───▶ ┌──────────────────────────┐
    │                        │  STYLE ADVISOR            │
    │                        │  Tools:                   │
    │                        │  ├── search_style_knowledge│ (RAG)
    │                        │  └── search_trends        │
    │                        │                           │
    │                        │  Personalizes answers     │
    │                        │  based on stored profile  │
    │                        └──────────────────────────┘
    │
    ├─── wardrobe_query ───▶ ┌──────────────────────────┐
    │                        │  WARDROBE ANALYST         │
    │                        │  Tools:                   │
    │                        │  ├── query_wardrobe       │
    │                        │  └── analyze_gaps         │
    │                        │                           │
    │                        │  Cross-references profile │
    │                        │  (color palette matching, │
    │                        │   capsule framework)      │
    │                        └──────────────────────────┘
    │
    └─── profile_update ───▶ ┌──────────────────────────┐
                             │  PROFILE BUILDER          │
                             │  Tools:                   │
                             │  ├── analyze_color_season │ (Vision API)
                             │  └── update_user_profile  │
                             │                           │
                             │  Conducts style assessment│
                             │  through conversation     │
                             │  or processes quiz results│
                             └──────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  MEMORY MANAGER             │
│  After every interaction:   │
│  - Extract new observations │
│  - Update semantic memory   │
│  - Summarize to episodic    │
│  - Refine procedural rules  │
└─────────────────────────────┘
```

**Memory evolution from cert → demo day:**

| Memory Type | Cert | Demo Day | Example |
|-------------|------|----------|---------|
| **Semantic** (facts & preferences) | ✅ InMemoryStore | ✅ Persistent store | "User is Deep Autumn, inverted triangle body shape" |
| **Episodic** (past interactions) | ❌ | ✅ Conversation summaries | "Last week user asked for wedding outfits, loved the navy blazer suggestion" |
| **Procedural** (learned rules) | ❌ | ✅ Self-improving instructions | "This user responds better when I explain the 'why' behind suggestions" |

**Data layer evolution:**

| Component | Cert | Demo Day |
|-----------|------|----------|
| Wardrobe data | In-memory dicts (preloaded sample) | Supabase Postgres (real upload pipeline with SAM3) |
| RAG vectors | FAISS/Chroma (local) | Qdrant Cloud or Supabase pgvector |
| Memory | LangGraph InMemoryStore | Supabase + LangGraph persistent store |
| User auth | None (single test user) | Better Auth via ShipAny |

---

## 5. The Data Flywheel

This is the competitive moat. The product has a natural learning loop:

```
Free SEO Tools (quizzes at /tools/*)
    → Generate user profile data (color season, body shape, style archetype)
        → Stored in agent long-term memory
            → Personalizes outfit suggestions & style advice
                → User keeps coming back (better experience each time)
                    → More interactions → richer memory
                        → Even better suggestions
```

**For cert:** We demonstrate the bottom half — memory → personalization → improvement over time. Profile data comes from conversation (user tells agent or agent asks).

**For demo day:** We can show the full loop — quiz tool generates profile data, data flows into memory, memory improves agent. This is the connection between the SEO traffic layer and the core product.

**Why this matters strategically:** Alta pushes daily notifications. Indyx uses expensive human stylists. Our edge is an agent that genuinely learns your style identity across every interaction and tool usage. The more you use it, the better it gets. This can't be copied by launching more features — it requires the memory architecture.

---

## 6. The "Deeper vs Wider" Choice (Demo Day)

### Deeper: Fewer features, more architectural sophistication

**What we show:**
- Multi-agent supervisor architecture (4 specialist agents)
- Three memory types (semantic + episodic + procedural)
- Demonstrable learning curve (first interaction vs fifth interaction)
- Style assessment through conversation (Profile Builder conducts mini-quiz via dialogue)
- Possibly: trip/event planner that chains multiple outfit suggestions

**Strengths:** Technically impressive for an AI engineering course. Shows multi-agent coordination, sophisticated memory, complex graph patterns. Maps to the product's long-term vision (multi-agent expansion to Interior, Wedding agents).

**Weaknesses:** Less product surface area to show. No visual "wow" of actual wardrobe photos or outfit cards.

### Wider: More features shown, broader product surface

**What we show:**
- Single agent from cert (enhanced but same architecture)
- Actual wardrobe upload pipeline (SAM3 segmentation → Vision classification → items appear)
- Visual outfit cards (item images rendered in outfit suggestion)
- Working Color Analysis Quiz demo (selfie → color season → stored in memory)
- Outfit rating/feedback UI that visibly updates memory

**Strengths:** Looks like a real product. Multiple working features you can demo. The SAM3 pipeline from the tech stack report is visually impressive.

**Weaknesses:** Architecturally simpler. Single agent + more tools isn't as impressive as multi-agent coordination. More frontend work, less AI engineering depth.

### Recommendation: Deeper

The AIE9 course evaluates AI engineering, not product design. A multi-agent system with sophisticated memory that demonstrably improves over interactions tells a stronger story than a broader product with simple agent patterns. The deeper path also directly maps to the Year 2+ vision (supervisor pattern naturally extends to Interior Agent, Wedding Agent).

However: if showing a compelling product demo matters for demo day (investors, potential users), wider might be the move. The choice depends on the audience.

---

## 7. RAG Knowledge Base Structure

The fashion corpus serves double duty: it powers the agent's style education AND it's the content backbone for future quiz tools.

### Corpus Organization

```
fashion_knowledge/
├── color_theory/
│   ├── seasonal_color_analysis.md      (12 seasons overview)
│   ├── 16_season_deep_dive.md          (16 season system)
│   ├── warm_seasons.md                 (Spring, Autumn details)
│   ├── cool_seasons.md                 (Summer, Winter details)
│   ├── undertone_guide.md              (warm/cool/neutral)
│   └── color_combinations.md           (complementary, analogous, etc.)
│
├── body_shapes/
│   ├── inverted_triangle.md            (9.9K/mo blog keyword)
│   ├── rectangle.md                    (8.1K/mo)
│   ├── hourglass.md                    (3.6K/mo)
│   ├── triangle_pear.md               (2.9K/mo)
│   ├── petite.md                       (1.9K/mo)
│   └── styling_principles.md           (general body shape dressing rules)
│
├── style_archetypes/
│   ├── classic.md
│   ├── romantic.md
│   ├── dramatic.md
│   ├── natural.md
│   ├── creative.md
│   └── aesthetic_guide.md              (Dark Academia, Coastal Grandmother, etc.)
│
├── occasion_dressing/
│   ├── business_casual.md              (91K/mo blog keyword cluster)
│   ├── smart_casual.md
│   ├── formal_events.md
│   ├── date_night.md                   (60K/mo blog keyword cluster)
│   └── casual_everyday.md
│
├── wardrobe_building/
│   ├── capsule_wardrobe.md             (880/mo tool keyword)
│   ├── essential_pieces.md
│   ├── mix_and_match.md
│   └── seasonal_rotation.md
│
└── fundamentals/
    ├── fabric_guide.md
    ├── fit_principles.md
    ├── proportion_and_balance.md
    └── color_matching_clothes.md       (1.3K/mo tool keyword)
```

### Chunk Strategy

Each document is chunked by section (H2 headers), with metadata tags for:
- Topic (color_theory, body_shape, style_archetype, occasion, wardrobe)
- Specificity (general_principle, specific_type, how_to)
- Related keywords (for connecting RAG results to SEO intent)

---

## 8. Cert Challenge Requirements Checklist

| Requirement | How We Satisfy It |
|-------------|------------------|
| **RAG** | Fashion knowledge base → chunked → embedded → retrieved for style education queries |
| **Agent** | LangGraph ReAct agent with 4 tools + memory read/write layer |
| **RAGAS evaluation** | Evaluate style education queries: faithfulness, context precision, context recall |
| **External API** | Tavily for trend searches |
| **Own data uploaded** | Curated fashion knowledge corpus (markdown files on color theory, body shapes, style archetypes, etc.) |
| **Local endpoint** | FastAPI serving the agent |
| **5-min Loom** | Demo: show agent learning user's style profile over multiple interactions, improving suggestions |

---

## 9. Build Sequence

### Cert (Weeks 1-3)

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1 | RAG + Basic Agent | Fashion knowledge corpus curated, chunked, embedded. Basic LangGraph ReAct agent with `search_style_knowledge` tool. FastAPI endpoint. |
| 2 | Wardrobe + Outfit Tools | Sample wardrobe data. `query_wardrobe` and `generate_outfit` tools. Tavily `search_trends` tool. Agent can answer all 4 query types. |
| 3 | Memory + Evaluation | InMemoryStore for user profile/observations. Memory read before agent, write after. RAGAS evaluation on style education queries. Polish + Loom recording. |

### Demo Day (Weeks 4-6)

| Week | Focus | Deliverables |
|------|-------|-------------|
| 4 | Multi-Agent Refactor | Supervisor + 4 specialist agents. Intent classification + routing. Same tools, now distributed across specialists. |
| 5 | Memory Evolution | Episodic memory (conversation summaries). Procedural memory (learned rules). Memory Manager after each interaction. |
| 6 | Polish + Demo | End-to-end demo showing: first interaction (agent knows nothing) → style assessment → 3-4 more interactions → noticeably personalized suggestions. Demo narrative for 5-min presentation. |

---

*Companion file: `stylists_ai_build_plan.html` — interactive visual mapping of SEO → Queries → Features → Architecture*
