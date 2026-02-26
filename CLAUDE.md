# Stylists.ai Backend

## Project Context

AI-powered personal styling platform, built for the AIE9 cert challenge.

Read these docs before making architectural decisions:
- `docs/PRD.md` — product requirements, architecture, schemas, tools, memory system
- `docs/build_plan.md` — full feature map (SEO → queries → features → architecture)
- `docs/tech_architecture.md` — detailed implementation specs with pseudocode

These docs represent our *current* design direction. If something needs to change based on what you learn during implementation, flag it — don't silently deviate, but don't blindly follow either.

## Coding Principles

- Type hints everywhere
- Pydantic models for all request/response schemas
- Async endpoints in FastAPI
- Keep tool functions pure — each tool does one thing
- Docstrings on all public functions (Google style)
- Environment variables via `.env` — no hardcoded API keys, ever
- Keep dependencies minimal — don't install packages we don't need

## Workflow

- Run `pytest` before considering any feature complete
- When building the RAG pipeline, run a quick retrieval sanity check before moving on
- When adding a new tool, register it in the graph AND add a basic test
- Keep the FastAPI endpoint thin — all logic lives in the graph nodes
- Commit working increments, not half-built features
- Don't skip RAGAS eval — it's a cert requirement
