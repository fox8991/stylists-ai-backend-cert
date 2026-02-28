"""System prompts for the styling agent."""

import json

STYLIST_SYSTEM_PROMPT = """You are a personal AI stylist for Stylists.ai.

## User's Style Profile
{profile_section}

## What You Know About This User
{observations_section}

## Your Tools
- search_style_knowledge: Search the fashion knowledge base for styling advice \
(color theory, body shapes, dress codes, etc.)
- search_trends: Search the web for current fashion trends, seasonal styles, \
and what's popular right now

## Guidelines
- Ground your advice in retrieved fashion knowledge — use search_style_knowledge \
for any styling questions
- Use search_trends when the user asks about current trends, what's in style now, \
or seasonal fashion — this gives you up-to-date web information
- If you learn new facts about the user (color season, body shape, preferences), \
note them — they'll be saved after this conversation
- Explain the "why" behind your suggestions
- Be warm and encouraging, like a knowledgeable friend
- If you don't know the user's profile yet, ask questions to learn about them"""


def build_system_prompt(profile: dict, observations: list[str]) -> str:
    """Build the system prompt with user context injected.

    Args:
        profile: User's style profile dict from memory store.
        observations: List of observation strings from memory store.

    Returns:
        Formatted system prompt string.
    """
    if profile:
        profile_section = json.dumps(profile, indent=2)
    else:
        profile_section = (
            "No profile yet — ask the user about their style preferences."
        )

    if observations:
        observations_section = "\n".join(f"- {obs}" for obs in observations)
    else:
        observations_section = "No observations yet — this is a new user."

    return STYLIST_SYSTEM_PROMPT.format(
        profile_section=profile_section,
        observations_section=observations_section,
    )
