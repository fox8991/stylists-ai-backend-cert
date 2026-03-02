"""Wardrobe query tool for filtering user's clothing items."""

import json

from langchain_core.tools import tool

from app.data.wardrobe import SAMPLE_WARDROBE


def query_wardrobe_func(
    category: str | None = None,
    color: str | None = None,
    occasion: str | None = None,
    season: str | None = None,
    formality: str | None = None,
) -> str:
    """Filter wardrobe items by attributes. Callable for testing.

    Args:
        category: Filter by category (tops, bottoms, outerwear, shoes,
                  accessories).
        color: Filter by primary color name (e.g., "olive", "burgundy").
        occasion: Filter by occasion tag (everyday, wfh, office, date_night,
                  weekend, active, travel, smart_casual).
        season: Filter by season (spring, summer, fall, winter).
        formality: Filter by formality level (casual, smart_casual,
                   business_casual, formal).

    Returns:
        JSON string of matching wardrobe items, or a message if nothing matches.
    """
    items = SAMPLE_WARDROBE

    if category:
        items = [i for i in items if i["category"] == category.lower()]
    if color:
        items = [i for i in items if color.lower() in i["color"]["primary"].lower()]
    if occasion:
        items = [i for i in items if occasion.lower() in i.get("occasions", [])]
    if season:
        items = [i for i in items if season.lower() in i.get("seasons", [])]
    if formality:
        items = [i for i in items if i.get("formality") == formality.lower()]

    if not items:
        return "No wardrobe items match those filters."

    # Return a concise view — exclude production-only null fields
    display_fields = [
        "id", "name", "category", "sub_category", "color", "pattern",
        "fabric", "fit", "style_tags", "formality", "seasons", "occasions",
    ]
    result = [{k: item[k] for k in display_fields if k in item} for item in items]
    return json.dumps(result, indent=2)


@tool
async def query_wardrobe(
    category: str | None = None,
    color: str | None = None,
    occasion: str | None = None,
    season: str | None = None,
    formality: str | None = None,
) -> str:
    """Query the user's wardrobe items with optional filters.

    Use this tool to find specific clothing items in the user's wardrobe.
    You can combine multiple filters to narrow results. Call with no filters
    to see the full wardrobe.

    Args:
        category: Filter by category (tops, bottoms, outerwear, shoes,
                  accessories).
        color: Filter by primary color name (e.g., "olive", "burgundy").
        occasion: Filter by occasion tag (everyday, wfh, office, date_night,
                  weekend, active, travel, smart_casual).
        season: Filter by season (spring, summer, fall, winter).
        formality: Filter by formality level (casual, smart_casual,
                   business_casual, formal).

    Returns:
        JSON list of matching wardrobe items with their attributes.
    """
    return query_wardrobe_func(category, color, occasion, season, formality)
