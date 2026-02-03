from __future__ import annotations

from pydantic import BaseModel
from agents import Agent


class SearchItem(BaseModel):
    query: str
    reason: str


class SearchPlan(BaseModel):
    searches: list[SearchItem]


planner_agent = Agent(
    name="PlannerAgent",
    instructions=(
        "Create a focused web-search plan to gather the latest fiscal information needed for "
        "an equity research note (recent quarter results, key metrics, valuation, technicals, macro/industry)."
        "Return 4-8 queries. Prefer authoritative sources (filings, IR pages, reputable finance sites)."
    ),
    output_type=SearchPlan,
    model="gpt-4.1-mini",
)
