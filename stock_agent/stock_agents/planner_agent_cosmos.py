"""
Cosmos-compatible version of the planner agent.

This version does NOT use output_type (structured outputs).
"""
from __future__ import annotations

from pydantic import BaseModel
from agents import Agent

from cosmos_agent_helpers import create_json_instructions


class SearchItem(BaseModel):
    query: str
    reason: str


class SearchPlan(BaseModel):
    searches: list[SearchItem]


PLANNER_INSTRUCTIONS = create_json_instructions(
    base_instructions="""Create a focused web-search plan to gather the latest fiscal information needed for
an equity research note (recent quarter results, key metrics, valuation, technicals, macro/industry).
Return 4-8 queries. Prefer authoritative sources (filings, IR pages, reputable finance sites).""",
    model_class=SearchPlan
)

planner_agent_cosmos = Agent(
    name="PlannerAgent",
    instructions=PLANNER_INSTRUCTIONS,
    # NO output_type - Cosmos doesn't support it
    model="gpt-5",
)
