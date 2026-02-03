from __future__ import annotations

from agents import Agent, WebSearchTool, ModelSettings

search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "Use web search to find the most recent, relevant fiscal/earnings/valuation/technical info."
        "Return a concise summary with dates and numbers. Include source hints in-text (site names)."
        "If you cannot find a metric, say 'Not found in sources'."
    ),
    tools=[WebSearchTool()],
    model_settings=ModelSettings(
        model="gpt-4.1-mini",
        temperature=0.2,
        tool_choice="required",
    ),
)
