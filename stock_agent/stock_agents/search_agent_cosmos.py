"""
Cosmos-compatible version of the search agent.

NOTE: WebSearchTool is NOT compatible with chat_completions API.
This version simulates web search results or instructs the agent to provide
analysis based on its training data.
"""
from __future__ import annotations

from agents import Agent, ModelSettings


# Option 1: Agent that simulates having done a search
search_agent_cosmos = Agent(
    name="SearchAgent",
    instructions=(
        "You are a financial research assistant. For the given search query, "
        "provide a concise summary of what recent, relevant fiscal/earnings/valuation/technical "
        "information would typically be found for this search.\n\n"
        "Use your knowledge up to your training cutoff date. Include:\n"
        "- Relevant financial metrics and trends\n"
        "- Key dates and numbers when applicable\n"
        "- Indicate uncertainty by saying 'As of [date]' or 'Historically'\n"
        "- If you don't have specific recent data, say 'Not found in sources' for that metric\n\n"
        "Format your response as a concise summary paragraph with dates and numbers."
    ),
    # NO tools - WebSearchTool doesn't work with chat_completions API
    model="gpt-5",
    # Note: gpt-5-mini only supports temperature=1 (default)
)


# Note: For production use with real-time data, you would need to:
# 1. Implement a custom tool that calls an external search API
# 2. Or use a different web search service that's compatible
# 3. Or use the responses API (but that requires OpenAI, not Cosmos)
