"""Test the search agent directly."""
import asyncio
from dotenv import load_dotenv
from agents import Runner
from llm_config import configure_agents_sdk
from stock_agents.search_agent_cosmos import search_agent_cosmos

load_dotenv()
configure_agents_sdk()

async def test():
    try:
        print("Testing search agent...")
        result = await Runner.run(
            search_agent_cosmos,
            "Query: Apple Q4 2025 earnings\nReason: Get latest financial results"
        )
        print(f"\n✓ Success!")
        print(f"Response: {result.final_output}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test())
