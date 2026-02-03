"""
Test structured outputs with Cosmos LLM API.

This simulates what the stock agents do with output_type parameter.
"""
from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel

from agents import Agent, Runner
from llm_config import configure_agents_sdk

load_dotenv()


class TestOutput(BaseModel):
    """Simple structured output for testing."""
    message: str
    success: bool


async def test_structured_output():
    """Test that structured outputs work with Cosmos."""
    print("=" * 60)
    print("Testing Structured Output with Cosmos")
    print("=" * 60)

    # Configure SDK
    configure_agents_sdk()

    # Create agent with structured output (like your stock agents)
    test_agent = Agent(
        name="TestAgent",
        instructions="You are a test assistant. Return a JSON with message and success fields.",
        output_type=TestOutput,
        model="gpt-4.1-mini"
    )

    print("\nTesting agent with structured output...")
    try:
        result = await Runner.run(
            test_agent,
            "Say hello and mark this as successful"
        )

        output: TestOutput = result.final_output
        print(f"\n✓ Success! Structured output received:")
        print(f"  Message: {output.message}")
        print(f"  Success: {output.success}")
        print("\nStructured outputs are working with Cosmos!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nStructured outputs may not be fully compatible.")
        print("You may need to remove output_type from agents or use a different approach.")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_structured_output())
