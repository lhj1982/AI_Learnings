"""
Test Cosmos-compatible agent (without structured outputs).
"""
from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel

from agents import Agent, Runner
from llm_config import configure_agents_sdk
from cosmos_agent_helpers import parse_agent_json_output, create_json_instructions

load_dotenv()


class TestOutput(BaseModel):
    """Simple structured output for testing."""
    message: str
    success: bool


async def test_cosmos_agent():
    """Test agent that returns JSON without using output_type."""
    print("=" * 60)
    print("Testing Cosmos-Compatible Agent (No Structured Outputs)")
    print("=" * 60)

    # Configure SDK
    configure_agents_sdk()

    # Create agent WITHOUT output_type
    instructions = create_json_instructions(
        base_instructions="You are a test assistant.",
        model_class=TestOutput
    )

    test_agent = Agent(
        name="TestAgent",
        instructions=instructions,
        # NO output_type - this is the key difference!
        model="gpt-4.1-mini"
    )

    print("\nTesting agent with JSON mode (no structured outputs)...")
    try:
        result = await Runner.run(
            test_agent,
            "Say hello and mark this as successful"
        )

        # Debug: see what we got
        print(f"\nResult type: {type(result.final_output)}")
        print(f"Result value: {repr(result.final_output)}")
        print(f"Result dir: {[x for x in dir(result.final_output) if not x.startswith('_')]}")

        # Parse the text response
        output: TestOutput = parse_agent_json_output(
            result.final_output,  # This is plain text/JSON string
            TestOutput
        )

        print(f"\n✓ Success! JSON parsed correctly:")
        print(f"  Message: {output.message}")
        print(f"  Success: {output.success}")
        print("\nThis approach should work with Cosmos!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_cosmos_agent())
