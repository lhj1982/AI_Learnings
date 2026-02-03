"""
Test script to verify LLM configuration (OpenAI or Cosmos).

Run this to test if your LLM API is configured correctly before running the main app.
"""
from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from llm_config import configure_agents_sdk, get_async_openai_client

load_dotenv()


async def test_llm_connection():
    """Test basic LLM API connection."""
    print("=" * 60)
    print("Testing LLM Configuration")
    print("=" * 60)

    # Configure SDK
    is_cosmos = configure_agents_sdk()

    # Get client
    client = get_async_openai_client()

    # Test basic completion
    print("\nTesting chat completion...")
    try:
        response = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello! LLM is working correctly.' in one sentence."}
            ],
            max_tokens=50
        )

        print("\n✓ Success! Response:")
        print(response.choices[0].message.content)
        print("\nConfiguration is working correctly!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check your API key and base URL configuration in .env")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_llm_connection())
