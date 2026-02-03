"""
LLM Configuration for OpenAI Agents SDK with Cosmos Compatibility.

This module configures the OpenAI Agents SDK to work with Cosmos LLM API.

Key Configuration:
- Uses chat_completions API (not responses API)
- Disables tracing (Cosmos doesn't support it)
- NOTE: Structured outputs (output_type) may not work with current Cosmos version
"""
from __future__ import annotations

import os
from openai import OpenAI, AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled, set_default_openai_api


def get_openai_client() -> OpenAI:
    """
    Get configured OpenAI client.

    If LLM_API_KEY and LLM_BASE_URL are set, uses Cosmos LLM API.
    Otherwise, uses standard OpenAI API with OPENAI_API_KEY.
    """
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")

    if llm_api_key and llm_base_url:
        print(f"Using Cosmos LLM API at {llm_base_url}")
        return OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url
        )
    else:
        print("Using OpenAI API")
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_async_openai_client() -> AsyncOpenAI:
    """
    Get configured async OpenAI client.

    If LLM_API_KEY and LLM_BASE_URL are set, uses Cosmos LLM API.
    Otherwise, uses standard OpenAI API with OPENAI_API_KEY.
    """
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")

    if llm_api_key and llm_base_url:
        print(f"Using Cosmos LLM API at {llm_base_url}")
        return AsyncOpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url
        )
    else:
        print("Using OpenAI API")
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def configure_agents_sdk():
    """
    Configure the OpenAI Agents SDK to work with Cosmos LLM API.

    Key steps:
    1. Set API to use chat_completions (not responses)
    2. Configure custom client with Cosmos endpoint
    3. Disable tracing (Cosmos doesn't support it)

    Returns:
        bool: True if configured for Cosmos, False if using OpenAI
    """
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")

    if llm_api_key and llm_base_url:
        # Step 1: Force chat_completions API (Cosmos doesn't support responses API)
        set_default_openai_api("chat_completions")

        # Step 2: Create AsyncOpenAI client pointing to Cosmos
        client = AsyncOpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url
        )

        # Step 3: Set as default client (use_for_tracing=False to avoid trace upload)
        set_default_openai_client(client, use_for_tracing=False)

        # Step 4: Disable tracing completely
        set_tracing_disabled(True)

        print(f"✓ Configured Agents SDK for Cosmos LLM API")
        print(f"  Endpoint: {llm_base_url}")
        print(f"  API Mode: chat_completions")
        print(f"  Tracing: disabled")
        return True
    else:
        print("✓ Using default OpenAI API configuration")
        return False
