import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test():
    client = AsyncOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )

    # Test with json_schema (what the SDK uses)
    print("Testing with json_schema format...")
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Return JSON with message and success fields"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "success": {"type": "boolean"}
                        },
                        "required": ["message", "success"]
                    }
                }
            }
        )
        print("✓ json_schema works!")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"✗ json_schema failed: {e}")

    # Test with json_object (simpler format)
    print("\nTesting with json_object format...")
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Return JSON with message and success fields"}],
            response_format={"type": "json_object"}
        )
        print("✓ json_object works!")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"✗ json_object failed: {e}")

asyncio.run(test())
