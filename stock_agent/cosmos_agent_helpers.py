"""
Helper utilities for using OpenAI Agents SDK with Cosmos LLM API.

Since Cosmos doesn't support structured outputs (output_type), these helpers
allow agents to return JSON that we parse manually.
"""
from __future__ import annotations

import json
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


def parse_agent_json_output(response_text: str, model_class: Type[T]) -> T:
    """
    Parse agent response text as JSON and validate with Pydantic model.

    Handles responses wrapped in markdown code blocks like:
    ```json
    {...}
    ```

    Args:
        response_text: The agent's text response (should be JSON)
        model_class: Pydantic model class to parse into

    Returns:
        Parsed and validated model instance

    Raises:
        json.JSONDecodeError: If response is not valid JSON
        pydantic.ValidationError: If JSON doesn't match model schema
    """
    # Strip markdown code blocks if present
    text = response_text.strip()

    # Remove ```json ... ``` or ``` ... ``` wrapping
    if text.startswith('```'):
        # Find the first newline (end of opening ```)
        first_newline = text.find('\n')
        # Find the last ``` (closing)
        last_backticks = text.rfind('```')

        if first_newline != -1 and last_backticks != -1:
            text = text[first_newline + 1:last_backticks].strip()

    # Parse JSON
    data = json.loads(text)

    # Validate with Pydantic model
    return model_class.model_validate(data)


def create_json_instructions(base_instructions: str, model_class: Type[BaseModel]) -> str:
    """
    Create instructions that tell the agent to return JSON matching a schema.

    Args:
        base_instructions: The base instructions for the agent
        model_class: Pydantic model defining the expected output

    Returns:
        Enhanced instructions with JSON schema requirements
    """
    schema = model_class.model_json_schema()

    instructions = f"""{base_instructions}

IMPORTANT: You must return ONLY valid JSON matching this exact schema:

```json
{json.dumps(schema, indent=2)}
```

Return ONLY the JSON object, with no additional text before or after.
"""
    return instructions
