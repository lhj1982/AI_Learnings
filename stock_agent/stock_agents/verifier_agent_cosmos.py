"""
Cosmos-compatible version of the verifier agent.

This version does NOT use output_type (structured outputs).
"""
from __future__ import annotations

from pydantic import BaseModel
from agents import Agent

from cosmos_agent_helpers import create_json_instructions


class VerificationResult(BaseModel):
    verified: bool
    issues: str


VERIFIER_INSTRUCTIONS = create_json_instructions(
    base_instructions="""You are a senior reviewer. Verify the research note is consistent with the provided context.
Flag: missing support for key claims, contradictions, or illogical recommendation.""",
    model_class=VerificationResult
)

verifier_agent_cosmos = Agent(
    name="VerificationAgent",
    instructions=VERIFIER_INSTRUCTIONS,
    # NO output_type - Cosmos doesn't support it
    model="gpt-5",
)
