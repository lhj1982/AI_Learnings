from __future__ import annotations

from pydantic import BaseModel
from agents import Agent


class VerificationResult(BaseModel):
    verified: bool
    issues: str


verifier_agent = Agent(
    name="VerificationAgent",
    instructions=(
        "You are a senior reviewer. Verify the research note is consistent with the provided context."
        "Flag: missing support for key claims, contradictions, or illogical recommendation."
        "Return {verified, issues}."
    ),
    output_type=VerificationResult,
    model="gpt-4.1-mini",
)
