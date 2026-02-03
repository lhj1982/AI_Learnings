"""
Cosmos-compatible version of the guardrail agent.

This version does NOT use output_type (structured outputs), which are incompatible
with Cosmos LLM API. Instead, it uses JSON mode and manual parsing.
"""
from __future__ import annotations

from pydantic import BaseModel
from agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput

from cosmos_agent_helpers import parse_agent_json_output, create_json_instructions


class GuardrailResult(BaseModel):
    is_relevant: bool
    tickers: list[str] = []
    reason: str = ""


# Create instructions that request JSON output
GUARDRAIL_INSTRUCTIONS = create_json_instructions(
    base_instructions="""Determine whether the user is requesting stock/equity analysis.
If yes, extract a list of public stock tickers/codes, if company identifiers mentioned.
If the user provides a company name without a ticker, convert it to a ticker symbol, 
if no ticker symbol is found, use company name as-is.""",
    model_class=GuardrailResult
)

_guardrail_agent = Agent(
    name="StockInputGuardrail",
    instructions=GUARDRAIL_INSTRUCTIONS,
    # NOTE: No output_type - Cosmos doesn't support structured outputs
    model="gpt-5",
)


@input_guardrail
async def stock_input_guardrail_cosmos(ctx, agent, user_input):
    """Cosmos-compatible version using manual JSON parsing."""
    res = await Runner.run(_guardrail_agent, user_input)
    # Parse the text response as JSON
    try:
        out: GuardrailResult = parse_agent_json_output(
            res.final_output,  # This is a string, not a structured object
            GuardrailResult
        )
    except Exception as e:
        # If parsing fails, reject the input
        print(f"Warning: Guardrail JSON parsing failed: {e}")
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info=GuardrailResult(is_relevant=False, reason="Parse error")
        )

    return GuardrailFunctionOutput(
        tripwire_triggered=not out.is_relevant,
        output_info=out,
    )
