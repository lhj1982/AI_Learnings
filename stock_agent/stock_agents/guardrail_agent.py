from __future__ import annotations

from pydantic import BaseModel
from agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput


class GuardrailResult(BaseModel):
    is_relevant: bool
    tickers: list[str] = []
    reason: str = ""


_guardrail_agent = Agent(
    name="StockInputGuardrail",
    instructions=(
        "Determine whether the user is requesting stock/equity analysis."
        "If yes, extract a list of public stock tickers/codes or company identifiers mentioned."
        "Return JSON: {is_relevant, tickers, reason}."
        "If the user provides a company name without a ticker, include the name as-is in tickers."
    ),
    output_type=GuardrailResult,
    model="gpt-4.1-mini",
)


@input_guardrail
async def stock_input_guardrail(ctx, agent, user_input):
    res = await Runner.run(_guardrail_agent, user_input)
    out: GuardrailResult = res.final_output
    return GuardrailFunctionOutput(
        tripwire_triggered=not out.is_relevant,
        output_info=out,
    )
