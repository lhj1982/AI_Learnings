from __future__ import annotations

import os
from datetime import date
import pprint

from agents import Agent, Runner, trace
from agents.exceptions import InputGuardrailTripwireTriggered

from printer import Printer
from db import get_conn, fetch_existing_daily, upsert_daily_analysis
from pdf_ingest import ingest_pdfs_for_entity
from pdf_retrieve import retrieve_pdf_chunks

from cosmos_agent_helpers import parse_agent_json_output

# Conditionally import agents based on API configuration
llm_api_key = os.getenv("LLM_API_KEY")
llm_base_url = os.getenv("LLM_BASE_URL")
USE_COSMOS = bool(llm_api_key and llm_base_url)

if USE_COSMOS:
    print("Loading Cosmos-compatible agents...")
    from stock_agents.guardrail_agent_cosmos import stock_input_guardrail_cosmos as stock_input_guardrail, GuardrailResult
    from stock_agents.planner_agent_cosmos import planner_agent_cosmos as planner_agent, SearchPlan
    from stock_agents.search_agent_cosmos import search_agent_cosmos as search_agent
    from stock_agents.analyst_agent_cosmos import analyst_agent_cosmos as analyst_agent, StockNote
    from stock_agents.verifier_agent_cosmos import verifier_agent_cosmos as verifier_agent, VerificationResult
else:
    print("Loading OpenAI-compatible agents...")
    from stock_agents.guardrail_agent import stock_input_guardrail, GuardrailResult
    from stock_agents.planner_agent import planner_agent, SearchPlan
    from stock_agents.search_agent import search_agent
    from stock_agents.analyst_agent import analyst_agent, StockNote
    from stock_agents.verifier_agent import verifier_agent, VerificationResult


front_agent = Agent(
    name="FrontAgent",
    instructions=(
        "You are the entrypoint for a stock research pipeline. "
        "Normalize the user request if needed. Keep the user's intent."
    ),
    input_guardrails=[stock_input_guardrail],
    model="gpt-5" if USE_COSMOS else "gpt-4.1",
)


class StockResearchManager:
    def __init__(self, pdf_root: str | None = None, printer: Printer | None = None):
        self.pdf_root = pdf_root or os.getenv("PDF_ROOT", "./pdfs")
        self.printer = printer or Printer(verbose=True)
        self.conn = get_conn()

    async def run(self, user_query: str):
        today = date.today().isoformat()
        self.printer.heading(f"Stock research run | {today}")

        # Guardrail pass + extraction: run FrontAgent (will tripwire if irrelevant)
        try:
            with trace("front"):
                front_res = await Runner.run(front_agent, user_query)
        except InputGuardrailTripwireTriggered as e:
            # The SDK raises on tripwire. We show a friendly message.
            self.printer.warn("Input blocked by guardrail (not a stock analysis request).")
            return

        # Extract tickers from guardrail output_info
        # Different SDK versions expose guardrail results slightly differently;
        # we handle both common shapes.
        tickers = []
        print("Has input_guardrail_results?", hasattr(front_res, "input_guardrail_results"))

        if getattr(front_res, "input_guardrail_results", None):
            info = front_res.input_guardrail_results[0].output.output_info
            print("Guardrail output_info:", info)
            print("Type of output_info:", type(info))

            if isinstance(info, dict):
                tickers = info.get("tickers", [])
            else:
                # pydantic object
                tickers = getattr(info, "tickers", []) or []

            print("Extracted tickers:", tickers)

        if not tickers:
            # fallback: if no extraction, treat the entire input as an entity
            tickers = [user_query.strip()]
            print("Using fallback tickers:", tickers)

        for entity_id in tickers:
            await self._run_for_entity(entity_id.strip(), today, user_query)

    async def _run_for_entity(self, entity_id: str, today: str, user_query: str):
        if not entity_id:
            return

        self.printer.step(f"Processing {entity_id} ({today})")

        # Daily cache
        existing = fetch_existing_daily(self.conn, today, entity_id)
        if existing:
            self.printer.step(f"Found existing DB entry for {entity_id} today; skipping.")
            return

        # Ingest local PDFs incrementally
        new_docs = ingest_pdfs_for_entity(self.conn, self.pdf_root, entity_id, entity_type="public")
        if new_docs:
            self.printer.step(f"Ingested {new_docs} new PDF(s) for {entity_id} into source tables.")

        # Plan searches
        with trace("plan"):
            plan_res = await Runner.run(planner_agent, f"{user_query}\nEntity: {entity_id}\nDate: {today}")
            # Parse response (manually for Cosmos, structured for OpenAI)
            if USE_COSMOS:
                plan: SearchPlan = parse_agent_json_output(plan_res.final_output, SearchPlan)
            else:
                plan: SearchPlan = plan_res.final_output

        # Run searches
        summaries: list[str] = []
        for item in plan.searches:
            with trace("search"):
                print(item.query, ", reason: ", item.reason)
                sres = await Runner.run(search_agent, f"Query: {item.query}\nReason: {item.reason}")
                summaries.append(sres.final_output)

        # Optional PDF RAG context from DB (prefers newest docs)
        pdf_chunks = retrieve_pdf_chunks(self.conn, entity_id, user_query, max_chunks=6, newest_docs=3)

        # Format analyst input as a prompt string
        analyst_input = f"""Date: {today}
Entity: {entity_id}
User Query: {user_query}

Web Search Summaries:
{chr(10).join(f"- {s}" for s in summaries)}

PDF Context:
{pdf_chunks if pdf_chunks else "No PDF context available"}
"""

        print(analyst_input)
        # Draft analysis
        with trace("analyze"):
            ares = await Runner.run(analyst_agent, analyst_input)
            # Parse response (manually for Cosmos, structured for OpenAI)
            if USE_COSMOS:
                note: StockNote = parse_agent_json_output(ares.final_output, StockNote)
            else:
                note: StockNote = ares.final_output

        # Verify
        verify_input = f"""Date: {today}
Entity: {entity_id}

Stock Note to Verify:
{pprint.pformat(note.model_dump(), indent=2)}

Web Search Summaries:
{chr(10).join(f"- {s}" for s in summaries)}

PDF Context:
{pdf_chunks if pdf_chunks else "No PDF context available"}
"""
        with trace("verify"):
            vres = await Runner.run(verifier_agent, verify_input)
            # Parse response (manually for Cosmos, structured for OpenAI)
            if USE_COSMOS:
                verdict: VerificationResult = parse_agent_json_output(vres.final_output, VerificationResult)
            else:
                verdict: VerificationResult = vres.final_output

        # Save
        raw_context = {
            "web_summaries": summaries,
            "pdf_context": pdf_chunks,
            "verifier": verdict.model_dump(),
        }
        upsert_daily_analysis(
            self.conn,
            today,
            entity_id,
            "public",
            note.model_dump(),
            raw_context,
        )

        if verdict.verified:
            self.printer.step(f"Saved analysis for {entity_id} ✓")
        else:
            self.printer.warn(f"Saved analysis for {entity_id} with verifier issues: {verdict.issues}")
