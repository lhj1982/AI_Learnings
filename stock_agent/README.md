# Stock Research Agent (OpenAI Agents SDK + Postgres + optional local PDF RAG)

**🌟 Now supports Cosmos LLM API!** See [COSMOS_README.md](COSMOS_README.md) for details.

This project implements a multi-agent stock analysis pipeline inspired by OpenAI's `financial_research_agent` example:
- **Guardrail**: blocks irrelevant input and extracts tickers/entities.
- **Planner**: generates focused web queries for fiscal/valuation/technical info.
- **Search agent**: uses `WebSearchTool` to gather public info.
- **Optional local PDF context**: PDFs under `./pdfs/<ENTITY_ID>/` are ingested into Postgres (`source_documents` + chunks) and retrieved (RAG-lite) as extra context.
- **Analyst**: produces a structured research note (JSON fields).
- **Verifier**: checks consistency with evidence.
- **Persistence**: saves one row per day per entity into Postgres.
- **Logging + tracing**: Python logging + Agents SDK tracing spans.

## Quick start

1) Create a Postgres database (e.g. `stocks`) and run schema:

```bash
psql -d stocks -f migrations/schema.sql
```

2) Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Set environment variables (or create `.env` from `.env.example`):

**Option A: Use OpenAI API (default)**
- `OPENAI_API_KEY`

**Option B: Use Cosmos LLM API (OpenAI-compatible)**
- `LLM_API_KEY` - Your Cosmos API key
- `LLM_BASE_URL` - Your Cosmos endpoint (e.g., `https://your-cosmos-endpoint.com/v1`)

**Database & PDF settings:**
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `PDF_ROOT` (default: `./pdfs`)

**Test your LLM configuration:**
```bash
python test_llm_config.py
```

4) (Optional) Put PDFs in `./pdfs/AAPL/*.pdf`, `./pdfs/MSFT/*.pdf`, etc.

5) Run:

```bash
python main.py
```

Type a query like:
- `AAPL`
- `Analyze AAPL and MSFT`
- `Please analyze TSLA`

The app will:
- ingest any new PDFs for each extracted ticker
- web-search fiscal info
- generate + verify a structured research note
- upsert to Postgres (once per day per entity)

## Notes
- This starter uses **keyword scoring** for PDF retrieval (simple and free). You can later add embeddings (pgvector) without changing the orchestration.
- `doc_date` for PDFs is inferred from filename if possible (e.g. `2025-11-07_report.pdf`, `Q3_2025.pdf`), otherwise uses file modified date.
