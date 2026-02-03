from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from manager import StockResearchManager
from printer import Printer
from llm_config import configure_agents_sdk

load_dotenv()

# Configure OpenAI Agents SDK to use Cosmos LLM API
configure_agents_sdk()


async def main():
    mgr = StockResearchManager(
        pdf_root=os.getenv("PDF_ROOT", "./pdfs"),
        printer=Printer(verbose=True),
    )

    while True:
        q = input("Enter stock query / tickers (or 'q' to quit): ").strip()
        if q.lower() == "q":
            break
        await mgr.run(q)


if __name__ == "__main__":
    asyncio.run(main())
