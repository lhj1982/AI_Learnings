import logging

from typing import Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    filter_messages
)

# System level instructions
INITIAL_INSTRUCTION = """
You are a professional equity research analyst. Analyze the stock by a given stock name or code based on its latest quarterly earnings, valuation metrics (P/E, P/S, EV/EBITDA), competitive positioning, and current macroeconomic trends. Include the following in your report:

1. Company Overview – brief business summary and key revenue streams

2. Recent Financial Performance – revenue, EPS, margins (QoQ and YoY)

3. Valuation Comparison 

4. Technical Analysis – key support/resistance levels, moving averages

5. Catalysts & Risks – what might drive the stock price up or down in the next 6–12 months

6. Buy/Hold/Sell Recommendation – with a price target and rationale

7. Source - where the data come from, either from local provided file or web search


"""

def get_llm(outputModel: Optional[BaseModel] = None):
    """
    Get LLM instance based on the output schema
    """
    if outputModel is None:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
            # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        )
    else:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
            # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        ).with_structured_output(outputModel)
    return llm

def get_last_human_message(messages):
    """get last human message

    Args:
        messages messages list

    Returns:
        last human message
    """
    return filter_messages(messages, include_types=[HumanMessage])[-1]

# Post-processing
def format_docs(docs):
    """format the documents into a single string

    Args:
        docs (_type_): _description_

    Returns:
        _type_: _description_
    """
    return "\n\n".join(doc.page_content for doc in docs)
