import logging
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from pydantic import ValidationError

from .state import GraphState


# local_llm = "llama3.2:3b-instruct-fp16"
# llm = ChatOllama(model=local_llm, temperature=0)
# llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

from .retriever_rag import retrieve_documents, grade_documents, generate_answer
from .agent import llm_search

# Setup module logger
logger = logging.getLogger(__name__)

# Prompt
INITIAL_PROMPT = """
You are a professional equity research analyst. Analyze the stock by a given stock name or code based on its latest quarterly earnings, valuation metrics (P/E, P/S, EV/EBITDA), competitive positioning, and current macroeconomic trends. Include the following in your report:

1. Company Overview – brief business summary and key revenue streams

2. Recent Financial Performance – revenue, EPS, margins (QoQ and YoY)

3. Valuation Comparison 

4. Technical Analysis – key support/resistance levels, moving averages

5. Catalysts & Risks – what might drive the stock price up or down in the next 6–12 months

6. Buy/Hold/Sell Recommendation – with a price target and rationale

Use up-to-date data and keep the tone professional and concise. Format your answer as a short research note.

Firstly, wait for me to input desired stock name or code to get analysis from

Then check go for the route_question, to determine where you can get information from, either vectorstore or run agent

Collect the information mentioned above

Then convert the response into a csv format with the following data as headers, wrapped with double quote

Date, stock name/code, company overview, recent financials, valuation metrics, technical analysis, cataysts, risks, recommendation

Start a new line, followed by field data, use double quote for fields and comma for separator.
Please use the query date as Date field, date format is YYYY-MM-DD, for example, 2025-12-30

Finally, ask me if I want to write the content into a file

No matter whether user want to write content to file or not, in the end, you should return to the beginning and ask user for next stock input
"""
### Edges
def route_question(state):
    """
    Route question to llm search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    
    logger.info("---ROUTE QUESTION---")
    if state.agent == "Yes":
        logger.info("---ROUTE QUESTION TO LLM Search---")
        return "llm_search"
    elif state.agent == "No":
        logger.info("---ROUTE QUESTION TO RAG Generate---")
        return "generate"

def run_workflow():
    """
    Run the workflow
    """
    workflow = StateGraph(GraphState)


    # Define the nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)  # generate response from documents
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("llm_search", llm_search)  # generate

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_question,
        {
            "llm_search": "llm_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("llm_search", END)
    

    # Compile
    graph = workflow.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))


    def print_stream(stream):
        try:
            for s in stream:
                message = s["messages"][-1]
                logger.info(f"Message received: {message.content[:200]}...")
                message.pretty_print()
        except ValidationError as exc:
            print("Error: " + repr(exc.errors()[0]))

    config = {"configurable": {"thread_id": 1}}
    logger.info(f"Set configuration: {config}")
    
    # logger.info("Starting conversation with initial prompt")
    # inputs = {"messages": [("user", INITIAL_PROMPT)]}
    # print_stream(graph.stream(inputs, config, stream_mode="values"))
            
    # Start chatbot
    logger.info("Entering interactive chat loop")
    while True:
        user_input = input("User: ")
        logger.info(f"Received user input: {user_input[:200]}...")
        inputs = {"messages": [("user", user_input)]}
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        print_stream(graph.stream(inputs, config, stream_mode="values"))