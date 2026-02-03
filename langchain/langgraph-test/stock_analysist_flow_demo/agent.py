import logging
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from .state import GraphState
from .utils import get_llm, get_last_human_message, INITIAL_INSTRUCTION
from .response_models import LLMResponse


load_dotenv()
logger = logging.getLogger()

def llm_search(state: GraphState):
    """
    Call the LLM model to generate an answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    logger.info("---Calling Agent---")
    last_message = get_last_human_message(state.messages)
    messages = last_message.content
    response = get_llm(LLMResponse).invoke(messages)
    
    question = get_last_human_message(state.messages).content
    response = get_llm(LLMResponse).invoke(
      [SystemMessage(content=INITIAL_INSTRUCTION)] 
      + [HumanMessage(content=question)])
    return { "messages": ("user", response.generation) }
