import operator
# from typing_extensions import TypedDict
from typing import List, Annotated, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class GraphState(BaseModel):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    # question: str  # User question
    messages: Annotated[list, add_messages]
    # generation: Optional[str] = Field(description="LLM generation", default=None) # LLM generation
    agent: Optional[str] = Field(description="Binary decision to run agent", default=None)  # Binary decision to run agent
    max_retries: Optional[int] = Field(description="", default=None)  # Max number of retries for answer generation
    answers: Optional[int] = Field(description="", default=None) # Number of answers generated
    loop_step: Optional[Annotated[int, operator.add]] = Field(description="", default=0)  # Loop step for retrying
    documents: Optional[List[Document]] = Field(description="", default=None)  # List of retrieved documents