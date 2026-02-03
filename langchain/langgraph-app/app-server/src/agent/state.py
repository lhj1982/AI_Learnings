"""State management for the graph."""

from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[Document]

