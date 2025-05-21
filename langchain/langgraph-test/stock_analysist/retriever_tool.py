import logging
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from .state import State

load_dotenv()

# Setup module logger
logger = logging.getLogger(__name__)

# Set embeddings
embd = OpenAIEmbeddings()


def read_pdfs_and_split():
    """_summary_

    Returns:
        _type_: _description_
    """
    filename = "/Users/james/projects/learnings/AI_Learnings/langchain/langgraph-test/data_sources/2025q1-alphabet-earnings-release.pdf"
    loader = PyPDFLoader(file_path=filename)
    docs = []
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=30, add_start_index=True
    )

    all_splits = text_splitter.split_documents(docs)

    logger.info(f"Split the pdf into {len(all_splits)} sub-documents.")
    return all_splits


@tool
def index_files(state: Annotated[dict, InjectedState]) -> State:
    """Index files if under predefined file path

    Args:

    Returns:
        Updated prompt
    """

    # print(f"afafadsd  {state.get('context')}")
    all_splits = read_pdfs_and_split()

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    retriever = vectorstore.as_retriever()
    state["context"] = retriever
    # state_update = {
    #   context: []
    # }
    return state


@tool
def retriev_docs():
    """Retrieve documents if there is any

    Args:

    Returns:
        Updated prompt
    """
    return
