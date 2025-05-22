import logging
from typing import Annotated

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import InjectedState

from .response_models import DocGraderResponse, LLMResponse
from .state import GraphState
from .utils import INITIAL_INSTRUCTION, format_docs, get_last_human_message, get_llm

load_dotenv()

# Setup module logger
logger = logging.getLogger(__name__)

# Set embeddings
embd = OpenAIEmbeddings()


def read_pdfs_and_split():
    """Read PDF files from a predefined directory, split them into smaller chunks, and return the chunks.

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


def retrieve_documents(state: Annotated[GraphState, InjectedState]):
    """Read documents, split them, and add to vectorstore.

    Args:

    Returns:
        Updated prompt
    """

    all_splits = read_pdfs_and_split()

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    retriever = vectorstore.as_retriever()
    # Write retrieved documents to documents key in state
    question = get_last_human_message(state.messages).content
    documents = retriever.get_relevant_documents(question)
    return { "documents": documents }
  
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    # Doc grader instructions
    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""


    # Grader prompt
    doc_grader_prompt = """
    Here is the retrieved document: \n\n {document} \n\n 
    Here is the user question: \n\n {question}. 
    
    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
    
    Return DocGraderResponse response, score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question.
    """


    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    last_message = get_last_human_message(state.messages)
    question = last_message.content
    documents = state.documents

    # Score each doc
    filtered_docs = []
    agent = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = get_llm(DocGraderResponse).invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = result.score
        # Document relevant
        if grade.lower() == "yes":
            logger.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            agent = "Yes"
            continue
    return {"documents": filtered_docs, "agent": agent}
  
def generate_answer(state):
    """
    Generate an answer based on the retrieved documents and the user question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logger.info("---GENERATE---")
    
    generate_prompt = """
    Here is the retrieved document: \n\n {document} \n\n 
    Here is the user question: \n\n {question}. 
    
    Find the answer for the given question in the document context,
    If there is no relevant information in the document, please say "I don't know".
    
    Return LLMResponse.
    """
    
    question = get_last_human_message(state.messages).content
    documents = state.documents
    # loop_step = state.get("loop_step", 0)
    # print(INITIAL_INSTRUCTION)
    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = generate_prompt.format(document=docs_txt, question=question)
    response = get_llm(LLMResponse).invoke(
      [SystemMessage(content=INITIAL_INSTRUCTION)] 
      + [HumanMessage(content=rag_prompt_formatted)])
#    generation = "dummy generation"
    return { "messages": ("user", response.generation) }