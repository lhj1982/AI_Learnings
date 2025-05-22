from pydantic import BaseModel, Field

# Pydantic models for structured output

class RouteResponse(BaseModel):
    """Response of the route"""
    datasource: str = Field(description="The data source used to answer the question")
    
class RetriverScore(BaseModel):
    """Result of PDF reader"""
    score: str = Field(description="The score of the retriever")


class DocGraderResponse(BaseModel):
    """Result of document grader"""
    score: str = Field(description="The score of the document grader")
    documents: list[str] = Field(description="List of documents used for the answer")

class LLMResponse(BaseModel):
    """Result of LLM model"""
    generation: str = Field(description="The generated text from the LLM")