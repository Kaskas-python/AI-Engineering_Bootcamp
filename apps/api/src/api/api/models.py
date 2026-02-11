from pydantic import BaseModel, Field
from typing import  Optional, Union

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    thread_id: str = Field(..., description="Thread ID of conversation")
    rerank: bool = Field(False, description="Enable reranking")

class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="URL of item image")
    price: Optional[float] = Field(None, description="The price of the item")
    description: str = Field(...,description="Short description of item used to answer the question")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    used_context: list[RAGUsedContext] = Field(..., description="Used context for generating response to the query")
    trace_id: str = Field(..., description="Trace ID")

class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(..., description="1 if the feedback is positive, 0 if the feedback is negative")
    feedback_text: str = Field(..., description="The feedback text")
    trace_id: str = Field(..., description="The trace ID")
    thread_id: str = Field(..., description="The thread ID")
    feedback_source_type: str = Field(..., description="The type of feedback. Human or API")

class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    status: str = Field(..., description="The status of the feedback submission")