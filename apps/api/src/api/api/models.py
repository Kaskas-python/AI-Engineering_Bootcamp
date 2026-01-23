from pydantic import BaseModel, Field
from typing import  Optional

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")

class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="URL of item image")
    price: Optional[float] = Field(None, description="The price of the item")
    description: str = Field(...,description="Short description of item used to answer the question")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    used_context: list[RAGUsedContext] = Field(..., description="Used context for generating response to the query")