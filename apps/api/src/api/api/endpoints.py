from fastapi import Request, APIRouter
from qdrant_client import QdrantClient

from api.api.models import RAGRequest, RAGResponse
from api.agents.retrieval_generation import rag_pipeline

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url="http://qdrant:6333")

rag_router = APIRouter()


@rag_router.post("/")
def chat(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    answer = rag_pipeline(question= payload.query, qdrant_client= qdrant_client)

    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer['answer']
    )

api_router = APIRouter()

api_router.include_router(rag_router, prefix="/rag", tags=["rag"])