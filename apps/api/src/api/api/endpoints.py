from fastapi import Request, APIRouter

from api.api.models import RAGRequest, RAGResponse, RAGUsedContext
from api.agents.graph import rag_agent_wrapper

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


rag_router = APIRouter()


@rag_router.post("/")
def chat(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    answer = rag_agent_wrapper(
        question=payload.query,
        thread_id=payload.thread_id,
        rerank=payload.rerank
    )

    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer['answer'],
        used_context= [RAGUsedContext(**used_context) for used_context in answer["used_context"]]
    )

api_router = APIRouter()

api_router.include_router(rag_router, prefix="/rag", tags=["rag"])