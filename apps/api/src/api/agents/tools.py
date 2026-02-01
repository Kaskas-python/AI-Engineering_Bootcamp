from typing import List
from pydantic import BaseModel

import openai
from cohere import ClientV2
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Document
from langsmith import traceable, get_current_run_tree

class RetrievalOutput(BaseModel):
    context_id: str
    context: str
    rating: float
    similarity_score: float

@traceable(
        name="embed_query",
        run_type="embedding",
        metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )
    
    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
    return response.data[0].embedding

@traceable(
        name="retrieve_data",
        run_type="retriever"
)
def retrieve_data(
    query: str, 
    k:int = 5,
    rerank:bool = False
    ):

    qdrant_client = QdrantClient(url="http://qdrant:6333")
    cohere_client = ClientV2() if rerank else None

    query_embedding= get_embedding(query)

    fetch_limit = 20 if cohere_client else k

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=fetch_limit
    )

    retrieval_outputs = []

    for result in results.points:
        retrieval_outuput = RetrievalOutput(
            context_id=result.payload["parent_asin"],
            context=result.payload["description"],
            rating=result.payload["average_rating"],
            similarity_score=result.score
        )
        retrieval_outputs.append(retrieval_outuput)

    if cohere_client:
        context_list =[c.context for c in retrieval_outputs]

        rerank_response = cohere_client.rerank(
            model="rerank-v4.0-fast",
            query=query,
            documents=context_list,
            top_n=k
        )

        reranked_outputs =[]

        for r in rerank_response.results:
            original = retrieval_outputs[r.index]
            reranked_output = RetrievalOutput(
                context_id=original.context_id,
                context=original.context,
                rating=original.rating,
                similarity_score=r.relevance_score
            )
            reranked_outputs.append(reranked_output)

        return reranked_outputs

    return retrieval_outputs

@traceable(
        name="format_retrieved_context",
        run_type="prompt"
)
def process_context(context: List[RetrievalOutput]) ->str:

    formatted_context = ""

    for i in context: 
        formatted_context += f"- ID: {i.context_id}, rating: {i.rating}, description: {i.context}\n"

    return formatted_context

def get_formatted_context(query: str, top_k: int = 5, rerank: bool = False) -> str:

    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    context = retrieve_data(query=query, k=top_k, rerank=rerank)
    formatted_context = process_context(context)

    return formatted_context