from typing import List
from pydantic import BaseModel

import openai
from cohere import ClientV2
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Document, Filter, FieldCondition, MatchAny
from langsmith import traceable, get_current_run_tree

class RetrievalOutput(BaseModel):
    context_id: str | None = None
    context: str | None = None
    rating: float | None = None
    similarity_score: float | None = None

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

### ITEM Description retrieval tool

@traceable(
        name="retrieve_data",
        run_type="retriever"
)
def retrieve_items_data(
    query: str, 
    k:int = 5,
    rerank:bool = False
    ):

    qdrant_client = QdrantClient(url="http://qdrant:6333")
    cohere_client = ClientV2() if rerank else None

    query_embedding= get_embedding(query)

    fetch_limit = 20 if cohere_client else k

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-02-hybrid-search",
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
        retrieval_output = RetrievalOutput(
            context_id=result.payload["parent_asin"],
            context=result.payload["description"],
            rating=result.payload["average_rating"],
            similarity_score=result.score
        )
        retrieval_outputs.append(retrieval_output)

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
def process_items_context(context: List[RetrievalOutput]) ->str:

    formatted_context = ""

    for i in context: 
        formatted_context += f"- ID: {i.context_id}, rating: {i.rating}, description: {i.context}\n"

    return formatted_context

def get_formatted_items_context(query: str, top_k: int = 5, rerank: bool = False) -> str:

    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    context = retrieve_items_data(query=query, k=top_k, rerank=rerank)
    formatted_context = process_items_context(context)

    return formatted_context

### ITEM Reviews retrieval tool

@traceable(
        name="retrieve_reviews_data",
        run_type="retriever"
)
def retrieve_reviews_data(
    query: str,
    item_list: list[str],
    k:int = 5,
    rerank:bool = False
    ):

    qdrant_client = QdrantClient(url="http://qdrant:6333")
    cohere_client = ClientV2() if rerank else None

    query_embedding= get_embedding(query)

    fetch_limit = 20 if cohere_client else k

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-02-reviews",
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=item_list
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=fetch_limit
    )

    retrieval_outputs = []

    for result in results.points:
        retrieval_output = RetrievalOutput(
            context_id=result.payload["parent_asin"],
            context=result.payload["text"],
            similarity_score=result.score
        )
        retrieval_outputs.append(retrieval_output)

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
                similarity_score=r.relevance_score
            )
            reranked_outputs.append(reranked_output)

        return reranked_outputs

    return retrieval_outputs

@traceable(
        name="format_retrieved_reviews_context",
        run_type="prompt"
)
def process_reviews_context(context: List[RetrievalOutput]) ->str:

    formatted_context = ""

    for i in context: 
        formatted_context += f"- ID: {i.context_id}, description: {i.context}\n"

    return formatted_context

def get_formatted_reviews_context(query: str, item_list: list[str], top_k: int = 15, rerank: bool = False) -> str:

    """Get the top k reviews matching a query for a list of prefiltered items.
    
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    context = retrieve_reviews_data(query=query, item_list=item_list, k=top_k, rerank=rerank)
    formatted_context = process_reviews_context(context)

    return formatted_context