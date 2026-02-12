from typing import List
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchAny, FusionQuery
import openai

class RetrievalOutput(BaseModel):
    context_id: str | None = None
    context: str | None = None
    rating: float | None = None
    similarity_score: float | None = None

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding

### ITEM Reviews retrieval tool

def retrieve_reviews_data(
    query: str,
    item_list: list[str],
    k:int = 5
    ):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    query_embedding= get_embedding(query)


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
        limit=k
    )
    retrieval_outputs = []

    for result in results.points:
        retrieval_output = RetrievalOutput(
            context_id=result.payload["parent_asin"],
            context=result.payload["text"],
            similarity_score=result.score
        )
        retrieval_outputs.append(retrieval_output)

    return retrieval_outputs


def process_reviews_context(context: List[RetrievalOutput]) ->str:

    formatted_context = ""

    for i in context: 
        formatted_context += f"- ID: {i.context_id}, description: {i.context}\n"

    return formatted_context