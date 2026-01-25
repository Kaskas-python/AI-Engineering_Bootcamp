from typing import Any
from pydantic import BaseModel, Field

import numpy as np
import openai
from cohere import ClientV2
import instructor
from instructor import Instructor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Document
from langsmith import traceable, get_current_run_tree

from api.agents.utils.prompt_management import prompt_template_config

class RAGUsedContext(BaseModel):
    id: str = Field(..., description="Id of item used to answer the question")
    description: str = Field(..., description="Short description of item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    references: list[RAGUsedContext] = Field(..., description="List of Items used to answer the question")

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
def retrieve_data(query: str, qdrant_client: QdrantClient , cohere_client:ClientV2 = None,  k:int =5):

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

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    if cohere_client:
        rerank_response = cohere_client.rerank(
            model="rerank-v4.0-fast",
            query=query,
            documents=retrieved_context,
            top_n=k
        )
        indices = [r.index for r in rerank_response.results]
        retrieved_context_ids = [retrieved_context_ids[i] for i in indices]
        retrieved_context = [retrieved_context[i] for i in indices]
        retrieved_context_ratings = [retrieved_context_ratings[i] for i in indices]
        similarity_scores = [r.relevance_score for r in rerank_response.results]

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }

@traceable(
        name="format_retrieved_context",
        run_type="prompt"
)
def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context

@traceable(
        name="generate_prompt_with_retireved_context",
        run_type="prompt"
)
def build_prompt(preprocessed_context: str, question: str) -> str:

    template = prompt_template_config(
        yaml_file="api/agents/prompts/retrieval_generation_prompts.yaml",
        prompt_key="retrieval_generation"
    )

    prompt = template.render(
        preprocessed_context=preprocessed_context,
        question=question
        )

    return prompt

@traceable(
        name="generate_answer",
        run_type="llm",
        metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def generate_answer(prompt:str, instructor_client: Instructor):

    response, raw_response = instructor_client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        response_model=RAGGenerationResponse
    )


    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return response

@traceable(
        name="RAG_pipeline"
)
def rag_pipeline(
    question:str, 
    qdrant_client: QdrantClient, 
    instructor_client: Instructor, 
    cohere_client:ClientV2 = None,
    top_k:int =5
    )-> dict[str, Any]:

    retrieved_context = retrieve_data(question, qdrant_client, cohere_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt, instructor_client)

    return {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrieved_context['retrieved_context_ids'],
        "retrieved_context": retrieved_context['retrieved_context'],
        "similarity_scores": retrieved_context['similarity_scores']
    }

def rag_pipeline_wrapper(question: str, top_k: int=5, rerank:bool = False):

    qdrant_client = QdrantClient(url="http://qdrant:6333")
    instructor_client = instructor.from_openai(openai.OpenAI())

    cohere_client = ClientV2() if rerank else None

    result= rag_pipeline(
        question=question,
        qdrant_client=qdrant_client,
        instructor_client=instructor_client,
        cohere_client=cohere_client,
        top_k=top_k
    )
    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=dummy_vector,
            limit=1,
            using="text-embedding-3-small",
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        ).points[0].payload
        
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append({
                "image_url": image_url,
                "price": price,
                "description": item.description
            })

    return {
        "answer": result["answer"],
        "used_context": used_context,
    }