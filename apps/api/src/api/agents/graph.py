from typing import Dict, Any, Annotated, List
from pydantic import BaseModel
from operator import add

import numpy as np
import openai
from cohere import ClientV2
import instructor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage

from api.agents.agents import(
    ToolCall, RAGUsedContext, agent_node, intent_router_node
)
from api.agents.utils.utils import get_tool_descriptions
from api.agents.tools import get_formatted_context


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []
    rerank: bool = False

### Edges
def tool_router(state: State) -> str:
    
    """Decide whether to continue or end"""

    if state.final_answer == True:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"
    

def intent_router_conditional_edges(state: State):

    if state.question_relevant:
        return "agent_node"
    else:
        return "end"
    
### Tool Node with state injection

def tool_node(state: State) -> dict:
    tools_map = {"get_formatted_context": get_formatted_context}
    messages = []

    for i, tc in enumerate(state.tool_calls):
        tool_fn = tools_map.get(tc.name)
        if tool_fn:
            args = {**tc.arguments, "rerank": state.rerank}
            result = tool_fn(**args)
            messages.append(ToolMessage(content=result, tool_call_id=f"call_{i}"))

    return {"messages": messages, "tool_calls": []}

### Workflow

workflow = StateGraph(State)

tools = [get_formatted_context]
tool_descriptions = get_tool_descriptions(tools)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("intent_router_node", intent_router_node)


workflow.add_edge(START, "intent_router_node")
workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
    
)
workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END

    }
)
workflow.add_edge("tool_node", "agent_node")


graph = workflow.compile()

def run_agent(question:str, rerank:bool) -> str:
    initial_state = {
        "messages": [{"role":"user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions,
        "rerank": rerank
    }
    return graph.invoke(initial_state)

def rag_agent_wrapper(question: str, rerank:bool = False):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result= run_agent(
        question=question,
        rerank=rerank
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
        "answer": result.get("answer", "No answer provided"),
        "used_context": used_context,
    }