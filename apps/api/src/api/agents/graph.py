from typing import Dict, Any, Annotated, List, Callable
from pydantic import BaseModel
from operator import add

import numpy as np
import openai
from cohere import ClientV2
import instructor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import ToolMessage

from api.agents.agents import(
    ToolCall, RAGUsedContext, agent_node, intent_router_node
)
from api.agents.utils.utils import get_tool_descriptions
from api.agents.tools import get_formatted_context

class Tools(BaseModel):
    tools: list = []
    descriptions: list | str

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

def build_tools(**functions:Callable) -> Tools:
    tools = list(functions.values())
    tool_descriptions = get_tool_descriptions(tools)

    return Tools(
        tools= tools,
        descriptions= tool_descriptions
    )

def build_workflow(state_schema:type[BaseModel]) -> StateGraph: 

    workflow = StateGraph(state_schema)

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

    return workflow


def run_agent(question:str, thread_id:str, rerank:bool) -> str:

    tools = build_tools(get_formatted_context=get_formatted_context)

    state = {
        "messages": [{"role":"user", "content": question}],
        "iteration": 0,
        "available_tools": tools.descriptions,
        "rerank": rerank
    }
    config ={
        "configurable": {
            "thread_id": thread_id
        }
    }
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:

        workflow = build_workflow(State)
        graph = workflow.compile(checkpointer=checkpointer)

        result= graph.invoke(state, config)

    return result


def rag_agent_wrapper(question:str, thread_id:str, rerank:bool = False):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result= run_agent(
        question=question,
        thread_id=thread_id,
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