from typing import Dict, Any, Annotated, List, Callable
from pydantic import BaseModel, Field
from operator import add
import json

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
    ToolCall, RAGUsedContext, Delegation, 
    product_qa_agent_node, shopping_cart_agent_node, coordinator_agent_node
)
from api.agents.utils.utils import get_tool_descriptions
from api.agents.tools import (
    get_formatted_items_context, get_formatted_reviews_context,
    add_to_shopping_cart, remove_from_cart, get_shopping_cart
)

class Tools(BaseModel):
    tools: list = []
    descriptions: list | str

class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []

class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(default_factory=CoordinatorAgentProperties)
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""
    trace_id: str = ""
    rerank: bool = False

### Edges

def product_qa_agent_tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"
    
def shopping_cart_agent_tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def coordinator_agent_edge(state: State):

    if state.coordinator_agent.iteration > 3:
        return "end"
    elif state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    else:
        return "end"
    
### Tool Nodes with state injection

def product_qa_tool_node(state: State) -> dict:
    tools_map = {
        "get_formatted_items_context": get_formatted_items_context,
        "get_formatted_reviews_context": get_formatted_reviews_context
    }
    messages = []

    for i, tc in enumerate(state.product_qa_agent.tool_calls):
        tool_fn = tools_map.get(tc.name)
        if tool_fn:
            args = {**tc.arguments, "rerank": state.rerank}
            result = tool_fn(**args)
            messages.append(ToolMessage(content=str(result), tool_call_id=f"call_{i}"))

    return {
        "messages": messages,
        "product_qa_agent": {**state.product_qa_agent.model_dump(), "tool_calls": []}
    }


def shopping_cart_agent_tool_node(state: State) -> dict:
    tools_map = {
        "add_to_shopping_cart": add_to_shopping_cart,
        "remove_from_cart": remove_from_cart,
        "get_shopping_cart": get_shopping_cart
    }
    messages = []

    for i, tc in enumerate(state.shopping_cart_agent.tool_calls):
        tool_fn = tools_map.get(tc.name)
        if tool_fn:
            result = tool_fn(**tc.arguments)
            messages.append(ToolMessage(content=str(result), tool_call_id=f"call_{i}"))

    return {
        "messages": messages,
        "shopping_cart_agent": {**state.shopping_cart_agent.model_dump(), "tool_calls": []}
    }

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

    workflow.add_node("product_qa_agent", product_qa_agent_node)
    workflow.add_node("product_qa_tool_node", product_qa_tool_node)

    workflow.add_node("shopping_cart_agent", shopping_cart_agent_node)
    workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)


    workflow.add_node("coordinator_agent_node", coordinator_agent_node)


    workflow.add_edge(START, "coordinator_agent_node")
    workflow.add_conditional_edges(
        "coordinator_agent_node",
        coordinator_agent_edge,
        {
            "product_qa_agent": "product_qa_agent",
            "shopping_cart_agent": "shopping_cart_agent",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "product_qa_agent",
        product_qa_agent_tool_router,
        {
            "tools": "product_qa_tool_node",
            "end": "coordinator_agent_node"
        }
    )

    workflow.add_conditional_edges(
        "shopping_cart_agent",
        shopping_cart_agent_tool_router,
        {
            "tools": "shopping_cart_agent_tool_node",
            "end": "coordinator_agent_node"
        }
    )

    workflow.add_edge("product_qa_tool_node", "product_qa_agent")
    workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")

    return workflow



def run_agent_stream(question:str, thread_id:str, rerank:bool):


    product_qa_agent_tools = build_tools(
        get_formatted_items_context=get_formatted_items_context,
        get_formatted_reviews_context=get_formatted_reviews_context
    )
    shopping_cart_agent_tools = build_tools(
        add_to_shopping_cart=add_to_shopping_cart,
        remove_from_cart=remove_from_cart,
        get_shopping_cart=get_shopping_cart
    )
    state= {
        "messages": [{"role": "user", "content": question}],
        "user_id": thread_id,
        "cart_id": thread_id,
        "product_qa_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qa_agent_tools.descriptions,
            "tool_calls": []
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_agent_tools.descriptions,
            "tool_calls": []
        },
        "coordinator_agent":{
            "iteration": 0,
            "final_answer": False,
            "next_agent": "",
            "plan": []
        },
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

        result = None
        for chunk in graph.stream(
            state,
            config=config,
            stream_mode=["debug", "values"]
        ):
            processed_chunk = _process_graph_event(chunk)

            if processed_chunk:
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = chunk[1]

    yield result


### Streaming

def _string_for_sse(message:str) -> str:
    return f"data: {message}\n\n"

def _process_graph_event(chunk):

    def _is_node_start(chunk):
        return chunk[1].get("type") == "task"

    def _is_node_end(chunk):
        return chunk[0] == "updates"

    def _tool_to_text(tool_call):
        if tool_call.name == "get_formatted_items_context":
            return f"Looking for items: {tool_call.arguments.get('query', '')}."
        elif tool_call.name == "get_formatted_reviews_context":
            return f"Fetching user reviews..."
        elif tool_call.name == "add_to_shopping_cart":
            return f"Adding items to you shooping cart..."
        elif tool_call.name == "remove_from_cart":
            return f"Removing items from you cart..."
        elif tool_call.name == "get_shopping_cart":
            return f"Fetching items from your cart..."
        else:
            return f"Unknown tool: {tool_call.name}"

    if _is_node_start(chunk):
        node_name = chunk[1].get("payload", {}).get("name")
        input_state = chunk[1].get("payload", {}).get("input")

        if node_name == "coordinator_agent_node":
            return "Analysing the question..."
        if node_name == "product_qa_agent":
            return "Planning..."
        if node_name == "shopping_cart_agent":
            return "Performing..."
        if node_name == "product_qa_tool_node":
            tool_calls = input_state.product_qa_agent.tool_calls
            return " ".join([_tool_to_text(tc) for tc in tool_calls])
        if node_name == "shopping_cart_agent_tool_node":
            tool_calls = input_state.shopping_cart_agent.tool_calls
            return " ".join([_tool_to_text(tc) for tc in tool_calls])
    return False

def rag_agent_stream_wrapper(question:str, thread_id:str, rerank:bool):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = None
    for chunk in run_agent_stream(question, thread_id, rerank):
        if isinstance(chunk, dict):
            result = chunk
        else:
            yield chunk

    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in (result or {}).get("references", []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-02-hybrid-search",
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

    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [
        {
            "price": float(item.get("price")) if item.get("price") else None,
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": float(item.get("total_price")) if item.get("total_price") else None
        }
        for item in shopping_cart
    ]

    yield _string_for_sse(json.dumps(
        {
            "type": "final_answer",
            "data": {
                "answer": result.get("answer", "No answer provided"),
                "used_context": used_context,
                "trace_id": result.get("trace_id", ""),
                "shopping_cart": shopping_cart_items
            }

        }
    ))

# def run_agent(question:str, thread_id:str, rerank:bool) -> str:

#     product_qa_agent_tools = build_tools(
#         get_formatted_items_context=get_formatted_items_context,
#         get_formatted_reviews_context=get_formatted_reviews_context
#     )
#     shopping_cart_agent_tools = build_tools(
#         add_to_shopping_cart=add_to_shopping_cart,
#         remove_from_cart=remove_from_cart,
#         get_shopping_cart=get_shopping_cart
#     )
#     state= {
#         "messages": [{"role": "user", "content": question}],
#         "user_id": "abc2",
#         "cart_id": "2ABC",
#         "product_qa_agent": {
#             "iteration": 0,
#             "final_answer": False,
#             "available_tools": product_qa_agent_tools.descriptions,
#             "tool_calls": []
#         },
#         "shopping_cart_agent": {
#             "iteration": 0,
#             "final_answer": False,
#             "available_tools": shopping_cart_agent_tools.descriptions,
#             "tool_calls": []
#         },
#         "coordinator_agent":{
#             "iteration": 0,
#             "final_answer": False,
#             "next_agent": "",
#             "plan": []
#         },
#         "rerank": rerank
#     }

#     config ={
#         "configurable": {
#             "thread_id": thread_id
#         }
#     }
#     with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:

#         workflow = build_workflow(State)
#         graph = workflow.compile(checkpointer=checkpointer)

#         result= graph.invoke(state, config)

#     return result

    
# def rag_agent_wrapper(question:str, thread_id:str, rerank:bool = False):

#     qdrant_client = QdrantClient(url="http://qdrant:6333")

#     result= run_agent(
#         question=question,
#         thread_id=thread_id,
#         rerank=rerank
#         )
    
#     used_context = []
#     dummy_vector = np.zeros(1536).tolist()

#     for item in result.get("references", []):
#         payload = qdrant_client.query_points(
#             collection_name="Amazon-items-collection-02-hybrid-search",
#             query=dummy_vector,
#             limit=1,
#             using="text-embedding-3-small",
#             with_payload=True,
#             query_filter=Filter(
#                 must=[
#                     FieldCondition(
#                         key="parent_asin",
#                         match=MatchValue(value=item.id)
#                     )
#                 ]
#             )
#         ).points[0].payload
        
#         image_url = payload.get("image")
#         price = payload.get("price")
#         if image_url:
#             used_context.append({
#                 "image_url": image_url,
#                 "price": price,
#                 "description": item.description
#             })

#     return {
#         "answer": result.get("answer", "No answer provided"),
#         "used_context": used_context,
#         "trace_id": result.get("trace_id", "")
#     }