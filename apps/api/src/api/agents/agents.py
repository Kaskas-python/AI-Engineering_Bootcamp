from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage, convert_to_openai_messages

from openai import OpenAI

import instructor

from api.agents.utils.utils import format_ai_message
from api.agents.utils.prompt_management import prompt_template_config

### Tool calls

class ToolCall(BaseModel):
    name: str
    arguments: dict

### Coordinator Agent Response schemas

class Delegation(BaseModel):
    agent: str
    task: str

class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: list[Delegation]
    final_answer: bool = False
    answer: str = "" 

### Q&A Agent Response schemas

class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")
    
class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

### Shopping Cart Agent Response schemas

class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Q&A Agent Node

@traceable(
    name="product_qa_agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def product_qa_agent_node(state) -> dict:

    template = prompt_template_config(
        yaml_file="api/agents/prompts/product_qa_agent.yaml",
        prompt_key="product_qa_agent"
    )
   
    prompt = template.render(
        available_tools=state.product_qa_agent.available_tools
    )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ProductQAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )

    current_run = track_current_run(raw_response)

    ai_message = format_ai_message(response)
    return {
        "messages": [ai_message],
        "product_qa_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.product_qa_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.product_qa_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references
    }
### Shopping Cart Agent Node

@traceable(
    name="shopping_cart_agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent_node(state) -> dict:

    template = prompt_template_config(
        yaml_file="api/agents/prompts/shopping_cart_agent.yaml",
        prompt_key="shopping_cart_agent"
    )

    prompt = template.render(
        available_tools=state.shopping_cart_agent.available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )

    current_run = track_current_run(raw_response)

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "shopping_cart_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.shopping_cart_agent.available_tools
        },
        "answer": response.answer
    }

### Coordinator Agent Node

@traceable(
    name="coordinator_agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)

def coordinator_agent_node(state):

    template = prompt_template_config(
        yaml_file="api/agents/prompts/coordinator_agent.yaml",
        prompt_key="coordinator_agent"
    )
    prompt = template.render()

    messages = state.messages

    conversation =[]

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1",
            response_model=CoordinatorAgentResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0,
    )

    current_run = track_current_run(raw_response)

    trace_id = str(getattr(current_run, "trace_id", current_run.id)) if current_run else None

    if response.final_answer:
        ai_message = [
            AIMessage(
            content=response.answer,
            )
        ]

    else:
        ai_message = []

    return {
        "messages": ai_message,
        "answer": response.answer,
        "coordinator_agent":{
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,
            "plan": response.plan
        },
        "trace_id": trace_id
   }

### Tracking Utils

def track_current_run(raw_response):
    
    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
        
    return current_run