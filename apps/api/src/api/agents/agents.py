from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import convert_to_openai_messages

from openai import OpenAI

import instructor

from api.agents.utils.utils import format_ai_message
from api.agents.utils.prompt_management import prompt_template_config


### Intent Router Response

class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str

### Q&A Agent Response sheemas

class ToolCall(BaseModel):
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")
    
class AgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

### Q&A Agent Node

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def agent_node(state) -> dict:

    template = prompt_template_config(
        yaml_file="api/agents/prompts/qa_agent.yaml",
        prompt_key="qa_agent"
    )
   
    prompt = template.render(
        available_tools=state.available_tools
    )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    current_run = track_current_run(raw_response)

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }

### Intent Router Agent Node

@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)

def intent_router_node(state):


    template = prompt_template_config(
        yaml_file="api/agents/prompts/intent_router_agent.yaml",
        prompt_key="intent_router_agent"
    )
    prompt = template.render()

    messages = state.messages

    conversation =[]

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1-mini",
            response_model=IntentRouterResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0,
    )

    current_run = track_current_run(raw_response)

    trace_id = str(getattr(current_run, "trace_id", current_run.id)) if current_run else None

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
        "trace_id": trace_id
        }

### Utils

def track_current_run(raw_response):
    
    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
        
    return current_run