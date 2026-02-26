from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import os

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.genai import types

from tools import check_warehouse_availability, reserve_warehouse_items

class WarehouseManagerAgent():
    def __init__(self, model_name: str = "openai/gpt-4.1-mini", api_key: str = None):

        self.model_name = model_name
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.model = self._init_model()
        self.agent = self._init_agent()
        
    def _init_model(self):

        model = LiteLlm(
            model=self.model_name,
            temperature=0,
            api_key=self.api_key
        )
        return model

    def _init_agent(self):

        agent = Agent(
            name="warehouse_manager_agent",
            model=self.model,
            tools=[check_warehouse_availability, reserve_warehouse_items],
            description="Agent is able to reserve items from the warehouses or check the availability of the items in warehouses",
            instruction="""
You are a part of the shopping assistant that can manage available inventory in the warehouses.

You will be given a conversation history and a list of tools, your task is to perform actions requested by the latest user query. Answer part of the query that you can answer with the available tools.

Instructions:
    - You must always check the availability of the items in the warehouses before reserving them.
    - Only reserve items in warehouses if entire order can be reserved or the user has confirmed that they want a partial reservation.
    - If you cannot reserve any items, return an answer that the order cannot be reserved.
    - If you can reserve some items, return an answer that the order can be partially reserved and include the details.
    - If only partial quantity can be reserved in some warehouses, try to combinethe required quantity from different warehouses.
    - Try to reserve items from the closest warehouse to the user first if users location is provided.
"""
        )
        return agent
    
    def get_agent(self) -> Agent:
        return self.agent