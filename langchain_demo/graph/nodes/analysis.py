import re
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessagesState
from pydantic import BaseModel
from langchain_core.messages import AIMessage

from langchain_demo.graph import NO_PROPAGATE
from langchain_demo.graph.llm import model, glm_model, tongyi_model
from langchain_demo.graph.my_message import MyMessagesState
from langchain_demo.graph.prompts import intention_analysis
from langchain_demo.graph.utils import call_llm_handler_result

members = [
    "flight-tickets-search",
    "flight-booking-reservation",
    "flight-dynamic",
    "manage-my-booking",
    "policy-enquiring",
    "payment-assistant",
    "self-service",
    "trip-receipts"
]


async def analysis_node(state: MessagesState):
    messages = state["messages"]
    latest_messages = messages[-10:]
    supervisor_chain = _get_chain()
    result = await supervisor_chain.ainvoke(
        {"messages": latest_messages}, {"tags": [NO_PROPAGATE]}
    )
    return result


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", intention_analysis.system_prompt),
        MessagesPlaceholder("messages"),
    ]
)


def _get_chain():
    supervisor_chain = prompt | call_llm_handler_result(tongyi_model, members=members)
    return supervisor_chain
