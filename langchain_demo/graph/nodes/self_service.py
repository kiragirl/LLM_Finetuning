import re

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessagesState

from langchain_demo.graph import NO_PROPAGATE
from langchain_demo.graph.llm import tongyi_model
from langchain_demo.graph.my_message import MyMessagesState
from langchain_demo.graph.utils import call_llm_handler_result

members = ["trip-receipts", "delay-proof", "irregular-flight-rescheduling"]

system_prompt = f"""
    你是一个自助服务代理，负责处理用户关于自助服务的请求。请根据用户的需求，从{members}中选择一个代理。
    自助服务包括三项服务分别是：开具电子行程单、开具航班延误证明（开具航延证明）和航班自助改期，分别对应trip-receipts、delay-proof和irregular-flight-rescheduling。
    trip-receipts：用于开具电子行程单
    delay-proof：用于开具航班延误证明
    irregular-flight-rescheduling：用于航班自助改期

    如果用户想要开具电子行程单，下一步应该是trip-receipts。
    如果用户想要开具航班延误证明（开具航延证明），下一步应该是delay-proof。
    如果用户想要进行航班自助改期，下一步应该是irregular-flight-rescheduling。 

    以下是一些示例对话：
    用户：我想要开具电子行程单
    AI：trip-receipts

    用户: 我想要开具航延证明
    AI：delay-proof

    用户: 我想要进行航班自助改期
    AI：irregular-flight-rescheduling
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("messages")
    ]
)


async def self_service_node(state: MessagesState):
    chain = prompt | call_llm_handler_result(tongyi_model, members=members)
    result = await chain.ainvoke(
        {"messages": state["messages"][-10:]},
    )
    return result
