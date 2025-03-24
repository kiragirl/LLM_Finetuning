import re

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from langchain_demo.graph import NO_PROPAGATE
from langchain_demo.graph.llm import tongyi_model
from langchain_demo.graph.my_message import MyMessagesState
from langchain_demo.graph.utils import call_llm_handler_result

members = ["trip-receipts", "delay-proof", "irregular-flight-rescheduling"]

system_prompt = f"""
    你是一个助手，帮助用户开具电子行程单。
    如果用户想要开具电子行程单没有提供时间, 则调用get_trip_receipts_orders_default函数
    如果用户想要开发票, 仅返回"目前仅支持电子行程单的开具，您是否需要"
    如果用户想要开具其他渠道客票的电子行程单, 则调用all_channel_trip_receipts函数
    
    仅按照以上条件返回,不要返回其他内容
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("messages")
    ]
)


@tool("all_channel_trip_receipts")
def all_channel_trip_receipts():
    """全渠道获取电子行程单"""
    return {
        "name": "all_channel_trip_receipts",
        "data": "",
        "custom_message": "全渠道获取电子行程单",
    }


@tool("get_trip_receipts_orders_default")
def get_trip_receipts_orders_default():
    """获取订单列表"""
    return {
        "name": "get_trip_receipts_orders",
        "data": {"id": "123456"},
        "custom_message": "查到您的订单",
    }


tools = [
    all_channel_trip_receipts,
    get_trip_receipts_orders_default,
]


async def trip_receipt_node(state: MessagesState):
    chain = prompt | call_llm_handler_result(tongyi_model, tools=tools)
    result = await chain.ainvoke(
        {"messages": state["messages"][-10:]}, {"tags": [NO_PROPAGATE]}
    )
    return result


trip_receipts_tool_node = ToolNode(tools)
