import asyncio
from typing import Annotated, Literal, TypedDict, Sequence

from langchain_core.messages import HumanMessage, BaseMessage
# from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState, add_messages
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv, find_dotenv

from langchain_demo.graph import NO_PROPAGATE
from langchain_demo.graph.my_message import MyMessagesState
from langchain_demo.graph.nodes.analysis import analysis_node
from langchain_demo.graph.nodes.self_service import self_service_node
from langchain_demo.graph.nodes.trip_receipt import trip_receipt_node, trip_receipts_tool_node

TRIP_RECEIPTS_TOOL = "trip-receipts-tool"

TRIP_RECEIPTS = "trip-receipts"

SELF_SERVICE = "self-service"

ANALYSIS = "analysis"


def conditional_func(state: MessagesState):
    messages = state["messages"]
    latest_message = messages[-1]
    next_node = latest_message.content
    print("The next node is " + next_node)
    return next_node


# Define a new graph
workflow = StateGraph(MessagesState)

workflow.add_node(ANALYSIS, analysis_node)
workflow.add_conditional_edges(ANALYSIS, conditional_func)
workflow.add_node(SELF_SERVICE, self_service_node)
#.add_edge(SELF_SERVICE, TRIP_RECEIPTS)
workflow.add_conditional_edges(SELF_SERVICE, conditional_func)
workflow.add_node(TRIP_RECEIPTS, trip_receipt_node)
workflow.add_edge(TRIP_RECEIPTS, TRIP_RECEIPTS_TOOL)
workflow.add_node(TRIP_RECEIPTS_TOOL, trip_receipts_tool_node)

workflow.set_entry_point(ANALYSIS)
workflow.set_finish_point(TRIP_RECEIPTS)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)


async def invoke():
    # Use the Runnable
    # events = app.astream_events({"messages": [HumanMessage(content="开具电子行程单")]},
    #                             config={"configurable": {"thread_id": 42}},
    #                             exclude_tags=[NO_PROPAGATE],
    #                             )
    # async for event in events:
    #     print(event)
    final_state = await app.ainvoke(
        {"messages": [HumanMessage(content="开具电子行程单")]},
        config={"configurable": {"thread_id": 42}}
    )

    print("------------------------")
    print(final_state["messages"][-1].content)
    print("------------------------")
    print(final_state)

if __name__ == '__main__':
    asyncio.run(invoke())
