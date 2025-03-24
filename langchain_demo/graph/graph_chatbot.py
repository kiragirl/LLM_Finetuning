from typing import Annotated

from langchain_community.chat_models import ChatTongyi
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
llm_model = "qwen-turbo"
print(os.environ["DASHSCOPE_API_KEY"])


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatTongyi(model="qwen-turbo-latest")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()
