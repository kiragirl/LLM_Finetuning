from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class MyMessagesState(TypedDict):
    """
    AgentState 类用于表示代理的状态信息，是一个 TypedDict 类型的类。
    它定义了代理在不同情境下的各种状态字段，以帮助跟踪和管理代理的行为和数据。
    """

    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The 'next' field indicates where to route to next
    next: str
    # search_result: dict | None
    booking_id: str | None
    step: int | None
    alarm: bool
    uid: str | None