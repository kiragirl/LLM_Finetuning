import re

from langchain_core.messages import AIMessage


def call_llm_handler_result(llm_agent, tools=None, members=None):
    """
    配置结构化输出的函数。

    该函数旨在处理来自llm_agent的输出，确保它符合next_agent预期的格式。
    如果原始输出不符合预期格式，函数会尝试从错误信息中检索下一个agent的信息。

    参数:
    - llm_agent: 当前的智能编码代理实例。
    - next_agent: 预期的下一个代理的名称或标识。

    返回值:
    返回一个内部函数inner，该函数接收一个输入x，并尝试以结构化的方式处理llm_agent的输出。
    """

    def inner(x):
        if tools is not None:
            llm_agent_l = llm_agent.bind_tools(tools)
        else:
            llm_agent_l = llm_agent
        result = llm_agent_l.invoke(x)
        print(f"The llm response is {result}, The type is {type(result)}")
        # 有的模型返回tool_calls，不返回content（glm4）
        if isinstance(result, AIMessage) and "tool_calls" in result.additional_kwargs:
            #tool_call = result.tool_calls[0]
            #return {"next": tool_call["args"]["next"]}
            return {"messages": result}
        if isinstance(result, AIMessage):
            if members is not None:
                result = list(filter(lambda t: re.search(t, result.content), members))
                result = result[0]
                return {"messages": result}
            return {"messages": result}
        elif result is None:
            raise AttributeError("The model's output is None")

    return inner
