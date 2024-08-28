from langchain.agents import load_tools, initialize_agent
from langchain_community.chat_models import ChatTongyi
from langchain.agents import AgentType
from langchain.agents import tool
from datetime import date
import langchain

def tool_test():
    llm = ChatTongyi()
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True)
    print(agent("What is the 25% of 300?"))
    question = "Tom M. Mitchell is an American computer scientist \
    and the Founders University Professor at Carnegie Mellon University (CMU)\
    what book did he write?"
    result = agent(question)
    print(result)


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())


def customize_tool():
    llm = ChatTongyi()
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    agent = initialize_agent(
        tools + [time],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True)
    try:
        langchain.debug = True
        result = agent("whats the date today?")
        print(result)
    except:
        print("exception on external access")
