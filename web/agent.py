from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatTongyi
from langchain.agents import AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from datetime import date, datetime
import langchain
from langchain_core.tools import tool
from zhdate import ZhDate
from web.common_param import ChatModelParam


class Agent:

    @staticmethod
    def customize_tool(query: str) -> str:
        llm = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME)
        tools = load_tools(["llm-math", "wikipedia"], llm=llm)
        agent = initialize_agent(
            tools + [birthday_convert],
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=False)
        try:
            langchain.debug = True
            result = agent(query)
            print("---------------------------------")
            print(result)
            return result['output']
        except:
            print("exception on external access")


@tool
def birthday_convert(text: str) -> str:
    """可以将农历日期转换为公历日期的工具.如果需要查阅当年的农历与公历对照表，使用此工具。如果需要日历转换工具，使用此工具。输入格式为MM-DD。返回公历日期。"""
    current_year = datetime.now().year.__str__()
    birthday = current_year + "-" + text
    date_birthday = datetime.strptime(birthday, "%Y-%m-%d").date()
    return ZhDate(date_birthday.year, date_birthday.month, date_birthday.day).to_datetime()
