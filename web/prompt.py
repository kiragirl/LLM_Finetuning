from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate

from web.common_param import ChatModelParam


class Prompt:

    @staticmethod
    def acg_tone(context: str) -> str:
        template_string = """用{style}回答我的问题。问题在三重引号中。问题：```{text}``` """
        prompt_template = ChatPromptTemplate.from_template(template_string)

        customer_style = """二次元的语气"""
        customer_messages = prompt_template.format_messages(
            style=customer_style,
            text=context)

        chat = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME)
        response = chat.invoke(customer_messages)
        return response.content

    @staticmethod
    def comment_analysis(context: str) -> str:
        template_string = """你是一个美食网站评论助手，根据文本中的用户反馈的内容判断用户的对菜品的态度。如果判断结果是满意，回答满意否则回答不满意。文本在三重引号中。文本：```{text}```"""
        prompt_template = ChatPromptTemplate.from_template(template_string)
        customer_messages = prompt_template.format_messages(text=context)

        chat = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME)
        response = chat.invoke(customer_messages)
        return response.content
