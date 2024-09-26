from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate

from web.common_param import ChatModelParam, DBParam
from web.util.access_util import AccessUtil


class RecordInfo:

    @staticmethod
    def record_personal_info(context: str) -> str:
        template_string = """你是一位个人助手，复制总结记录一些信息。总结一下三重引号中的文本是关于什么的。文本：```{text}```"""
        prompt_template = ChatPromptTemplate.from_template(template_string)
        customer_messages = prompt_template.format_messages(text=context)
        chat = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME)
        response = chat.invoke(customer_messages)
        intro = response.content
        metadata = [{"intro": intro}]
        AccessUtil.store(context, metadata, DBParam.private_agent_db_name)
        return intro
