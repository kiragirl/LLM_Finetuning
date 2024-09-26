from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate

from web.agent import Agent
from web.common_param import ChatModelParam, DBParam
from web.rag import RAG
from web.util.access_util import AccessUtil


class PrivateInfo:

    # 保存角色介绍
    @staticmethod
    def intro_save(context: str) -> str:
        # 定义一个模板字符串，用于生成提示信息
        template_string = """你是一位个人助手，负责总结记录一些信息。简单总结一下三重引号中的文本是关于什么的。文本：```{text}```"""
        # 根据模板字符串创建一个提示模板
        prompt_template = ChatPromptTemplate.from_template(template_string)
        # 使用提示模板和上下文信息生成用户消息
        customer_messages = prompt_template.format_messages(text=context)
        # 创建一个 ChatTongyi 类的实例，用于调用聊天模型
        chat = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME)
        # 调用聊天模型，获取响应
        response = chat.invoke(customer_messages)
        # 从响应中提取介绍信息
        intro = response.content
        # 创建一个元数据列表，包含介绍信息
        metadata = [{"intro": intro}]
        # 将上下文信息和元数据保存到数据库中
        AccessUtil.store(context, metadata, DBParam.private_agent_db_name)
        # 返回介绍信息
        return intro

    @staticmethod
    def search(context: str) -> str:
        prompt = "。答案同时需要包括农历生日，格式为：MM-DD。"
        rag_response = RAG.retrieval_augmented_generation(context+prompt, DBParam.private_agent_db_name)
        return Agent.customize_tool(rag_response)
