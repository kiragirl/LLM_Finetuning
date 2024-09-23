from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from web.chat_model_param import ChatModelParam


class Prompt:

    @staticmethod
    def acg_tone(context: str):
        template_string = """用{style}回答我的问题。问题在三重引号中。问题：```{text}``` """
        prompt_template = ChatPromptTemplate.from_template(template_string)

        customer_style = """二次元的语气"""
        customer_messages = prompt_template.format_messages(
            style=customer_style,
            text=context)

        chat = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME)
        response = chat.invoke(customer_messages)
        return response.content
