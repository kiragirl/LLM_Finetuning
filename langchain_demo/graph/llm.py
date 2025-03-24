import os

from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
_ = load_dotenv(find_dotenv())
llm_model = "qwen-turbo"
print(os.environ["DASHSCOPE_API_KEY"])
model = ChatTongyi(model="qwen-turbo-latest")

tongyi_model = ChatOpenAI(
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.environ["DASHSCOPE_API_KEY"],
    model=llm_model,
    temperature=0.1,
    streaming=False,
)

glm_model = ChatOpenAI(
    openai_api_base="http://10.223.224.231:8088/api/paas/v4/",
    openai_api_key="0e955cb8dace43add2fec28f08b74760.F5TIIcZYse62Kbcq",
    model="glm-4",
    temperature=0.1,
    streaming=False,
)


