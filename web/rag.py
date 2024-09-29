from langchain_chroma import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain.chains import RetrievalQA
from web.common_param import ChatModelParam
from web.common_param import DBParam
from flask import current_app
import time


class RAG:

    @staticmethod
    def retrieval_augmented_generation_file(query: str) -> str:
        start_time = time.time()
        embedding_function = current_app.config['embeddings']
        chroma_client = Chroma(persist_directory=DBParam.file_db_name, embedding_function=embedding_function)
        retriever = chroma_client.as_retriever()
        llm = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME)
        qa_stuff = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=False
        )
        response = qa_stuff.invoke(query)
        end_time = time.time()
        print(f"RAG took {end_time - start_time:.2f} seconds.")
        print(response)
        return response['result']

    @staticmethod
    def retrieval_augmented_generation(query: str, db_name: str) -> str:
        """
        使用检索增强生成（RAG）来回答给定的查询。

        参数：
            query (str)：用户的查询字符串。
            db_name (str)：存储嵌入数据的数据库名称。

        返回：
            str：生成的回答。

        该方法使用预配置的嵌入函数和 Chroma 客户端来检索相关信息，然后使用聊天模型生成回答。
        """
        start_time = time.time()
        embedding_function = current_app.config['embeddings']
        chroma_client = Chroma(persist_directory=db_name, embedding_function=embedding_function)
        retriever = chroma_client.as_retriever()
        llm = ChatTongyi(model=ChatModelParam.LLM_MODEL_NAME, temperature=0)
        qa_stuff = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=False
        )
        response = qa_stuff.invoke(query)
        end_time = time.time()
        print(f"RAG took {end_time - start_time:.2f} seconds.")
        print(response)
        return response['result']
