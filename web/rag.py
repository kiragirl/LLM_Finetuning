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
