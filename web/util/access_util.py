import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from web.common_param import DBParam
from flask import current_app
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)


class AccessUtil:

    @staticmethod
    def store(content: str, metadata: Optional[List[dict]], db_name: str):
        chunks = [content]
        embedding_function = current_app.config['embeddings']
        chroma_client = Chroma(persist_directory=db_name, embedding_function=embedding_function)
        chroma_client.add_texts(chunks, metadatas=metadata)
