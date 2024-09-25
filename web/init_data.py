from pdf_util import PDFUtil
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag import RAG
from dotenv import load_dotenv, find_dotenv


def store_vector(path: str):
    text = PDFUtil.extract_text_from_pdf(path)
    # chunks = PDFUtil.split_into_chunks(text)
    chunks = [text]
    embedding_function = HuggingFaceEmbeddings()
    # embeddings = embedding_function.embed_documents(chunks)
    persist_directory = 'pdf_chroma_db'
    # 创建或连接到一个 Chroma 数据库
    chroma_client = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    metadata = [{"fileName":path}]
    # 将提取的文本转换为向量，并将元数据一起添加到Chroma数据库中
    chroma_client.add_texts(chunks, metadatas=metadata)


def query_vector_chroma_test(query: str, persist_directory: str):
    start_time = time.time()
    embedding_function = HuggingFaceEmbeddings()
    end_time = time.time()
    print(f"Load embedding took {end_time - start_time:.2f} seconds.")
    # 创建或连接到一个 Chroma 数据库
    start_time = time.time()
    chroma_client = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    end_time = time.time()
    print(f"Load chroma took {end_time - start_time:.2f} seconds.")
    # 记录开始时间
    start_time = time.time()
    # 定义查询
    # 执行查询
    results = chroma_client.similarity_search_with_score(query, k=1)
    # 记录结束时间
    end_time = time.time()
    # 输出查询时间
    print(f"Query took {end_time - start_time:.2f} seconds.")

    # 打印结果
    for result, score in results:
        print(f"Similarity Score: {score}")
        print(f"Document: {result.page_content}")
        print(f"Metadata: {result.metadata}")
        print("-----")


#store_vector("yiming.pdf")
#query_vector_chroma_test("李一明有几年工作经验", "pdf_chroma_db")
_ = load_dotenv(find_dotenv())
RAG.retrieval_augmented_generation("李一明有几年工作经验")
