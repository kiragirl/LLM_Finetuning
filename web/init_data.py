from web.util.pdf_util import PDFUtil
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def store_file(path: str):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    store_vector(path, text)


def store_pdf(path: str):
    text = PDFUtil.extract_text_from_pdf(path)
    store_vector(path, text)


def store_vector(path: str, text: str):
    chunks = [text]
    embedding_function = HuggingFaceEmbeddings()
    persist_directory = 'pdf_chroma_db'
    chroma_client = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    metadata = [{"fileName": path}]
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


#store_file("../readme.md")
#query_vector_chroma_test("李一明有几年工作经验", "pdf_chroma_db")
query_vector_chroma_test("大模型实践Demo", "pdf_chroma_db")
