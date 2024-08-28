from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_models import ChatTongyi
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import time
from langchain.chains import RetrievalQA

def vector_test():
    file = 'OutdoorClothingCatalog_1000.csv'
    loader = CSVLoader(file_path=file, encoding='utf-8')
    embedding_function = HuggingFaceEmbeddings()
    index = VectorstoreIndexCreator(
        embedding=embedding_function,
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])

    query = "Please list all your shirts with sun protection \
    in a table in markdown and summarize each one."

    llm = ChatTongyi()

    response = index.query(query, llm=llm)
    print(response)


def vector_retrival_test():
    embeddings = HuggingFaceEmbeddings()
    persist_directory = 'chroma_db'
    # 创建或连接到一个 Chroma 数据库
    chroma_client = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = chroma_client.as_retriever()
    llm = ChatTongyi()
    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    query = "Please list all your shirts with sun protection in a table \
    in markdown and summarize each one."
    response = qa_stuff.invoke(query)
    print(response)


def query_vector_chroma_test():
    embedding_function = HuggingFaceEmbeddings()
    persist_directory = 'chroma_db'
    # 创建或连接到一个 Chroma 数据库
    chroma_client = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    # 记录开始时间
    start_time = time.time()
    # 定义查询
    query = "Women's Campside Oxfords"
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


def store_vector_test():
    embedding_function = HuggingFaceEmbeddings()
    # 初始化 OpenAI 的嵌入服务 DashScopeEmbeddings付费的
    # embedding_function = DashScopeEmbeddings()
    persist_directory = 'chroma_db'
    # 创建或连接到一个 Chroma 数据库
    chroma_client = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    csv_file_path = 'OutdoorClothingCatalog_1000.csv'
    df = pd.read_csv(csv_file_path)
    # 3. 数据预处理
    # 提取'name'和'description'列
    texts = df['description'].tolist()
    metadatas = [{'name': name} for name in df['name']]
    # 将提取的文本转换为向量，并将元数据一起添加到Chroma数据库中
    chroma_client.add_texts(texts, metadatas=metadatas)
    # 如果之前存在数据，需要保证维度一致，HuggingFaceEmbeddings的更换会导致维度不一致
    chroma_client.persist()
    # 添加文档到 Chroma
    # documents = ["这是一段示例文本。", "这是另一段不同的文本。"]
    # ids = ["id1", "id2"]
    # chroma_client.add_texts(texts=documents, ids=ids)

    # 查询与给定文本最相似的嵌入
    # query = "示例查询文本"
    query = "Women's Campside Oxfords"
    results = chroma_client.similarity_search_with_score(query, k=1)
    print(results)
