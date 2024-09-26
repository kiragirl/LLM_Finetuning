from config import Config
from flask import Flask, request, current_app
from flask_swagger_ui import get_swaggerui_blueprint
from prompt import Prompt
from rag import RAG
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from web.chat_request_message import ChatRequestMessage
from web.chat_response_message import ChatResponseMessage
import time

from web.record_info import RecordInfo

app = Flask(__name__)
_ = load_dotenv(find_dotenv())
app.config['PORT'] = Config.PORT
print(os.environ["DASHSCOPE_API_KEY"])


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/chat/acg', methods=['POST'])
def chat_acg():
    return get_response(request.get_json(), Prompt.acg_tone)


@app.route('/chat/commentAnalysis', methods=['POST'])
def comment_analysis():
    return get_response(request.get_json(), Prompt.comment_analysis)


@app.route('/chat/rag', methods=['POST'])
def comment_rag():
    return get_response(request.get_json(), RAG.retrieval_augmented_generation)


@app.route('/private/store', methods=['POST'])
def private_store():
    return get_response(request.get_json(), RecordInfo.record_personal_info)


def get_response(request_json, func):
    message = ChatRequestMessage(**request_json)
    response = ChatResponseMessage(func(message.context))
    return response.to_json()


def load_embeddings():
    start_time = time.time()
    app.config['embeddings'] = HuggingFaceEmbeddings()
    end_time = time.time()
    print(f"Load embedding took {end_time - start_time:.2f} seconds.")


# Swagger UI 配置
SWAGGER_URL = '/api/docs'  # 访问 Swagger UI 的 URL
API_URL = '/static/swagger.yaml'  # Swagger YAML 文件的 URL

# Swagger UI 蓝图
swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "LLM DEMO"
    }
)

app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    load_embeddings()
    app.run(port=app.config['PORT'])
