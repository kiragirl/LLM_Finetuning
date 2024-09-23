from config import Config
from flask import Flask, request, jsonify

from prompt import Prompt
import os
from dotenv import load_dotenv, find_dotenv

from web.chat_request_message import ChatRequestMessage
from web.chat_response_message import ChatResponseMessage

app = Flask(__name__)


@app.route('/chat/acg', methods=['POST'])
def chat_acg():
    message = ChatRequestMessage(**request.get_json())
    response = ChatResponseMessage(Prompt.acg_tone(message.context))
    return response.to_json()


# 设置端口号
_ = load_dotenv(find_dotenv())
app.config['PORT'] = Config.PORT
print(os.environ["DASHSCOPE_API_KEY"])


@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(port=app.config['PORT'])
