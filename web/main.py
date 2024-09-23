from config import Config
from flask import Flask, request, jsonify

from prompt import Prompt
import os
from dotenv import load_dotenv, find_dotenv

from web.chat_request_message import ChatRequestMessage
from web.chat_response_message import ChatResponseMessage

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


def get_response(request_json, func):
    message = ChatRequestMessage(**request_json)
    response = ChatResponseMessage(func(message.context))
    return response.to_json()


if __name__ == '__main__':
    app.run(port=app.config['PORT'])