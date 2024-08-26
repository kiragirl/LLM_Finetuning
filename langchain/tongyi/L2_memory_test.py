from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_models import ChatTongyi
from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate


def test_conversation_buffer_memory():
    chat = ChatTongyi()
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        verbose=True
    )

    conversation.predict(input="Hi, my name is Andrew")

    conversation.predict(input="What is 1+1?")

    conversation.predict(input="What is my name?")

    print(memory.buffer)
    print("*********************************************")
    memory.load_memory_variables({})

    memory = ConversationBufferMemory()

    memory.save_context({"input": "Hi"},
                        {"output": "What's up"})

    print(memory.buffer)
    print("*********************************************")
    print(memory.load_memory_variables({}))

    memory.save_context({"input": "Not much, just hanging"},
                        {"output": "Cool"})

    print(memory.load_memory_variables({}))

# Define a custom method to count tokens
def count_tokens(messages, llm):
    # Convert the list of messages to a single string
    full_text = "\n".join([str(message['input']) + str(message['output']) for message in messages])
    # Count the tokens
    return llm.get_num_tokens(full_text)

def test_conversation_summary_memory():
    schedule = "There is a meeting at 8am with your product team. \
    You will need your powerpoint presentation prepared. \
    9am-12pm have time to work on your LangChain \
    project which will go quickly because Langchain is such a powerful tool. \
    At Noon, lunch at the italian resturant with a customer who is driving \
    from over an hour away to meet you to understand the latest in AI. \
    Be sure to bring your laptop to show the latest LLM demo."
    # 定义PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template="历史对话:\n{history}\n\n当前输入:\n{input}\n\n请回复:"
    )

    # 由于通译缺少get tokens方法需要重写
    class CustomTongyi(Tongyi):
        def get_num_tokens_from_messages(self, messages):
            # Convert the list of messages to a single string
            full_text = "\n".join([message.content for message in messages])
            # Count the tokens
            return self.get_num_tokens(full_text)

        def get_num_tokens(self, text):
            # Implement a simple token counting function
            # For simplicity, we'll assume each word is a token
            return len(text.split())

    llm = CustomTongyi()
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100,
        return_messages=False,
    )

    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},
                         {"output": "Cool"})
    memory.save_context({"input": "What is on the schedule today?"},
                        {"output": f"{schedule}"})

    memory.load_memory_variables({})

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    print(conversation.predict(input="What would be a good demo to show?"))
    print(memory.load_memory_variables({}))
