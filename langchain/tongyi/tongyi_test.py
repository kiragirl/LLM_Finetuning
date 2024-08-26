from prompt_parser_utils import *
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
llm_model = "qwen-turbo"
print(os.environ["DASHSCOPE_API_KEY"])
# print(get_completion("1+1等于几?", llm_model).choices[0].message.content)

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

# print(get_completion(prompt, llm_model).choices[0].mo  essage.content)

# langchain_chat()
# get_information()
get_structured_output()
