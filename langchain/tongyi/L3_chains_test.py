from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatTongyi


def simple_sequential_test():
    llm = ChatTongyi()
    first_prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe \
        a company that makes {product}?"
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt)
    second_prompt = ChatPromptTemplate.from_template(
        "Write a 20 words description for the following \
        company:{company_name}"
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)
    overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                                 verbose=True
                                                 )
    product = "Queen Size Sheet Set"
    print(overall_simple_chain.invoke(product))


def sequential_test():
    llm = ChatTongyi()

    # prompt template 1: translate to english
    first_prompt = ChatPromptTemplate.from_template(
        "把评论翻译成英文:"
        "\n\n{Review}"
    )
    # chain 1: input= Review and output= English_Review
    chain_one = LLMChain(llm=llm, prompt=first_prompt,
                         output_key="English_Review"
                         )

    second_prompt = ChatPromptTemplate.from_template(
        "把评论总结成一句话:"
        "\n\n{English_Review}"
    )
    # chain 2: input= English_Review and output= summary
    chain_two = LLMChain(llm=llm, prompt=second_prompt,
                         output_key="summary"
                         )

    # prompt template 3: translate to english
    third_prompt = ChatPromptTemplate.from_template(
        "评论是什么语言写的:\n\n{Review}"
    )
    # chain 3: input= Review and output= language
    chain_three = LLMChain(llm=llm, prompt=third_prompt,
                           output_key="language"
                           )

    # prompt template 4: follow up message
    fourth_prompt = ChatPromptTemplate.from_template(
        "用指定的语言对以下摘要写一篇后续回复:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    # chain 4: input= summary, language and output= followup_message
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                          output_key="followup_message"
                          )
    # overall_chain: input= Review
    # and output= English_Review,summary, followup_message
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three, chain_four],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "language", "followup_message"],
        verbose=True
    )

    print(overall_chain.invoke("肉给的很多，鸡肉特别嫩，很入味，不柴，好吃，菜也特别多，米饭很香，好评好评"))


def router_test():
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""

    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:
    {input}"""

    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

    # In[ ]:

    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": math_template
        },
        {
            "name": "History",
            "description": "Good for answering history questions",
            "prompt_template": history_template
        },
        {
            "name": "computer science",
            "description": "Good for answering computer science questions",
            "prompt_template": computerscience_template
        }
    ]

    llm = ChatTongyi()

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: “destination” MUST be one of the candidate prompts specified below. If the input is not \
    well suited for any of the candidate prompts, it should be "DEFAULT". \
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    chain = VerifiedMultiPromptChain(router_chain=router_chain,
                                     destination_chains=destination_chains,
                                     default_chain=default_chain, verbose=True
                                     )
    print(chain.run("What is black body radiation?"))
    print(chain.run("what is 2 + 2"))
    print(chain.run("Why does every cell in our body contain DNA?"))


class VerifiedMultiPromptChain(MultiPromptChain):
    def _call(self, inputs):
        route = self.router_chain(inputs)
        destination = route['destination']
        next_inputs = route['next_inputs']

        # Simple verification step
        if destination not in self.destination_chains:
            print(f"Invalid destination: {destination}. Routing to DEFAULT.")
            destination = "DEFAULT"

        if destination == "DEFAULT":
            return self.default_chain(next_inputs, callbacks=self.callbacks)
        else:
            return self.destination_chains[destination](next_inputs, callbacks=self.callbacks)
