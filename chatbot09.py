#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

'''
SummaryBufferMemory通过总结之前的对话，并记录前面几轮的对话，提升对话效果
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    SUMMARIZER_TEMPLATE = """请将以下内容逐步概括所提供的对话内容，并将新的概括添加到之前的概括中，形成新的概括。

    EXAMPLE
    Current summary:
    Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量。

    New lines of conversation:
    Human：为什么你认为人工智能是一种积极的力量？
    AI：因为人工智能将帮助人类发挥他们的潜能。

    New summary:
    Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量，因为它将帮助人类发挥他们的潜能。
    END OF EXAMPLE

    Current summary:
    {summary}

    New lines of conversation:
    {new_lines}

    New summary:"""

    SUMMARY_PROMPT = PromptTemplate(
        input_variables=["summary", "new_lines"], template=SUMMARIZER_TEMPLATE
    )
    # max_token_limit：当对话的长度到多长之后，我们就应该调用LLM去把文本内容小结一下
    memory = ConversationSummaryBufferMemory(llm=OpenAI(), prompt=SUMMARY_PROMPT, max_token_limit=256)

    CHEF_TEMPLATE = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文。
    2. 对于做菜步骤的回答尽量详细一些。

    {history}
    Human: {input}
    AI:"""
    CHEF_PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=CHEF_TEMPLATE
    )

    # 开始了verbose模式
    conversation_with_summary = ConversationChain(
        llm=OpenAI(model_name="text-davinci-003", stop="\n\n", max_tokens=2048, temperature=0.5), 
        prompt=CHEF_PROMPT,
        memory=memory,
        verbose=True
    )
    answer = conversation_with_summary.predict(input="你是谁？")
    print(answer)
    
    answer = conversation_with_summary.predict(input="请问鱼香肉丝怎么做？")
    print(answer)

    # 此时会触发summary
    answer = conversation_with_summary.predict(input="那蚝油牛肉呢？")
    print(answer)


    # 但即使是ConversationSummaryBufferMemory，有时候也并不能自动提取到最关心的内容
    # max_token_limit：当对话的长度到多长之后，我们就应该调用LLM去把文本内容小结一下
    memory = ConversationSummaryBufferMemory(llm=OpenAI(), prompt=SUMMARY_PROMPT, max_token_limit=40)
    memory.save_context(
        {"input": "你好"}, 
        {"ouput": "你好，我是客服李四，有什么我可以帮助您的么"}
        )
    memory.save_context(
        {"input": "我叫张三，在你们这里下了一张订单，订单号是 2023ABCD，我的邮箱地址是 customer@abc.com，但是这个订单十几天了还没有收到货"}, 
        {"ouput": "好的，您稍等，我先为您查询一下您的订单"}
        )
    # 会丢失部分必要信息
    memory.load_memory_variables({})
