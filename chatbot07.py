#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

'''
通过BufferWindow记录前面几轮的对话，提升对话效果
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], 
        template=template
    )
    # 保持3轮对话
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)

    llm_chain = LLMChain(
        llm=OpenAI(), 
        prompt=prompt, 
        memory=memory,
        verbose=True
    )

    llm_chain.predict(human_input="你是谁？")
    llm_chain.predict(human_input="鱼香肉丝怎么做？")
    llm_chain.predict(human_input="那宫保鸡丁呢？")
    llm_chain.predict(human_input="我问你的第一句话是什么？")
    # 此时就会丢失第一个问题了
    llm_chain.predict(human_input="我问你的第一句话是什么？")

    # 查看历史问题记录
    memory.load_memory_variables({})
