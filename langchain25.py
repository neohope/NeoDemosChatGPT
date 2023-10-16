#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

'''
用get_openai_callback构造token计数器
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI(temperature=0.5, model_name="text-davinci-003")

    # 初始化对话链
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory()
    )

    # 使用context manager进行token counting
    with get_openai_callback() as cb:
        # 第一天的对话
        # 回合1
        conversation("我姐姐明天要过生日，我需要一束生日花束。")
        print("第一次对话后的记忆:", conversation.memory.buffer)

        # 回合2
        conversation("她喜欢粉色玫瑰，颜色是粉色的。")
        print("第二次对话后的记忆:", conversation.memory.buffer)

        # 回合3 （第二天的对话）
        conversation("我又来了，还记得我昨天为什么要来买花吗？")
        print("/n第三次对话后时提示:/n",conversation.prompt.template)
        print("/n第三次对话后的记忆:/n", conversation.memory.buffer)

    # 输出使用的tokens
    print("\n总计使用的tokens:", cb.total_tokens)
