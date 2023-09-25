#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory

'''
通过memory记忆之前的聊天，保持语境
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

'''
ConversationChain生成的提示信息
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:
'''
def conversation_chain_demo():
    # 初始化大语言模型
    llm = OpenAI(
        temperature=0.5,
        model_name="text-davinci-003"
    )

    # 初始化对话链
    conv_chain = ConversationChain(llm=llm)

    # 打印对话的模板
    print(conv_chain.prompt.template)

def conversation_buffer_memory_demo():
    # 初始化大语言模型
    llm = OpenAI(
        temperature=0.5,
        model_name="text-davinci-003")

    # 初始化对话链
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory()
    )

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

def conversation_buffer_window_memory_demo():
    # 创建大语言模型实例
    llm = OpenAI(
        temperature=0.5,
        model_name="text-davinci-003")

    # 初始化对话链
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferWindowMemory(k=1)
    )

    # 第一天的对话
    # 回合1
    result = conversation("我姐姐明天要过生日，我需要一束生日花束。")
    print(result)

    # 回合2
    result = conversation("她喜欢粉色玫瑰，颜色是粉色的。")
    # print("\n第二次对话后的记忆:\n", conversation.memory.buffer)
    print(result)

    # 第二天的对话
    # 回合3
    result = conversation("我又来了，还记得我昨天为什么要来买花吗？")
    print(result)

def conversation_summary_memory_demo():
    # 创建大语言模型实例
    llm = OpenAI(
        temperature=0.5,
        model_name="text-davinci-003")

    # 初始化对话链
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationSummaryMemory(llm=llm)
    )

    # 第一天的对话
    # 回合1
    result = conversation("我姐姐明天要过生日，我需要一束生日花束。")
    print(result)

    # 回合2
    result = conversation("她喜欢粉色玫瑰，颜色是粉色的。")
    # print("\n第二次对话后的记忆:\n", conversation.memory.buffer)
    print(result)

    # 第二天的对话
    # 回合3
    result = conversation("我又来了，还记得我昨天为什么要来买花吗？")
    print(result)

if __name__ == '__main__':
    # 初始化模型
    get_api_key()

    conversation_chain_demo()
    conversation_buffer_memory_demo()
    conversation_buffer_window_memory_demo()
    conversation_summary_memory_demo()
