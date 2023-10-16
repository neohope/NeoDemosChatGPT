#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
import asyncio
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

# 进行更多的异步交互和token计数
async def additional_interactions():
    with get_openai_callback() as cb:
        await asyncio.gather(
            *[llm.agenerate(["我姐姐喜欢什么颜色的花？"]) for _ in range(3)]
        )
    print("\n另外的交互中使用的tokens:", cb.total_tokens)

if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI(temperature=0.5, model_name="text-davinci-003")

    # 初始化对话链
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory()
    )

    # 异步调用
    asyncio.run(additional_interactions())
