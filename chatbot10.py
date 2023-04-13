#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

'''
EntityMemory通过命名实体识别的方式，提升对话效果
实体的内容可以放到redis中
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    llm = OpenAI()
    entityMemory = ConversationEntityMemory(llm=llm)
    # 命名实体会被放到Context中
    conversation = ConversationChain(
        llm=llm, 
        verbose=True,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=entityMemory
    )

    answer=conversation.predict(input="我叫张老三，在你们这里下了一张订单，订单号是 2023ABCD，我的邮箱地址是 customer@abc.com，但是这个订单十几天了还没有收到货")
    print(answer)

    # 输出实体和上下文
    print(conversation.memory.entity_store.store)

    answer=conversation.predict(input="我刚才的订单号是多少？")
    print(answer)

    answer=conversation.predict(input="订单2023ABCD是谁的订单？")
    print(answer)
