#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI

'''
KGMemory通过知识图谱，提升对话效果
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    llm = OpenAI(temperature=0)

    memory = ConversationKGMemory(llm=llm)
    memory.save_context({"input": "say hi to sam"}, {"ouput": "who is sam"})
    memory.save_context({"input": "sam is a friend"}, {"ouput": "okay"})
    memory.load_memory_variables({"input": 'who is sam'})

    # 同时返回历史信息
    memory = ConversationKGMemory(llm=llm, return_messages=True)
    memory.save_context({"input": "say hi to sam"}, {"ouput": "who is sam"})
    memory.save_context({"input": "sam is a friend"}, {"ouput": "okay"})
    memory.load_memory_variables({"input": 'who is sam'})
    memory.get_current_entities("what's Sams favorite color?")
    memory.get_knowledge_triplets("her favorite color is red")
