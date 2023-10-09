#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain import OpenAI, SerpAPIWrapper 
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

'''
Self-Ask with Search代理
解决多跳问题
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI(temperature=0)

    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer", 
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]

    self_ask_with_search = initialize_agent(
        tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
    )
    self_ask_with_search.run(
        "使用玫瑰作为国花的国家的首都是哪里?"  
    )
