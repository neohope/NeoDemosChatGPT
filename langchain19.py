#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

'''
利用arxiv检索论文信息
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()

    # 初始化模型和工具
    llm = ChatOpenAI(temperature=0.0)
    tools = load_tools(
        ["arxiv"],
    )

    # 初始化链
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # 运行链
    agent_chain.run("介绍一下2005.14165这篇论文的创新点?")
