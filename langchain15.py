#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


'''
通过智能代理实现ReAct响应模式
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI(temperature=0)

    # 加载Google搜索工具、数学计算工具
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # 初始化代理
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # 进行提问
    agent.run("目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")
