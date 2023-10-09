#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain

'''
Plan and execute代理
计划和执行不再是由同一个代理

pip install -U langchain langchain_experimental
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI(temperature=0)

    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
    ]

    model = ChatOpenAI(temperature=0)
    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)

    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

    agent.run("在纽约，100美元能买几束玫瑰?")
