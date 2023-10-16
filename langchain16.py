#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
import asyncio
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import create_async_playwright_browser

'''
通过playwright作为网页查询工具
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


async def web_query():
    response = await agent_chain.arun("What are the headers on python.langchain.com?")
    print(response)


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = ChatOpenAI(temperature=0.5)  

    # 展示工具清单
    async_browser = create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    print(tools)

    # 初始化chain
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(web_query())
