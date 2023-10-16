#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

'''
利用sql agent，将自然语言自动生成sql，查询数据库，并给出自然语言结果
pip install langchain-experimental
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI(temperature=0, verbose=True)

    # 连接到FlowerShop数据库
    db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

    # 创建SQL Agent
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # 使用Agent执行SQL查询

    questions = [
        "哪种鲜花的存货数量最少？",
        "平均销售价格是多少？",
    ]

    for question in questions:
        response = agent_executor.run(question)
        print(response)
