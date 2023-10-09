#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

'''
利用api查询gmail信息
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = ChatOpenAI(temperature=0, model='gpt-4')

    # 初始化Gmail工具包
    toolkit = GmailToolkit()

    # 获取Gmail API的凭证，并指定相关的权限范围
    credentials = get_gmail_credentials(
        token_file="token.json",  # Token文件路径
        scopes=["https://mail.google.com/"],  # 具有完全的邮件访问权限
        client_secrets_file="credentials.json",  # 客户端的秘密文件路径
    )

    # 使用凭证构建API资源服务
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)

    # 获取工具
    tools = toolkit.get_tools()
    print(tools)

    # 通过指定的工具和聊天模型初始化agent
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )

    # 使用agent运行一些查询或指令
    result = agent.run(
        "今天易速鲜花客服给我发邮件了么？最新的邮件是谁发给我的？"
    )

    # 打印结果
    print(result)
