#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from loguru import logger
from langchain.callbacks import FileCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

'''
回调示例
实现日志记录
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    # 初始化模型
    get_api_key()
    llm = OpenAI()
    prompt = PromptTemplate.from_template("1 + {number} = ")

    logfile = "output.log"
    logger.add(logfile, colorize=True, enqueue=True)
    handler = FileCallbackHandler(logfile)

    # this chain will both print to stdout (because verbose=True) and write to 'output.log'
    # if verbose=False, the FileCallbackHandler will still write to 'output.log'
    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler], verbose=True)
    answer = chain.run(number=2)
    logger.info(answer)
