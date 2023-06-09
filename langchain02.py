#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


'''
通过langchain进行链式调用
支持多个参数
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    # 模型选用
    llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

    q1_prompt = PromptTemplate(
        input_variables=["year1"],
        template="{year1}年的欧冠联赛的冠军是哪支球队，只说球队名称。"
    )
    q2_prompt = PromptTemplate(
        input_variables=["year2"],
        template="{year2}年的欧冠联赛的冠军是哪支球队，只说球队名称。"
    )
    q3_prompt = PromptTemplate(
        input_variables=["team1", "team2"],
        template="{team1}和{team2}哪只球队获得欧冠的次数多一些？"
    )
    chain1 = LLMChain(llm=llm, prompt=q1_prompt, output_key="team1")
    chain2 = LLMChain(llm=llm, prompt=q2_prompt, output_key="team2")
    chain3 = LLMChain(llm=llm, prompt=q3_prompt)

    sequential_chain = SequentialChain(chains=[chain1, chain2, chain3], input_variables=["year1", "year2"], verbose=True)
    answer = sequential_chain.run(year1=2000, year2=2010)
    print(answer)

