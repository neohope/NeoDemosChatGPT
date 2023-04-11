#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain


'''
通过langchain进行链式调用
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


def with_no_chain(question_translate_chain, qa_chain, answer_translate_chain):
    english = question_translate_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
    print(english)

    english_answer = qa_chain.run(english_question=english)
    print(english_answer)

    answer = answer_translate_chain.run(english_answer=english_answer)
    print(answer)


def with_chain():
    chinese_qa_chain = SimpleSequentialChain(
        chains=[question_translate_chain, qa_chain, answer_translate_chain], input_key="question",
        verbose=True)
    answer = chinese_qa_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")

    return answer

if __name__ == '__main__':
    get_api_key()

    # 模型选用
    llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

    # 一个LLMChain里，所使用的PromptTemplate里的输入参数，之前必须在LLMChain里，通过output_key定义过
    en_to_zh_prompt = PromptTemplate(
        template="请把下面这句话翻译成英文： \n\n {question}?", input_variables=["question"]
    )

    question_prompt = PromptTemplate(
        template = "{english_question}", input_variables=["english_question"]
    )

    zh_to_cn_prompt = PromptTemplate(
        input_variables=["english_answer"],
        template="请把下面这一段翻译成中文： \n\n{english_answer}?",
    )

    question_translate_chain = LLMChain(llm=llm, prompt=en_to_zh_prompt, output_key="english_question")
    qa_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="english_answer")
    answer_translate_chain = LLMChain(llm=llm, prompt=zh_to_cn_prompt)
    

    with_no_chain(question_translate_chain, qa_chain, answer_translate_chain)
    
    answer = with_chain(question_translate_chain, qa_chain, answer_translate_chain)
    print(answer)

