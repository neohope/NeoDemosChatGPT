#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader

'''
通过VectorDBQA，集成本地FAISS向量库
可以实现FAQ的功能
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    llm = OpenAI(temperature=0)

    # 加载文本
    loader = TextLoader('./data/ecommerce_faq.txt')
    documents = loader.load()

    # 文本文段
    text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
    texts = text_splitter.split_documents(documents)

    # 通过chatgpt建立向量，并存入FAISS
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)
    
    # faq chain
    faq_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=docsearch, verbose=True)

    # 查询
    question = "请问你们的货，能送到三亚吗？大概需要几天？"
    result = faq_chain.run(question)
    print(result)

    question = "请问你们的退货政策是怎么样的？" 
    result = faq_chain.run(question)
    print(result)
