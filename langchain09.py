#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os, openai, yaml, logging
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from flask import Flask, request, render_template

'''
客服FAQ系统
加载硬盘上的资料，分割后，放入向量数据库
使用RetrievalQA，结合chatgpt和向量数据库的知识，给予客户答复

其中MultiQueryRetriever，会根据输入问题，让chatgpt生成多个相似问题
然后通过这些问题，到向量数据库中检索答案，并返回去重后的文档
最后通过chatgpt，整合检索答案，给出反馈
'''

# 初始化openai的密钥
def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

# 加载文档
def load_docs(base_dir):
    documents = []
    for file in os.listdir(base_dir): 
        # 构建完整的文件路径
        file_path = os.path.join(base_dir, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

# 加载文档到向量数据库
def init_vectorstore(doc_dir):
    # 枚举文档
    documents = load_docs(doc_dir)

    # 将Documents切分成块以便后续进行嵌入和向量存储
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    # 将分割嵌入并存储在矢量数据库Qdrant中
    vectorstore = Qdrant.from_documents(
        documents=chunked_documents, # 以分块的文档
        embedding=OpenAIEmbeddings(), # 用OpenAI的Embedding Model做嵌入
        location=":memory:",  # in-memory 存储
        collection_name="my_documents",) # 指定collection_name

    return vectorstore

# 基于flask的问答页面 
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')        
        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})
        # 把大模型的回答结果返回网页进行渲染
        return render_template('langchain09.html', result=result)
    return render_template('langchain09.html')


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    # 载入数据到向量数据库
    vectorstore = init_vectorstore("data/langchan09")

    # 设置OpenAI的API密钥
    get_api_key()
    # 实例化一个大模型工具
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # 实例化一个MultiQueryRetriever
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
    # 实例化一个RetrievalQA链
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

    # 启动flask
    # 访问 http://127.0.0.1:5000/
    app.run(host='0.0.0.0',debug=True,port=5000)
