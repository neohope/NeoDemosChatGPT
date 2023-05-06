#!/usr/bin/env python3
# -*- coding utf-8 -*-

import yaml, json, requests

"""
1、首先可以通过Inference API远程试用模型

可以在这里找到模型列表，其中带闪电标记的就可以用Inference API远程调用
https://huggingface.co/models

可以在这里找到access token，从而可以调用Inference API
https://huggingface.co/settings/tokens

Inference API技术文档
https://huggingface.co/docs/api-inference/detailed_parameters


2、对于效果较好的模型可以通过Inference Endpoint快速部署到AWS或Azure
https://ui.endpoints.huggingface.co/new

Public，互联网上的任何人都能调用
Protected，需要HuggingFace的Access Token
Private，不仅需要权限验证，还需要通过一个AWS或Azure的私有网络才能访问
"""

def get_hf_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        api_key = yaml_data["huggingface"]["api_key"]
        return api_key

def query(payload, api_url, headers):
    data = json.dumps(payload)
    response = requests.request("POST", api_url, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

if __name__ == '__main__':
    headers = {"Authorization": f"Bearer {get_hf_key()}", "Content-Type": "application/json"}

    # 知识问答
    model = "google/flan-t5-xxl"
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    question = "Please answer the following question. What is the capital of France?"
    data = query({"inputs" : question, "wait_for_model" : True}, api_url=api_url, headers=headers)
    print(data)

    # 文本转向量
    model = "hfl/chinese-pert-base"
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    question = "今天天气真不错！"
    data = query({"inputs" : question, "wait_for_model" : True}, api_url=api_url, headers=headers)
    print(data)

    # 测试部署好的GPT2模型
    api_url = "https://??????.us-east-1.aws.endpoints.huggingface.cloud"
    text = "My name is Lewis and I like to"
    data = query({"inputs" : text}, api_url=api_url, headers=headers)
    print(data)
