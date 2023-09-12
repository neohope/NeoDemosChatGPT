#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

'''
根据输入的图片 URL
首先利用图像字幕生成工具将图片转化为文字描述
然后对图片文字描述做进一步处理，生成中文推广文案
'''

# 初始化openai的密钥
def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

# 指定要使用的工具模型，初始化处理器和工具模型
hf_model = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(hf_model)
model = BlipForConditionalGeneration.from_pretrained(hf_model)

# 图像字幕生成工具类
class ImageCapTool(BaseTool):
    name = "Image captioner"
    description = "为图片创作说明文案"

    def _run(self, url: str):
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


if __name__ == '__main__':
    # 设置OpenAI的API密钥并初始化大语言模型
    get_api_key()
    llm = OpenAI(temperature=0.2)

    # 使用工具初始化智能代理并运行它
    tools = [ImageCapTool()]
    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
    )
    img_url = 'http://static.neohope.com/lm/kindergarten01.jpg'
    agent.run(input=f"{img_url}\n请给出合适的中文文案")
