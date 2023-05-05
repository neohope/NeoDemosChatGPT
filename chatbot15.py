#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os, yaml
import openai
import time, requests
import gradio as gr
from gradio import HTML
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

'''
使用ChatGPT和Gradio构建聊天机器人
增加语音输入的方式
增加d-id的虚拟形象视频输出聊天方式 https://studio.d-id.com/
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

def get_did_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        api_key = yaml_data["did"]["api_key"]
        return api_key

# 生成视频
def generate_talk(input, avatar_url, 
                  voice_type = "microsoft", 
                  voice_id = "zh-CN-YunyeNeural", 
                  api_key = get_did_key()):
    url = "https://api.d-id.com/talks"
    payload = {
        "script": {
            "type": "text",
            "provider": {
                "type": voice_type,
                "voice_id": voice_id
            },
            "ssml": "false",
            "input": input
        },
        "config": {
            "fluent": "false",
            "pad_audio": "0.0"
        },
        "source_url": avatar_url
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Basic " + api_key
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# 获取生成结果
def get_a_talk(id, api_key = get_did_key()):
    url = "https://api.d-id.com/talks/" + id
    headers = {
        "accept": "application/json",
        "authorization": "Basic "+api_key
    }
    response = requests.get(url, headers=headers)
    return response.json()

# 生成并获取mp4视频地址
def get_mp4_video(input, avatar_url):
    # 生成视频
    response = generate_talk(input=input, avatar_url=avatar_url)
    # 获取生成结果
    talk = get_a_talk(response['id'])
    # 轮询获取视频地址，会尝试30次
    # 应该采用webhook，而不是轮询方式
    video_url = ""
    index = 0
    while index < 30:
        index += 1
        if 'result_url' in talk:    
            video_url = talk['result_url']
            return video_url
        else:
            time.sleep(1)
            talk = get_a_talk(response['id'])
    return video_url

def predict(input, history=[]):
    if input is not None:
        history.append(input)
        # 获取聊天反馈
        response = conversation.predict(input=input)
        # 生成聊天反馈视频
        video_url = get_mp4_video(input=response, avatar_url=avatar_url)
        # 输出结果
        video_html = f"""<video width="320" height="240" controls autoplay><source src="{video_url}" type="video/mp4"></video>"""
        history.append(response)
        responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
        return responses, video_html, history
    else:
        # 返回静态图片
        video_html = f'<img src="{avatar_url}" width="320" height="240" alt="John Carmack">'
        responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
        return responses, video_html, history        

# 语音转文本
def transcribe(audio):
    os.rename(audio, audio + '.wav')
    audio_file = open(audio + '.wav', "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt="这是一段简体中文的问题。")
    return transcript['text']    

# 语音先转文本，然后调用predict方法
def process_audio(audio, history=[]):
    if audio is not None:
        text = transcribe(audio)
        return predict(text, history)
    else:
        text = None
        return predict(text, history)


if __name__ == '__main__':
    get_api_key()
    avatar_url = "https://cdn.discordapp.com/attachments/1065596492796153856/1095617463112187984/John_Carmack_Potrait_668a7a8d-1bb0-427d-8655-d32517f6583d.png"

    # 通过SummaryBufferMemory保留之前聊天的语境
    memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)
    conversation = ConversationChain(
        llm=OpenAI(max_tokens=2048, temperature=0.5), 
        memory=memory,
    )

    # 页面设计
    with gr.Blocks(css="#chatbot{height:500px} .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        # 文本录入
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

        # 语音录入
        with gr.Row():
            audio = gr.Audio(source="microphone", type="filepath")

        # 视频输出
        with gr.Row():
            video = gr.HTML(f'<img src="{avatar_url}" width="320" height="240" alt="John Carmack">', live=False)

        # 文本关联predict方法
        txt.submit(predict, [txt, state], [chatbot, video, state])

        # 语音关联process_audio方法
        audio.change(process_audio, [audio, state], [chatbot, video, state])

    # 启动 
    demo.launch()
