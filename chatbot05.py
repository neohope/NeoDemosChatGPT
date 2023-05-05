#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
import gradio as gr

'''
通过Gradio增加了聊天机器人界面
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

class Conversation3:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": message})

        # 仅保留prompt 和 最新的几轮信息（每一轮一问一答）
        if len(self.messages) > self.num_of_round*2 + 1:
            del self.messages[1:3] 
        return message

# 通过chatgpt获取聊天反馈
def answer(question, history=[]):
    history.append(question)
    response = conv.ask(question)
    history.append(response)
    responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
    return responses, history


if __name__ == '__main__':
    get_api_key()

    prompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内"""

    conv = Conversation3(prompt, 10)

    with gr.Blocks(css="#chatbot{height:300px} .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

        txt.submit(answer, [txt, state], [chatbot, state])

    demo.launch()

