#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml

'''
通过ChatCompletion接口，实现聊天机器人功能
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

class Conversation:
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


if __name__ == '__main__':
    get_api_key()

    prompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内"""

    conv1 = Conversation(prompt, 2)
    question1 = "你是谁？"
    print("User : %s" % question1)
    print("Assistant : %s\n" % conv1.ask(question1))

    question2 = "请问鱼香肉丝怎么做？"
    print("User : %s" % question2)
    print("Assistant : %s\n" % conv1.ask(question2))

    question3 = "那蚝油牛肉呢？"
    print("User : %s" % question3)
    print("Assistant : %s\n" % conv1.ask(question3))

    question4 = "我问你的第一个问题是什么？"
    print("User : %s" % question4)
    print("Assistant : %s\n" % conv1.ask(question4))

    question5 = "你计算的时候上下文会追溯多久？"
    print("User : %s" % question5)
    print("Assistant : %s\n" % conv1.ask(question5))


'''
User : 你是谁？
Assistant : 我是一个AI语言模型，可以用中文回答你的疑问和问题。

User : 请问鱼香肉丝怎么做？
Assistant : 鱼香肉丝的做法如下：1.瘦肉切丝，加入盐、淀粉、料酒腌制10分钟；2.青红椒切丝，葱姜蒜切末；3.锅中倒油，放入肉丝煸炒至变色；4.加入青红椒丝、葱姜蒜末炒匀；5.加入酱油、糖、醋、盐、味精、水淀粉调味翻炒均匀即可。

User : 那蚝油牛肉呢？
Assistant : 蚝油牛肉的做法如下：1.牛肉切薄片，加入盐、生粉、料酒腌制10分钟；2.葱姜蒜切末，青红椒切丝备用；3.锅中倒油，放入牛肉煸炒至变色；4.加入葱姜蒜末、青红椒丝炒匀；5.加入蚝油、酱油、糖、水淀粉调味，翻炒均匀即可。

User : 我问你的第一个问题是什么？
Assistant : 你问我：“请问鱼香肉丝怎么做？”

User : 我们对话时，上下文会追溯多久？
Assistant : 我们对话时，上下文会追溯到我们开始交流的那一刻。也就是说，我们的每一次对话都是在之前对话的基础上展开的。
'''