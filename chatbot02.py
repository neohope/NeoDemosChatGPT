#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml

'''
一个命令行版本的聊天机器人
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


def ask_gpt3(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message


def generate_prompt(prompt, questions, answers):
    num = len(answers)
    for i in range(num):
        prompt += "\n Q : " + questions[i]
        prompt += "\n A : " + answers[i]
    prompt += "\n Q : " + questions[num] + "\n A : "        
    return prompt


if __name__ == '__main__':
    get_api_key()
        
    questions = []
    answers = []
    print("你好，我是一个聊天机器人，请你提出你的问题吧?")

    while True:
        user_input = input("> ")
        questions.append(user_input)
        if user_input.lower() in ["bye", "goodbye", "exit"]:
            print("Goodbye!")
            break
        
        prompt = generate_prompt("", questions, answers)

        answer = ask_gpt3(prompt)
        print(answer)
        answers.append(answer)


'''
你好，我是一个聊天机器人，请你提出你的问题吧?
> 我想学习编程，可以给我推荐几本书吗？
你可以查阅不同的编程书籍来学习编程，也可以通过网络资源来学习编程。
> 如果是我想学习Python呢？哪些书比较合适？
《Python编程：从入门到实践》、《Python Cookbook》、《Python编程：快速上手-让繁琐工作自动化》、《Python核心编程》等都是不错的选择。
> 如果是想学习chatgpt呢？
你可以参考《Chatbot Development with Python》，这本书涵盖了Chatbot开发的基础知识，介绍了如何使用Python来实现Chatbot功能。
> activemq、rabbitmq、rocketmq之间有什么区别呢？
三者都是消息中间件，但是它们有一些明显的差异。ActiveMQ是一个完整的消息中间件，支持JMS，支持多种传输协议；RabbitMQ是一个消息队列，支持AMQP协议；RocketMQ是一个 分布式消息中间件，支持分布式系统的消息通信。
> rocketmq与kafka有什么区别呢
两者都是消息中间件，但是它们有一些明显的差异。RocketMQ是一个分布式消息中间件，支持分布式系统的消息通信，而Kafka是一个分布式流处理平台，支持流处理和消息通信。
> 可以说的再详细一些吗？
从功能上来说，RocketMQ主要用于消息发布订阅、消息持久化、消息路由等，而Kafka主要用于数据流处理、日志收集和消息路由等。从性能上来说，RocketMQ拥有更高的吞吐量和 更快的消息传递速度，而Kafka拥有更好的可用性和更高的容错性。
> bye
Goodbye!
'''