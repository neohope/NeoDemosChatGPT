#!/usr/bin/env python3
# -*- coding utf-8 -*-

from transformers import AutoTokenizer, AutoModel

"""
通过ChatGLM模型，根据提示，实现问答

模型安装：
pip install icetk
pip install cpm_kernels
"""

if __name__ == '__main__':

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
    # 加载GPU模型
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
    # 加载CPU模型
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4",trust_remote_code=True).float()
    model = model.eval()

    # 根据输入进行作答
    question = """
    自收到商品之日起7天内，如产品未使用、包装完好，您可以申请退货。某些特殊商品可能不支持退货，请在购买前查看商品详情页面的退货政策。

    根据以上信息，请回答下面的问题：

    Q: 你们的退货政策是怎么样的？
    """
    response, history = model.chat(tokenizer, question, history=[])
    print(response)

    # 没有提示情况下，会创造答案
    question = """
    Q: 你们的退货政策是怎么样的？
    A: 
    """
    response, history = model.chat(tokenizer, question, history=[])
    print(response)

    # 模型是否能理解城市和省的概念
    question = """
    我们支持全国大部分省份的配送，包括北京、上海、天津、重庆、河北、山西、辽宁、吉林、黑龙江、江苏、浙江、安徽、福建、江西、山东、河南、湖北、湖南、广东、海南、四川、贵州、云南、陕西、甘肃、青海、台湾、内蒙古、广西、西藏、宁夏和新疆.

    根据以上信息，请回答下面的问题：

    Q: 你们能配送到三亚吗？
    """
    response, history = model.chat(tokenizer, question, history=[])
    print(response)

    question = """
    我们支持全国大部分省份的配送，包括北京、上海、天津、重庆、河北、山西、江苏、浙江、安徽、福建、江西、山东、河南、湖北、湖南、广东、海南、四川、贵州、云南、陕西、甘肃、青海、台湾、内蒙古、广西、西藏、宁夏和新疆.但是不能配送到东三省

    根据以上信息，请回答下面的问题：

    Q: 你们能配送到哈尔滨吗？
    """
    response, history = model.chat(tokenizer, question, history=[])
    print(response)
