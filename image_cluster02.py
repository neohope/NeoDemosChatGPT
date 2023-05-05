#!/usr/bin/env python3
# -*- coding utf-8 -*-

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

"""

"""

if __name__ == '__main__':
    # 加载clip-vit-base-patch32模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 加载了一张猫的图片
    image_file = "./data/cat.jpg"
    image =  Image.open(image_file)

    # 用文本标识分成四类
    categories = ["cat", "dog", "truck", "couch"]
    categories_text = list(map(lambda x: f"a photo of a {x}", categories))

    # 用softmax算法来计算图片分类到某一类的概率
    inputs = processor(text=categories_text, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1) 

    # 输出概率
    for i in range(len(categories)):
        print(f"{categories[i]}\t{probs[0][i].item():.2%}")


"""
输出：
cat  74.51%         # 图片中有两只猫
dog  0.39%
truck  0.04%
couch  25.07%       # 图片中有沙发
"""
