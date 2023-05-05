#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
from PIL import Image
from IPython.display import display
from IPython.display import Image as IPyImage
from transformers import CLIPProcessor, CLIPModel

"""
多模态的CLIP模型，可以同时支持图片和文本
这里演示一下，多模态模型中，如何关联图片和文本
"""

# 获取图片特征
def get_image_feature(filename: str):
    image = Image.open(filename).convert("RGB")
    processed = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=processed["pixel_values"])
    return image_features

# 获取文本特征
def get_text_feature(text: str):
    processed = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(processed['input_ids'])
    return text_features

# 获取相关性
def cosine_similarity(tensor1, tensor2):
    tensor1_normalized = tensor1 / tensor1.norm(dim=-1, keepdim=True)
    tensor2_normalized = tensor2 / tensor2.norm(dim=-1, keepdim=True)
    return (tensor1_normalized * tensor2_normalized).sum(dim=-1)

if __name__ == '__main__':
    # 加载clip-vit-base-patch32模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 加载了一张猫的图片
    image_tensor = get_image_feature("./data/cat.jpg")
    display(IPyImage(filename='./data/cat.jpg'))

    # 加载了五段文字：一只猫、一只狗、两只猫、一辆卡车、一个沙发
    cat_text = "This is a cat."
    cat_text_tensor = get_text_feature(cat_text)

    dog_text = "This is a dog."
    dog_text_tensor = get_text_feature(dog_text)

    two_cats_text = "There are two cats."
    two_cats_text_tensor = get_text_feature(two_cats_text)

    truck_text = "This is a truck."
    truck_text_tensor = get_text_feature(truck_text)

    couch_text = "This is a couch."
    couch_text_tensor = get_text_feature(couch_text)

    # 计算图片与五段文字相关性
    print("Similarity with cat : ", cosine_similarity(image_tensor, cat_text_tensor))
    print("Similarity with dog : ", cosine_similarity(image_tensor, dog_text_tensor))
    print("Similarity with two cats : ", cosine_similarity(image_tensor, two_cats_text_tensor))
    print("Similarity with truck : ", cosine_similarity(image_tensor, truck_text_tensor))
    print("Similarity with couch : ", cosine_similarity(image_tensor, couch_text_tensor))


"""
输出：
Similarity with cat :  tensor([0.2482])            # 图片中有猫
Similarity with dog :  tensor([0.2080])            # 图片中有宠物
Similarity with two cats :  tensor([0.2723])       # 图片中有两只猫
Similarity with truck :  tensor([0.1814])          # 最不相关
Similarity with couch :  tensor([0.2376])          # 图片中有沙发
"""
