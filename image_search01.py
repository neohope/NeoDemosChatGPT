#!/usr/bin/env python3
# -*- coding utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import faiss
import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

"""
OpenAI开源的CLIP模型，是通过互联网上的海量图片数据，以及图片对应的img标签里面的alt和title字段信息训练出来的
通过文本查找图片
"""

# 展示图片
def display_images(images):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

# 获取图片特征
def get_image_features(image):
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs.to(device)
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()

# 对每个图片，获得图片特征，并保存到features中
def add_image_features(example):
    example["features"] = get_image_features(example["image"])
    return example

# 获取文本特征
def get_text_features(text):
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs.to(device)
        features = model.get_text_features(**inputs)
    return features.cpu().numpy()

# 通过文本查找图片
def search(query_text, top_k=5):
    # 先获取文本特征
    text_features = get_text_features(query_text)

    # 通过faiss找到最接近的图片特征及距离
    distances, indices = index.search(text_features.astype("float32"), top_k)

    # 获取图片及距离
    results = [
        {"image": training_split[i]["image"], "distance": distances[0][j]}
        for j, i in enumerate(indices[0])
    ]

    return results

# 展示查询到的图片
def display_search_results(results):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    axes = axes.ravel()

    for idx, result in enumerate(results):
        axes[idx].imshow(result["image"])
        axes[idx].set_title(f"Distance: {result['distance']:.2f}")
        axes[idx].axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


if __name__ == '__main__':
    # 加载数据集
    dataset = load_dataset("rajuptvs/ecommerce_products_clip")

    # 数据集一般分为分成训练集（train）、验证集（validation）和测试集（test）
    # 这里选用了训练集
    training_split = dataset["train"]

    # 展示前十张图片
    images = [example["image"] for example in training_split.select(range(10))]
    display_images(images)

    # 优先使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载clip-vit-base-patch32模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 调用add_image_features
    training_split = training_split.map(add_image_features)

    # 图片向量加载到faiss中
    features = [example["features"] for example in training_split]
    features_matrix = np.vstack(features)
    dimension = features_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(features_matrix.astype('float32'))

    # 通过文本，查找图片
    query_text = "A red dress"
    results = search(query_text)

    # 展示查询到的图片
    display_search_results(results)
