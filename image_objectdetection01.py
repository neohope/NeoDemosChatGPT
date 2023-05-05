#!/usr/bin/env python3
# -*- coding utf-8 -*-
from transformers import pipeline

"""
使用OWL-ViT模型进行目标检测
"""

if __name__ == '__main__':
    # 加载模型
    detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")

    # 进行检测
    detected = detector(
        "./data/cat.jpg",
        candidate_labels=["cat", "dog", "truck", "couch", "remote"],
    )

    # 输出检测到的物体
    print(detected)


"""
输出：
[{'score': 0.2868116796016693, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, 
{'score': 0.2770090401172638, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 72, 'xmax': 177, 'ymax': 115}}, 
{'score': 0.2537277638912201, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, 
{'score': 0.14742951095104218, 'label': 'remote', 'box': {'xmin': 335, 'ymin': 74, 'xmax': 371, 'ymax': 187}}, 
{'score': 0.12083035707473755, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]
"""
