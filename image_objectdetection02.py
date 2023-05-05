#!/usr/bin/env python3
# -*- coding utf-8 -*-

import cv2
from matplotlib import pyplot as plt
from transformers import pipeline

"""
使用OWL-ViT模型进行目标检测
然后使用opencv对图片进行标注
"""

if __name__ == '__main__':

    # 加载模型
    detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")

    # 进行检测
    detected = detector(
        "./data/cat.jpg",
        candidate_labels=["cat", "dog", "truck", "couch", "remote"],
    )

    # opencv读入文件
    image_path = "./data/cat.jpg"
    image = cv2.imread(image_path)

    # 转换色彩空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 对物体边缘画框，并进行文字标注
    for detection in detected:
        box = detection['box']
        label = detection['label']
        score = detection['score']
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 展示标注后的图片
    plt.imshow(image)
    plt.axis('off')
    plt.show()
