#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
from diffusers import DiffusionPipeline
from IPython.display import display

"""
使用 Stable Diffusion 生成图片

整个 Stable Diffusion 文生图的过程是由这样三个核心模块组成的：
第一个模块是一个 Text-Encoder，把我们输入的文本变成一个向量
第二个模块是 Generation 模块，顾名思义是一个图片信息生成模块
第三个模块是 Decoder 或者叫做解码器，会根据第二步的返回结果把这个图像信息还原成最终的图片
"""

if __name__ == '__main__':
    # 几种方式GPU内存不足的设置，用钞能力解决问题其实效果更好
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    # torch.cuda.empty_cache()

    with torch.no_grad():
        pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipeline.to("cpu")
        # pipeline.to("cuda")
        # image = pipeline("a photograph of an astronaut riding a horse").images[0]
        image = pipeline("an astronaut riding a horse").images[0]
        image.save("./created/astronaut_riding_horse.jpg")
        display(image)
