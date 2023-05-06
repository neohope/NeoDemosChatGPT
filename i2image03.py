#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from IPython.display import display

"""
通过宝宝手绘的草图，按描述生成图片
"""

if __name__ == '__main__':
    # 创建StableDiffusionImg2ImgPipeline
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    # pipeline = pipeline.to("cpu")
    pipeline = pipeline.to("cuda")

    # 载入初始图片
    image_file = "./data/ryder001.jpg"
    init_image = Image.open(image_file).convert("RGB")
    init_image = init_image.resize((852, 640))

    # 生成图片
    # prompt = "realism style, beautiful flowers on grassland"
    # prompt = "Chinese ink painting style, flowers in the forest"
    # prompt = "finger painting style, beautiful big flowers in the forest"
    prompt = "finger painting style, one boat and several beautiful flowers on grassland"
    negative_prompt = "building, water, human"
    images = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    images[0].save("./created/ryder001.jpg")
    display(images[0])
