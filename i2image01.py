#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from IPython.display import display

"""
通过草图，按描述生成图片
"""

if __name__ == '__main__':
    # 创建StableDiffusionImg2ImgPipeline
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    # 载入初始图片
    image_file = "./data/sketch-mountains-input.jpg"
    init_image = Image.open(image_file).convert("RGB")
    init_image = init_image.resize((768, 512))

    # 1、要生成的图片描述
    prompt = "A fantasy landscape, trending on artstation"
    images = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    display(init_image)
    display(images[0])

    # 2、改变要生成图片的描述，用宫崎骏风格+城堡
    prompt = "ghibli style, a fantasy landscape with castles"
    images = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    display(init_image)
    display(images[0])

    # 3、在2的基础上，排除河流
    prompt = "ghibli style, a fantasy landscape with castles"
    negative_prompt = "river"
    images = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    display(images[0])
