#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
from PIL import Image
from IPython.display import display
from diffusers import StableDiffusionUpscalePipeline

"""
通过模糊的图片，生成一张清晰的图片
"""

if __name__ == '__main__':
    # 加载模型，创建StableDiffusionUpscalePipeline
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")

    # 使用一张模糊的图片
    low_res_img_file = "./data/low_res_cat.png"
    low_res_img = Image.open(low_res_img_file).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    # 通过提示语，生成一张清晰的图片
    prompt = "a white cat"
    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    low_res_img_resized = low_res_img.resize((512, 512))

    display(low_res_img_resized)
    display(upscaled_image)
