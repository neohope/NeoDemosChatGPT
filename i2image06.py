#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

"""
以一个简单的简笔画为基础，生成精美的图片
"""

def draw_image_grids(images, rows, cols):
    # Create a rows x cols grid for displaying the images
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for row in range(rows):
      for col in range(cols):
        axes[row, col].imshow(images[col + row * cols])
    for ax in axes.flatten():
        ax.axis('off')
    # Display the grid
    plt.show()


if __name__ == '__main__':
    # 加载模型
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
  
    # 显存不够用时，将模型从显存中移除
    pipe.enable_model_cpu_offload()
    # 通过xformers加速推理
    pipe.enable_xformers_memory_efficient_attention()

    # 加载狗狗简笔画
    image_file = "./data/scribble_dog.png"
    scribble_image = load_image(image_file)

    # 生成四种狗狗图片
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(4)]
    prompt = "dog"
    prompt = [prompt + t for t in [" in a room", " near the lake", " on the street", " in the forest"]]
    output = pipe(
        prompt,
        scribble_image,
        negative_prompt=["lowres, bad anatomy, worst quality, low quality"] * 4,
        generator=generator,
        num_inference_steps=50,
    )

    draw_image_grids(output.images, 2, 2)
