#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
import matplotlib.pyplot as plt
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler

"""
用ControlNet获取图像的动作轮廓
然后生成超级英雄的图像
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
    # 加载模型用于获取动作轮廓
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    # 思考者”和的动作轮廓
    image_file1 = "./data/rodin.jpg"
    original_image1 = load_image(image_file1)
    openpose_image1 = openpose(original_image1)

    # “掷铁饼者”的动作轮廓
    image_file2 = "./data/discobolos.jpg"
    original_image2 = load_image(image_file2)
    openpose_image2 = openpose(original_image2)

    # 展示结果
    images = [original_image1, openpose_image1, original_image2, openpose_image2]
    draw_image_grids(images, 2, 2)

    # 加载模型用于生成图像
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
  
    # 加速图像生成
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # 显存不够用时，将模型从显存中移除
    pipe.enable_model_cpu_offload()
    # 通过xformers加速推理
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cpu")

    # 生成沉思蝙蝠侠
    # poses = [openpose_image1]
    # generator = [torch.Generator(device="cpu").manual_seed(42)]
    # prompt1 = "batman character, best quality, extremely detailed"
    # output = pipe(
    #     [prompt1],
    #     poses,
    #     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
    #     generator=generator,
    #     num_inference_steps=20,
    # )
    # draw_image_grids(output.images, 1, 1)

    # 生成沉思蝙蝠侠，铁饼蝙蝠侠，沉思蜘蛛侠，铁饼蜘蛛侠
    poses = [openpose_image1, openpose_image2, openpose_image1, openpose_image2]
    generator = [torch.Generator(device="cpu").manual_seed(42) for i in range(4)]
    prompt1 = "batman character, best quality, extremely detailed"
    prompt2 = "ironman character, best quality, extremely detailed"
    output = pipe(
        [prompt1, prompt1, prompt2, prompt2],
        poses,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        generator=generator,
        num_inference_steps=20,
    )
    draw_image_grids(output.images, 2, 2)
