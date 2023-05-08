#!/usr/bin/env python3
# -*- coding utf-8 -*-

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

"""
使用ControlNet增强图像生成的稳定性
先用opencv获取头部轮廓，然后把图片换成明星脸
"""

def get_canny_image(original_image, low_threshold=100, high_threshold=200):
  image = np.array(original_image)

  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  canny_image = Image.fromarray(image)
  return canny_image

def display_images(image1, image2):
  # Combine the images horizontally
  combined_image = Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height)))
  combined_image.paste(image1, (0, 0))
  combined_image.paste(image2, (image1.width, 0))
  # Display the combined image
  plt.imshow(combined_image)
  plt.axis('off')
  plt.show()


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
    # 使用opencv获取“戴珍珠耳环的少女”中的头像
    image_file = "data/input_image_vermeer.png"
    original_image = load_image(image_file)
    canny_image = get_canny_image(original_image)
    display_images(original_image, canny_image)

    # 加载模型
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    # 显存不够用时，将模型从显存中移除
    pipe.enable_model_cpu_offload()
    # 通过xformers加速推理
    pipe.enable_xformers_memory_efficient_attention()

    # 同时生成四张图片，根据四位女星
    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Audrey Hepburn", "Elizabeth Taylor", "Scarlett Johansson", "Taylor Swift"]]
    generator = [torch.Generator(device="cpu").manual_seed(42) for i in range(len(prompt))]

    # 计算并展示生成的四张图片
    output = pipe(
        prompt,
        canny_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
    )
    draw_image_grids(output.images, 2, 2)
