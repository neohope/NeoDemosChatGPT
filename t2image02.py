#!/usr/bin/env python3
# -*- coding utf-8 -*-

import PIL
import numpy as np
import torch
from PIL import Image
from IPython.display import display
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm

"""
模拟 Stable Diffusion 生成图片
"""

# 展示生成的图片信息
def display_denoised_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Denoised Sample @ Step {i}")
    display(image_pil)
    return image_pil

# 通过VAE模型，对图片进行扩展和解码，并展示最终解码后的图片
def display_decoded_image(latents, i):
  # scale and decode the image latents with vae
  latents = 1 / 0.18215 * latents
  with torch.no_grad():
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    display(f"Decoded Image @ step {i}")
    display(pil_images[0])
    return pil_images[0]

if __name__ == '__main__':
    # 加载需要的模型组件
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # 超参数设置
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 25  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(42)  # Seed generator to create the inital latent noise

    # 通过tokenizer和text_encoder将文本向量化
    prompt = ["a photograph of an astronaut riding a horse"]
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # 创造一个无条件的空向量，作为初始向量
    batch_size = len(prompt)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    # 图像生成的过程，就是从初始空向量，到目标向量的过程
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 生成一系列随机噪声
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # 一共会迭代25次
    # 每5次迭代保存一次结果
    denoised_images = []
    decoded_images = []
    # 使用unet和scheduler进行噪声去除，从空向量到指定文本向量对应图片的过程
    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 在做无分类器修正（事前训练）时，为了避免两次前向传播的影响，需要增强噪声
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # 用unet预测噪声残留
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # 修正噪声
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 计算之前的噪声采样 x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 每5次迭代展示一次结果，并进行保存
        if i % 5 == 0:
            denoised_image = display_denoised_sample(latents, i)
            decoded_image = display_decoded_image(latents, i)
            denoised_images.append(denoised_image)
            decoded_images.append(decoded_image)

    # 查看生成图片的尺寸
    # 生成的图像 torch.Size([1, 4, 64, 64])
    # 解码后图像 torch.Size([1, 3, 512, 512])
    print(latents.shape)
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
        print(image.shape)
