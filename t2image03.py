#!/usr/bin/env python3
# -*- coding utf-8 -*-

import torch
from diffusers import DiffusionPipeline

"""
可以在这里查找训练好的模型，而且不少模型可以在Huggingface中使用
https://civitai.com/

可以使用Stable-Diffusion-Web-UI在本地进行模型部署，并直接通过web访问
https://github.com/AUTOMATIC1111/stable-diffusion-webui
"""

if __name__ == '__main__':
    # 创建DiffusionPipeline
    model_id_or_path = "gsdf/Counterfeit-V3.0"
    pipeline = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipeline.to("cuda")

    # 根据提示生成二次元图像
    prompt = "((masterpiece,best quality)), 1girl, solo, animal ears, rabbit, barefoot, knees up, dress, sitting, rabbit ears, short sleeves, looking at viewer, grass, short hair, smile, white hair, puffy sleeves, outdoors, puffy short sleeves, bangs, on ground, full body, animal, white dress, sunlight, brown eyes, dappled sunlight, day, depth of field"
    negative_prompt = "EasyNegative, extra fingers,fewer fingers,"
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt).images[0]
    image
