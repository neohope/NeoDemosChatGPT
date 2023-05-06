What is this project about
=========
This is just a project of ChatGPT3.5 demos. 


How to build
============
1. install python 3.10+


2. update pip
```shell
    python -m pip install --upgrade pip
```

3. install packages
```shell
    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
    python test01.py
```

4. run
```shell
    python -m venv venv
    python test01.py
```

5. 要匹配cuda版本及pytorch版本
```shell
# 查询cuda版本及pytorch版本
https://pytorch.org/get-started/locally/
https://developer.nvidia.com/cuda-toolkit-archive

# 下载所需cuda，比如118

# 安装对应版本的pytorch
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 测试一下
import torch
print(torch.cuda.is_available())
```

Reference
=========
