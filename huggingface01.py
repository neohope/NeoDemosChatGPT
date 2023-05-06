#!/usr/bin/env python3
# -*- coding utf-8 -*-

from transformers import pipeline
from transformers import WhisperProcessor

"""
可以在这里找到模型列表：
https://huggingface.co/models


模型支持的任务类型有：
feature-extraction，文本向量
fill-mask，完形填空
ner，命名实体识别
question-answering和table-question-answering，自动问答
text-generation 和 text2text-generation，文本生成
summarization，文本摘要
translation，机器翻译
sentiment-analysis， 情感分析
text-classification，文本分类
zero-shot-classification，零样本分类


pipeline可以做的任务：
https://huggingface.co/docs/transformers/main_classes/pipelines
"""

if __name__ == '__main__':
    # 使用默认模型，只支持英文
    # device=0使用GPU
    classifier = pipeline(task="sentiment-analysis", device=0)
    preds = classifier("I am really happy today!")
    print(preds)

    # 指定中文模型，进行情感分析
    classifier = pipeline(model="uer/roberta-base-finetuned-jd-binary-chinese", task="sentiment-analysis", device=0)
    preds = classifier("这个餐馆太难吃了。")
    print(preds)

    # 翻译模型，英翻中
    translation = pipeline(task="translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh", device=0)
    text = "I like to learn data science and AI."
    translated_text = translation(text)
    print(translated_text)

    # 语音识别
    transcriber = pipeline(model="openai/whisper-medium", device=0)
    result = transcriber("./data/podcast_clip.mp3")
    print(result)

    # 中文语音识别
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    transcriber = pipeline(model="openai/whisper-medium", device=0, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
    result = transcriber("./data/podcast_clip.mp3")
    print(result)
