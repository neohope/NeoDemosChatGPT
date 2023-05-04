#!/usr/bin/env python3
# -*- coding utf-8 -*-
import yaml
import azure.cognitiveservices.speech as speechsdk

'''
文本转语音
使用azure云服务
'''

def get_azure_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        region = yaml_data["azure"]["region"]
        subscription = yaml_data["azure"]["subscription"]
        return region, subscription


if __name__ == '__main__':
    # KEY及区域
    region, subscription = get_azure_key()
    speech_config = speechsdk.SpeechConfig(subscription=subscription, region=region)

    # 选用扬声器
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # 输出语音的语言
    speech_config.speech_synthesis_voice_name='zh-CN-XiaohanNeural'
    # speech_config.speech_synthesis_voice_name='zh-CN-YunfengNeural'

    # 创建speech_synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # 文本转语音
    text = "今天天气真不错，ChatGPT真好用。"
    speech_synthesizer.speak_text_async(text)
