#!/usr/bin/env python3
# -*- coding utf-8 -*-
import yaml
import azure.cognitiveservices.speech as speechsdk

'''
文本转语音
使用azure云服务，语音保存到内存
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

    # 指定语言及声音
    speech_config.speech_synthesis_language='zh-CN'
    speech_config.speech_synthesis_voice_name='zh-CN-XiaohanNeural'
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3)

    # 声音保存到内存
    # 创建speech_synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    # 获取文件流
    text = "今天天气真不错，ChatGPT真好用"
    result = speech_synthesizer.speak_text_async(text).get()
    stream =speechsdk.AudioDataStream(result)

    # 保存文件流
    stream.save_to_wav_file("./data/tts.mp3")
