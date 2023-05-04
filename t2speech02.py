#!/usr/bin/env python3
# -*- coding utf-8 -*-
import yaml
import azure.cognitiveservices.speech as speechsdk

'''
文本转语音
使用azure云服务，可以选用语音、选用角色、选用语音风格
可以通过SSML格式输入
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

    # 创建speech_synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # 带场景的文本转语音
    # 通过Styles和Roles进行控制
    ssml = """<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
        xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
        <voice name="zh-CN-YunyeNeural">
            儿子看见母亲走了过来，说到：
            <mstts:express-as role="Boy" style="cheerful">
                “妈妈，我想要买个新玩具”
            </mstts:express-as>
        </voice>
        <voice name="zh-CN-XiaomoNeural">
            母亲放下包，说：
            <mstts:express-as role="SeniorFemale" style="angry">
                “我看你长得像个玩具。”
            </mstts:express-as>
        </voice>
    </speak>"""
    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()

    
    ssml = """<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
        xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
        <voice name="en-US-JennyNeural">
            <mstts:express-as style="excited">
                That'd be just amazing!
            </mstts:express-as>
            <mstts:express-as style="friendly">
                What's next?
            </mstts:express-as>
        </voice>
    </speak>"""
    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()