#!/usr/bin/env python3
# -*- coding utf-8 -*-

# import wave
# import pyaudio
from paddlespeech.cli.tts.infer import TTSExecutor

'''
文本转语音
使用服务，语音保存到内存
'''

'''
# 使用pyaudio播放wav格式文件
# 1、需要先安装portaudio
# MAC：
# brew install portaudio
#
# Ubuntu：
# sudo apt-get install portaudio19-dev
#
# Win+MSYS2：
# pacman -S mingw-w64-x86_64-portaudio
# 
# 2、安装pyaudio
# pip install pyaudio
def play_wav_audio(wav_file):
    # open the wave file
    wf = wave.open(wav_file, 'rb')

    # instantiate PyAudio
    p = pyaudio.PyAudio()

    # open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data from the wave file and play it
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
'''


if __name__ == '__main__':
    tts_executor = TTSExecutor()

    text = "早上好, how are you? 百度Paddle Speech一样能做中英文混合的语音合成。"
    output_file = "./data/paddlespeech_mix.wav"

    # am（acoustic model），声学模型
    # voc（vocoder），音码器
    # lang，语言模型
    # spk_id，声音选择
    tts_executor(text=text, output=output_file, 
                am="fastspeech2_mix", voc="hifigan_csmsc", 
                lang="mix", spk_id=174)

    # play_wav_audio(output_file)
