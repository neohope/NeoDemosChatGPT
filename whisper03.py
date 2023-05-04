#!/usr/bin/env python3
# -*- coding utf-8 -*-

import whisper

'''
通过本地模型使用whisper
https://github.com/openai/whisper
'''

def transcript(clip, prompt, output):
        result = model.transcribe(clip, initial_prompt=prompt)
        with open(output, "w") as f:
            f.write(result['text'])
        print("Transcripted: ", clip)


if __name__ == '__main__':
    # 加载最大的模型
    model = whisper.load_model("large")

    # 对于分割好的文件进行翻译，使用本地GPU
    original_prompt = "这是一段Onboard播客，里面会聊到ChatGPT以及PALM这个大语言模型。这个模型也叫做Pathways Language Model。\n\n"
    prompt = original_prompt
    index = 11
    for i in range(index):
        clip = f"./data/podcast_clip_{i}.mp3"
        output = f"./data/transcripts/local_podcast_clip_{i}.txt"
        transcript(clip, prompt, output)

        # 获取本次翻译的最后一句话，作为下一次的promot
        with open(output, "r") as f:
            transcript = f.read()
        sentences = transcript.split("。")
        prompt = original_prompt + sentences[-1]
