#!/usr/bin/env python3
# -*- coding utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import yaml

'''
工具类
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        print(yaml_data["openai"]["api_key"])


embedding_encoding = "cl100k_base" # this the encoding for text-embedding-ada-002
max_tokens = 8000 # the maximum for text-embedding-ada-002 is 8191
def twenty_newsgroup_to_csv():
    # 下载20_newsgroup.csv
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame( newsgroups_train.target_names, columns=['title'])

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out.to_csv('20_newsgroup.csv', index=False)


if __name__ == '__main__':
    get_api_key()
    #twenty_newsgroup_to_csv()
