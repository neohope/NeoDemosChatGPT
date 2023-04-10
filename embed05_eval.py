#!/usr/bin/env python3
# -*- coding utf-8 -*-


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression


'''
加载向量化文件toutiao_cat_data_all_with_embeddings.parquet
用两种算法，分别评价向量化后的效果
'''


'''
抽取5万条，按4：1拆分
用随机森林方法进行评估
'''
def eval_RandomForest(training_data):
    training_data.head()

    df =  training_data.sample(50000, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedding.values), df.category, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    report = classification_report(y_test, preds)
    print(report)

'''
准确率可以达到84%：
f1-score = 2/(1/precision + 1/recall)

                    precision    recall  f1-score   support

       news_travel       0.79      0.76      0.78       599
        news_world       0.80      0.71      0.75       671
             stock       0.00      0.00      0.00         8

          accuracy                           0.84     10000
         macro avg       0.80      0.76      0.77     10000
      weighted avg       0.84      0.84      0.84     10000
'''


'''
全部数据，按4：1拆分
用逻辑回归方法进行评估
'''
def eval_LogisticRegression(training_data):
    training_data.head()

    X_train, X_test, y_train, y_test = train_test_split(
        list(training_data.embedding.values), training_data.category, test_size=0.2, random_state=42
    )

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    report = classification_report(y_test, preds)
    print(report)

'''
准确率可以达到86%：
f1-score = 2/(1/precision + 1/recall)

                    precision    recall  f1-score   support
  news_agriculture       0.85      0.88      0.87      3908
          news_car       0.92      0.92      0.92      7101
      news_culture       0.82      0.85      0.83      5719
          news_edu       0.88      0.89      0.89      5376
news_entertainment       0.85      0.88      0.86      7908
      news_finance       0.82      0.78      0.80      5409
         news_game       0.91      0.87      0.89      5899
        news_house       0.90      0.91      0.91      3463
     news_military       0.86      0.82      0.84      4976
       news_sports       0.93      0.93      0.93      7611
        news_story       0.83      0.81      0.82      1308
         news_tech       0.84      0.86      0.85      8168
       news_travel       0.80      0.79      0.79      4252
        news_world       0.79      0.80      0.80      5370
             stock       0.00      0.00      0.00        70

          accuracy                           0.86     76538
         macro avg       0.80      0.80      0.80     76538
      weighted avg       0.86      0.86      0.86     76538
'''


if __name__ == '__main__':
    training_data = pd.read_parquet("data\\toutiao_cat_data_all_with_embeddings.parquet")
    eval_RandomForest(training_data)
    eval_LogisticRegression(training_data)

