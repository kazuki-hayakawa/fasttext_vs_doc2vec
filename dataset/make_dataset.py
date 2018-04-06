# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os

classes = ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme",
           "movie-enter","peachy","smax","sports-watch","topic-news"]

df_texts = pd.DataFrame(columns=['class','body'])

if __name__ == '__main__':

    # テキストの抽出
    for c in classes:
        filepath = './news_text/' + c + '/*.txt'
        files = glob.glob(filepath)

        for i in files:
            f = open(i)
            text = f.read()
            text = text.replace("\u3000","")
            f.close()
            row = pd.Series([c, "".join(text.split("\n")[3:])], index=df_texts.columns)
            df_texts = df_texts.append(row, ignore_index=True)


    # トレーニング、バリデーションで 8:2 に分割
    df_train = pd.DataFrame(columns=['class','body'])
    df_validation = pd.DataFrame(columns=['class','body'])

    for c in classes:
        df_text_c = df_texts[df_texts['class'] == c]

        df_text_c_train = df_text_c.head(round(len(df_text_c)*0.8))
        df_train = df_train.append(df_text_c_train, ignore_index=True)

        df_text_c_validation = df_text_c.tail(round(len(df_text_c)*0.2))
        df_validation = df_validation.append(df_text_c_validation, ignore_index=True)

    # テキストの保存
    df_train.to_csv('dataset_train.csv')
    df_validation.to_csv('dataset_validation.csv')
