# coding: utf-8
import sys
import os
import pandas as pd

sys.path.append(os.pardir)
from fthandler import FastTextHandler
from lib.mecab_split import mecab_split as ms

dataset_file = '../dataset/dataset_train.csv'
words_file = 'newswords.txt'

model_name = 'ft_model'
epoch = 10
window_size = 10
dim = 200


def main():
    # ニュースのテキストから fasttext 用に分かち書きした単語テキストファイル作成
    make_wordsfile()

    # fasttext のインスタンス作成
    fth = FastTextHandler(epoch=epoch, window_size=window_size, dim=dim)

    # fasttext の skip-gram で学習モデル作成
    fth.generate_model(file_name=words_file, model_name=model_name)


def make_wordsfile():
    # ニュースのテキストデータを取り出し分かち書きを行う
    df_text = pd.read_csv(dataset_file)
    wordlist = [ms.split(body) for body in df_text['body'].astype('str').values]
    words = ' '.join(wordlist)

    with open(words_file, 'w') as file:
        file.write(words)


if __name__ == '__main__':
    main()
