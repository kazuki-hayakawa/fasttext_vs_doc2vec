# coding: utf-8
import sys
import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

sys.path.append(os.pardir)
from lib.mecab_split import mecab_split as ms

dataset_file = '../dataset/dataset_train.csv'

def main():
    # doc2vec 用データセットの作成
    df_text = pd.read_csv(dataset_file)
    trainings = [TaggedDocument(words = ms.split(body), tags = [i])
                for i, body in enumerate(df_text['body'])]

    # モデルの学習, dmpvで学習させる
    # ハイパーパラメータはこちら参考： https://deepage.net/machine_learning/2017/01/08/doc2vec.html
    model = Doc2Vec(documents=trainings, dm=1, size=300, window=5, min_count=5)

    # モデルの保存
    model.save('doc2vec.model')


if __name__ == '__main__':
    main()
