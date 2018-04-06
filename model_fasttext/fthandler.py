# coding: utf-8

from __future__ import print_function

import os
import sys
import re

import fasttext as ft
import numpy as np
import pandas as pd

sys.path.append(os.pardir)
from lib.mecab_split import mecab_split as ms

class FastTextHandler(object):

    def __init__(self, use_surface=False, use_neologd=False,
                 chain_noun=False, unique=False, dim=200, epoch=5,
                 window_size=5):
        super(FastTextHandler, self).__init__()
        self.__params_fasttext = {
            'dim': dim,
            'epoch': epoch,
            'ws': window_size
        }
        self.model = None
        self.df_word = pd.DataFrame()

    def generate_model(self, file_name='doc.txt', model_name='model',
                       load=True):
        u"""document名・model名を指定してmodelを生成する。

        Args:
            file_name : String,  形態素解析したドキュメントデータ
            model_name: String,  skip-gramの結果生成されるファイルの名前
            load      : Boolean, model生成後、そのmodelをloadするか否か
        """
        ft.skipgram(file_name, model_name, **self.__params_fasttext)
        if load:
            self.model = ft.load_model(model_name + '.bin')
            self.df_word = self.model2df()

    def load(self, model_name='model'):
        u"""model名を指定してloadする。"""
        self.model = ft.load_model(model_name + '.bin')
        self.df_word = self.model2df()

    def model2df(self):
        u"""fasttext.modelをpd.DataFrameに変換する。"""
        return pd.DataFrame([[i + 1, v, self.model[v]] for i, v in
                             enumerate(self.model.words)],
                            columns=['id', 'content', 'vector'])

    def set_df_text(self, dataframe):
        """ テキストのデータフレームを整形してベクトル追加 """
        if 'body' not in dataframe.columns:
            print(u'文書がありません。body の名前でカラムを作成してください。')
            return

        else:
            df_text = dataframe.copy()
            if 'vector' not in df_text.columns:
                df_text['vector'] = None


            df_text['body'] = df_text['body'].str.replace(r'<[^>]*>|\r|\n',' ')
            df_text['body'] = df_text['body'].str.replace(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+',' ')
            df_text['vector'] = [
                self.text2vec(body, consider_length=True) for body
                in df_text['body'].astype('str').values]

            return df_text

    def combine_word_vectors(self, words, is_mean=True):
        u"""指定されたワードを組み合わせて新たなベクトルを生成。

        Args:
            words  : Array,   modelに含まれているワード
            is_mean: Boolean, np.meanかnp.sumを使うかのフラグ

        Returns:
            Array, self.dimと同じ次元のベクトル
            np.array([0.123828, -0.232955, ..., 2.987532])
        """
        # modelをloadしていない場合は警告
        if self.model is None:
            print('Model is not set.')
            print('Try `cls.load` or `cls.generate_model`.')
            return

        # modelに含まれているワードかどうかチェック
        words = list(filter(lambda x: x in self.model.words, words))
        if not words:
            print('No words are in this model.')
            return list(np.zeros(self.__params_fasttext['dim']))

        # 指定された関数を適用してベクトルを生成
        apply_func = np.mean if is_mean else np.sum
        return apply_func(list(map(lambda x: self.model[x], words)), axis=0)

    def text2vec(self, text, consider_length=True):
        u"""文章を形態素解析してベクトルの和に変換する。

        Args:
            text           : String , ベクトル化する文章
            consider_length: Boolean, 文章の長さを考慮するか否かのフラグ
        """
        return list(self.combine_word_vectors(
            ms.split(text), is_mean=consider_length))
