# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from lib.mecab_split import mecab_split as ms
from model_fasttext.fthandler import FastTextHandler

dataset_train_path = './dataset/dataset_train.csv'
dataset_validation_path = './dataset/dataset_validation.csv'

# 圧縮後の文書ベクトルの次元数
vector_dim = 20

# クラス名をクラスIDに変換するための辞書
classToClassid = {
    'dokujo-tsushin' : 0,
    'it-life-hack' : 1,
    'kaden-channel' : 2,
    'livedoor-homme' : 3,
    'movie-enter' : 4,
    'peachy' : 5,
    'smax' : 6,
    'sports-watch' : 7,
    'topic-news' : 8
}

# RandomForest のグリッドサーチで探索するパラメータ
params = {
    'min_samples_leaf' : [i+1 for i in range(10)],
    'max_depth' : [i+1 for i in range(5)]
}


def main():
    # トレーニング、バリデーション用データ作成
    df_train = preprocessing(dataset_train_path)
    df_validation = preprocessing(dataset_validation_path)

    # fasttest モデルの成果率を計算
    acc_ft_model = classifier_by_ft_model(df_train, df_validation)

    # doc2vec モデルの成果率を計算
    acc_d2v_model = classifier_by_d2v_model(df_train, df_validation)

    # 結果を表示
    print('\n================================')
    print('accuracy: fasttext model ')
    print("{:.3}".format(acc_ft_model))

    print('\n================================')
    print('accuracy: doc2vec model')
    print("{:.3}".format(acc_d2v_model))

    print('\n================================')


def preprocessing(filepath):
    """ データセットの読み込み、前処理 """
    dataset = pd.read_csv(filepath)
    dataset['class_id'] = [classToClassid[c] for c in dataset['class']]

    return dataset


def dimension_reduction(data, pca_dimension):
    """ dataframeの vector カラムのベクトルを任意の次元に圧縮する

    Parameters
    ----------
        data : DataFrame
            vector カラムを持つデータフレーム
        pca_dimension : int
            PCAで圧縮したい次元数

    Returns
    -------
        pca_data : DataFrame
            vector カラムを次元圧縮したデータフレーム
    """

    # 文章ベクトルの次元圧縮
    pca_data = data.copy()
    pca = PCA(n_components=pca_dimension)
    vector = np.array([np.array(v) for v in pca_data['vector']])
    pca_vector = pca.fit_transform(vector)
    pca_data['pca_vector'] = [v for v in pca_vector]
    del pca_data['vector']
    pca_data.rename(columns={'pca_vector':'vector'}, inplace=True)

    return pca_data


def set_ft_vector(dataframe, fth_instance, dim=50):
    """ fasttext モデルを使って文章をベクトル化し、カラムに加える

    Parameters
    ----------
        dataframe : DataFrame
            body カラムを持つデータフレーム
        fth_instance : class instance
            モデルロード済み FastTextHandler クラスのインスタンス
        dim : int, default 50
            圧縮ベクトルの次元数

    Returns
    -------
        df_vecadd : DataFrame
            次元圧縮済みベクトルのカラムを追加したデータフレーム

    """
    # テキストのベクトル化
    df_tmp = fth_instance.set_df_text(dataframe)

    # ベクトルの次元を圧縮
    df_tmp = dimension_reduction(df_tmp, dim)

    # 不要なカラムを削除
    del df_tmp['body']
    del df_tmp['class_id']

    df_vecadd = pd.merge(dataframe, df_tmp, how='left',
                        left_index=True, right_index=True)

    return df_vecadd


def set_d2v_vector(dataframe, d2v_instance, dim=50):
    """ doc2vec モデルを使って文章をベクトル化し、カラムに加える

    Parameters
    ----------
        dataframe : DataFrame
            body カラムを持つデータフレーム
        d2v_instance : class instance
            モデルロード済み Doc2Vec クラスのインスタンス
        dim : int, default 50
            圧縮ベクトルの次元数

    Returns
    -------
        df_vecadd : DataFrame
            次元圧縮済みベクトルのカラムを追加したデータフレーム

    """
    df_tmp = dataframe.copy()

    # doc2vec でベクトル化するには文書を単語のリストとして保つ必要があるので、変形する
    df_tmp['doc_words'] = [ms.split(body).split(' ') for body in df_tmp['body']]

    # 文書ベクトル作成
    df_tmp['vector'] = [d2v_instance.infer_vector(doc_words) for doc_words in df_tmp['doc_words']]

    # ベクトルの次元を圧縮
    df_tmp = dimension_reduction(df_tmp, dim)

    # 不要なカラムを削除
    del df_tmp['body']
    del df_tmp['class_id']

    df_vecadd = pd.merge(dataframe, df_tmp, how='left',
                        left_index=True, right_index=True)

    return df_vecadd


def accuracy_randomforest_calassifier(train_data, validation_data):
    """ ランダムフォレストでクラス分類を行い、正解率を算出
    Parameters
    ----------
        train_data : DataFrame
        validation_data : DataFrame
            いずれも vector, class_id のカラムを持つ

    Returns
    -------
        acc : float
    """

    X_train = np.array([np.array(v) for v in train_data['vector']])
    X_validation = np.array([np.array(v) for v in validation_data['vector']])
    y_train = np.array([i for i in train_data['class_id']])
    y_validation = np.array([i for i in validation_data['class_id']])

    # RandomForest モデル学習
    mod = RandomForestClassifier()
    clf = GridSearchCV(mod, params)
    clf.fit(X_train, y_train)

    # 予測, 正解率算出
    y_pred = clf.predict(X_validation)
    acc = accuracy_score(y_validation, y_pred)

    return acc


def classifier_by_ft_model(train_data, validation_data):
    """ fasttext で作成したモデルを利用したクラス分類

    Parameters
    ----------
        train_data : DataFrame
        validation_data : DataFrame

    Returns
    -------
        accuracy : float
    """
    # モデルのロード
    ft_model_path = './model_fasttext/ft_model'
    fth = FastTextHandler()
    fth.load(model_name=ft_model_path)

    # ベクトルデータ作成
    train_data = set_ft_vector(train_data, fth, dim=vector_dim)
    validation_data = set_ft_vector(validation_data, fth, dim=vector_dim)

    # ランダムフォレストで正解率計算
    accuracy = accuracy_randomforest_calassifier(train_data, validation_data)

    return accuracy


def classifier_by_d2v_model(train_data, validation_data):
    """ doc2vec で作成したモデルを利用したクラス分類

    Parameters
    ----------
        train_data : DataFrame
        validation_data : DataFrame

    Returns
    -------
        accuracy : float
    """
    # モデルのロード
    d2v_model_path = './model_doc2vec/doc2vec.model'
    d2v = Doc2Vec.load(d2v_model_path)

    # ベクトルデータ作成
    train_data = set_d2v_vector(train_data, d2v, dim=vector_dim)
    validation_data = set_d2v_vector(validation_data, d2v, dim=vector_dim)

    # ランダムフォレストで正解率計算
    accuracy = accuracy_randomforest_calassifier(train_data, validation_data)

    return accuracy


if __name__ == '__main__':
    main()
