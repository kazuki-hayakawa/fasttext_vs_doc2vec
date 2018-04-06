# fasttext と doc2vec でモデルを作成して、文書のクラス分類を行う

## データセット

データセットはlivedoorニュースコーパスを利用
https://www.rondhuit.com/download.html#ldcc

`/dataset/news_text` ディレクトリ配下にテキストを格納。

```
python make_dataset.py
```

でデータセット作成。

## fasttext モデル作成

`/model_fasttext/` ディレクトリ配下で

```
python make_ft_model.py
```

を実行

## doc2vec モデル作成

`/model_doc2vec/` ディレクトリ配下で

```
python make_d2v_model.py
```

を実行

## 文書分類

```
python classifierl.py
```

を実行
