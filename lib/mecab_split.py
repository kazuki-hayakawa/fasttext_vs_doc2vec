import sys
import MeCab
import unicodedata

class mecab_split():
    """ mecabで分かち書きなどを行う処理まとめ """
    def __init__(self):
        pass

    @staticmethod
    def split(text):
        #文字コード変換処理。変換しないと濁点と半濁点が分離する。
        text = unicodedata.normalize('NFC', text)

        result = []
        tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        tagger.parse('') #parseToNode前に一度parseしておくとsurfaceの読み取りエラーを回避できる

        nodes = tagger.parseToNode(text)
        while nodes:
            if nodes.feature.split(',')[0] in ['名詞']:
                word = nodes.surface
                result.append(word)
            nodes = nodes.next
        return ' '.join(result)
