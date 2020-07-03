import jieba
from LAC import LAC
from tokenizers import BertWordPieceTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer = None

    def tokenize(self, text):
        raise NotImplementedError()

class BERTTokenizer(Tokenizer):
    def __init__(self):
        super(BERTTokenizer, self).__init__()
        self.tokenizer = BertWordPieceTokenizer("data/vocab/bert.txt", lowercase=True)

    def tokenize(self, text):
        seg_result = self.tokenizer.encode(text).tokens
        return seg_result

class LACTokenizer(Tokenizer):
    def __init__(self, use_cuda=False):
        super(LACTokenizer, self).__init__()
        self.tokenizer = LAC(mode='seg', use_cuda=use_cuda)

    def tokenize(self, text):
        seg_result = self.tokenizer.run(text)
        return seg_result

class JiebaTokenizer(Tokenizer):
    def __init__(self):
        super(JiebaTokenizer, self).__init__()
        self.tokenizer = jieba

    def tokenize(self, text):
        if isinstance(text, str):
            seg_result = list(jieba.cut(text))
        elif isinstance(text, list):
            seg_result = [list(jieba.cut(t)) for t in text]
        return seg_result


class CharTokenizer(Tokenizer):
    def __init__(self):
        super(CharTokenizer, self).__init__()

    # TODO: 英文单词和数字不进行拆分
    def tokenize(self, text):
        if isinstance(text, str):
            seg_result = list(text)
        elif isinstance(text, list):
            seg_result = [list(text) for t in text]
        return seg_result
