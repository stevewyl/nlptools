import re
from pathlib import Path

import ahocorasick
import jieba
from LAC import LAC
from tokenizers import BertWordPieceTokenizer

DATA_DIR = Path(__file__).parent.parent / "data"


class Tokenizer:
    def __init__(self):
        self.tokenizer = None

    def tokenize(self, text):
        raise NotImplementedError()


class BERTTokenizer(Tokenizer):
    def __init__(self):
        super(BERTTokenizer, self).__init__()
        self.tokenizer = BertWordPieceTokenizer(DATA_DIR / "vocab" / "bert.txt", lowercase=True)

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
        seg_result = list(jieba.cut(text))
        return seg_result


class CharTokenizer(Tokenizer):
    def __init__(self):
        super(CharTokenizer, self).__init__()
        self.cn_pattern = re.compile(r"[\u4e00-\u9fa5]+")
        self.en_pattern = re.compile(r"[a-zA-Z]+")
        self.num_pattern = re.compile(r"[0-9]+")

    def basic_tokenize(self, text):
        return list(text)

    def advance_tokenize(self, text):
        def char_type(text):
            for char in text:
                if self.cn_pattern.search(char):
                    yield (char, 'cn')
                elif self.en_pattern.search(char):
                    yield (char, 'en')
                elif self.num_pattern.search(char):
                    yield (char, 'num')
                else:
                    yield (char, 'other')
        char_list = [c for c in char_type(text)]
        char_list_len = len(char_list)
        tmp = []
        for idx, item in enumerate(char_list):
            if item[1] in ['en', 'num']:
                if idx < char_list_len - 1:
                    if char_list[idx+1][1] == item[1]:
                        tmp.append(item[0])
                    else:
                        tmp.append(item[0])
                        yield ''.join(tmp)
                        tmp = []
                else:
                    tmp.append(item[0])
                    yield ''.join(tmp)
            else:
                yield item[0]

    def tokenize(self, text, cut_all=True):
        """
        cut_all=True: 字符切分
        cut_all=False: 连续的英文和数字将不进行切分
        """
        if cut_all:
            return self.basic_tokenize(text)
        else:
            return list(self.advance_tokenize(text))


class PinyinTokenizer(Tokenizer):
    def __init__(self, pinyin_freqs=Path(DATA_DIR) / "pinyin_freqs.txt"):
        """
        Implement from: https://spaces.ac.cn/archives/3908
        内置的是中文人名中的拼音频数，可以根据自己需求更改频数文件
        """
        super(PinyinTokenizer, self).__init__()
        self.pinyin_ac = self.build(pinyin_freqs)
        self.break_ac = ahocorasick.Automaton()
        for c in ['，', '。', '！', '、', '？', ' ', '\n']:
            self.break_ac.add_word(c, c)
        self.break_ac.make_automaton()

    def build(self, dict_fn):
        dic = ahocorasick.Automaton()
        total = 0.0
        words = []
        for line in open(dict_fn):
            word, cnt = line.strip().split('\t')
            words.append((word, int(cnt)))
            total += int(cnt)
        for word, cnt in words:
            dic.add_word(word, (word, log(cnt / total))) #这里使用了对数概率，防止溢出
        dic.make_automaton()
        return dic

    def all_cut(self, text):
        """全切分"""
        words = []
        for end_idx, word_score in self.pinyin_ac.iter(text):
            words.append(word_score[0])
        return words

    def max_match_cut(self, text):
        """最长匹配"""
        words = ['']
        for char in text:
            if self.pinyin_ac.match(words[-1] + char):
                words[-1] += char
            else:
                words.append(char)
        return words

    def max_proba_cut(self, text):
        """
        最大概率组合切分
        有向无环图里边的最大概率路径
        """
        paths = {0: ([], 0)}
        end = 0
        for end_idx, word_score in self.pinyin_ac.iter(text):
            word_tmp, score = word_score
            start, end = 1 + end_idx - len(word_tmp), end_idx + 1
            if start not in paths:
                last = max([i for i in paths if i < start])
                paths[start] = (paths[last][0] + [text[last:start]], paths[last][1] - 10)
            proba = paths[start][1] + score
            if end not in paths or proba > paths[end][1]:
                paths[end] = (paths[start][0] + [word_tmp], proba)
        if end < len(text):
            return paths[end][0] + [text[end:]]
        else:
            return paths[end][0]

    def map_cut(self, text):
        """
        结合一些天然的标点分隔符
        """
        start = 0
        words = []
        for end_idx, word_score in self.break_ac.iter(text):
            words.extend(self.max_proba_cut(text[start:end_idx + 1]))
            start = end_idx + 1
        words.extend(self.max_proba_cut(text[start:]))
        return words

    def min_words_cut(self, text):
        """
        最烧词数切分
        有多少个词罚多少分，未登录词再罚一分，最后罚分最少的胜出
        """
        paths = {0: ([], 0)}
        end = 0
        for end_idx, word_score in self.pinyin_ac.iter(text):
            word_tmp, score = word_score
            start, end = 1 + end_idx - len(word_tmp), end_idx + 1
            if start not in paths:
                last = max([i for i in paths if i < start])
                paths[start] = (paths[last][0] + [text[last:start]], paths[last][1] + 1)
            num = paths[start][1] + 1
            if end not in paths or num < paths[end][1]:
                paths[end] = (paths[start][0] + [word_tmp], num)
        if end < len(text):
            return paths[end][0] + [text[end:]]
        else:
            return paths[end][0]

    def tokenize(self, text):
        return self.max_proba_cut(text)
