# -*- coding: utf-8 -*-

import html
import re
from pathlib import Path

import pkg_resources
from symspellpy.symspellpy import SymSpell

from nlptools.utils.common import read_dict

URL_REGEX = re.compile(
    r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>\u4e00-\u9fa5【】]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’【】\u4e00-\u9fa5]))',
    re.IGNORECASE)
EMAIL_REGEX = re.compile(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", re.IGNORECASE)
WEIBO_REGEX = re.compile(r"(回复)?(//)?\s*@\S*?\s*(:|：| |$)")
PUNC_REGEX = re.compile(r"[，\_《。》、？；：‘’＂“”【「】」、·！@￥…（）—\,\<\.\>\/\?\;\:\'\"\[\]\{\}\~\`\!\@\#$\%\^\&\*\(\)\-\=\+]")

sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
_ = sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

BASE_DIR = Path(__file__).parent.parent / "data"
START_PATTERN = [
    r'\* ',
    r'\d{1,2}\.\d{1,2}\.\d{1,2}',  # 1.2.1
    r'\d+\t',
    r'(?<![:])[1-9]?[0-9][；:、\.\t/]{1}\s?(?![年月日\d+])',
    r'[1-9]?[0-9][)）]{1}、?',
    r'\n[1-9][0-9]',
    r'(?<![A-Za-z0-9/])[A-Za-z]{1}\s?[、\.\)、\t]{1}',
    r'\(1?[1-9]\)',
    r'(?<!周)第?[一二三四五六七八九十]+[、\)\.) \t]{1}',
    r'\([一二三四五六七八九十]+\)\.?',
    r'[①②③④⑤⑥⑦⑧⑨⑩]+',
    r'[★◇]\s?'
]
START_PATTERN = re.compile(r'(' + '|'.join(START_PATTERN) + ')+', re.UNICODE)  # 项目序号
END_PATTERN = r'([。！!﹗？\?])([^"”‘\'])'  # 单字符断句符
EN_ELLIPSIS = r'(\.{6})([^"”‘\'])'  # 英文省略号
CN_ELLIPSIS = r'(\…{2})([^"”‘\'])'  # 中文省略号
QUOTE = r'([。！？\?][”’])([^，。！？\?])'  # 双引号内有终止符，引号为句子终点
TURN_WORDS = read_dict(Path(BASE_DIR) / "turn_words.txt")
TURN_PATTERN = re.compile(r'(' + "|".join(TURN_WORDS) + ')+', re.UNICODE) # 转折词
COO_WORDS = read_dict(Path(BASE_DIR) / "coordinate_words.txt")
COO_PATTERN = re.compile(r'(' + "|".join(COO_WORDS) + ')+', re.UNICODE) # 并列词

def strQ2B(ustring):
    """把字符串全角转半角"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def clear_rare_char(input_char):
    if u'\u4e00' <= input_char <=u'\u9fa5' \
        or re.search(PUNC_REGEX, input_char) \
        or u'\u0030' <= input_char <= u'\u0039' \
        or u'\u0041' <= input_char <= u'\u005A' \
        or u'\u0061' <= input_char <= u'\u007A' \
        or input_char in ["\n", "\t"]:
        return input_char
    return ''

# TODO: 去除连续相同字符
def clean_text(text, remove_url=True, email=True, weibo_at=True, weibo_topic=False,
               remove_rare_char=True, emoji=True, norm_html=True, remove_punc=False,
               q2b=False, remove_dup=False, verbose=False):
    if verbose:
        print(text)
    if remove_dup:
        text = re.sub(r"([^\u4e00-\u9fa5a-zA-Z0-9])\1{3,}", r"\1", text)
        if verbose:
            print(text, 1)
    if remove_url:
        text = re.sub(URL_REGEX, "", text)
        if verbose:
            print(text, 2)
    if email:
        text = re.sub(EMAIL_REGEX, "", text)
        if verbose:
            print(text, 3)
    if weibo_at:
        text = re.sub(WEIBO_REGEX, "", text)
        if verbose:
            print(text, 4)
    if weibo_topic:
        text = re.sub(r"#\S+#[：]?", "", text)
        if verbose:
            print(text, 5)
    if emoji:
        text = re.sub(r"\[\S+\]", "", text)
        if verbose:
            print(text, 6)
    if norm_html:
        text = html.unescape(text)
        if verbose:
            print(text, 7)
    if remove_punc:
        text = re.sub(PUNC_REGEX, "", text)
        if verbose:
            print(text, 8)
    if q2b:
        text = strQ2B(text)
        if verbose:
            print(text, 9)
    if remove_rare_char:
        new_text = ""
        for char in text:
            new_text += clear_rare_char(char)
        text = new_text
        if verbose:
            print(text, 10)
    text = re.sub(r"\s{2,}", " ", text)
    if re.search(r"[a-zA-Z]+", text):
        text = sym_spell.word_segmentation(text).corrected_string
    return text.strip()

def split_text(sentence, bullet=True, turn=False, coo=False,
               cut_comma=False, cut_all=False):
    sentence = re.sub(END_PATTERN, r'\1\n\2', sentence)
    sentence = re.sub(EN_ELLIPSIS, r'\1\n\2', sentence)
    sentence = re.sub(CN_ELLIPSIS, r'\1\n\2', sentence)
    sentence = re.sub(QUOTE, r'\1\n\2', sentence)
    if bullet:
        sentence = re.sub(r'(?<=[\u4e00-\u9fa5a-z])(\.)(?=[\u4e00-\u9fa5a-z])', r'\1\n', sentence)
        sentence = re.sub(START_PATTERN, r'\n\1', sentence)
    if turn:
        sentence = re.sub(TURN_PATTERN, r'\n\1', sentence)
    if coo:
        sentence = re.sub(COO_PATTERN, r'\n\1', sentence)
    sentence = re.sub(r'(?<=\n)(\s+)(?=\n)', '', sentence)
    sentence = re.sub(r'\n{2,}|\\n', '\n', sentence)
    sub_sents = [sub.strip() for sub in re.split("\n|SEP", sentence)]
    sub_sents = list(filter(lambda x: len(x) > 1, sub_sents))
    if cut_comma:
        new_sub_sents = []
        for sent in sub_sents:
            new_subs = re.split(r",|，", sent)
            ss = []
            for i in range(len(new_subs)):
                current_sent = new_subs[i]
                if len(current_sent) < 8:
                    if i == len(new_subs) - 1:
                        new_sub_sents.append(current_sent)
                    else:
                        ss.append(current_sent)
                else:
                    new_sub_sents.append(current_sent)
                    ss = []
                if len(ss) > 2:
                    new_sub_sents.append("，".join(ss))
                    ss = []
        sub_sents = new_sub_sents
    if cut_all:
        sub_sents = [ss for sent in sub_sents for ss in re.split(r",|，", sent)]
    return sub_sents
