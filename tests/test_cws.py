from nlptools.metrics.cws import cws_metrics
from nlptools.tokenizer import *

tokenizers = {
    "lac": LACTokenizer(),
    "jieba": JiebaTokenizer()
}

for src in ["pku", "msr"]:
    print("======")
    print("Data: ", src)
    gold_list = [line.strip() for line in open(f"data/cws/{src}_test_gold.txt")]
    corpus = [line.strip() for line in open(f"data/cws/{src}_test.txt")]
    words = set(word.strip() for word in open(f"data/cws/{src}_words.txt"))
    for name, tokenizer in tokenizers.items():
        print("Tokenizer: ", name)
        seg_result = [" ".join(seg) for seg in tokenizer.tokenize(corpus)]
        cws_metrics(gold_list, seg_result, words)
    print("======")