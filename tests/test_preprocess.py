from nlptools.preprocess import clean_text, split_text

print("Test [SPLIT]")
sent = "哈哈哈, 你好呀，嘿嘿哈哈哈哈，诶诶阿法！"
assert len(split_text(sent)) == 1
assert len(split_text(sent, cut_comma=True)) == 2
assert len(split_text(sent, cut_all=True)) == 4
sent = "虽然今天天气不错而且还发工资，但我还是很不开心因为失恋了"
assert len(split_text(sent, turn=True, coo=True)) == 3

print("Test [CLEAN]")
