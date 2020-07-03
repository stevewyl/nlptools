import re

def to_region(segmentation: str) -> list:
    """
    将分词结果转换为区间
    :param segmentation: 商品 和 服务
    :return: [(0, 2), (2, 3), (3, 5)]
    """
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end
    return region

def cws_metrics(gold_list: str, pred_list: str, dic, digit_size=4) -> tuple:
    """
    计算P、R、F1
    :param gold: 标准答案字符串列表，比如“商品 和 服务”
    :param pred: 分词结果字符串列表，比如“商品 和服 务”
    :param dic: 词典
    :return: (P, R, F1, OOV_R, IV_R)
    """
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    for gold, pred in zip(gold_list, pred_list):
        A, B = set(to_region(gold)), set(to_region(pred))
        A_size += len(A)
        B_size += len(B)
        A_cap_B_size += len(A & B)
        text = re.sub("\\s+", "", gold)
        for (start, end) in A:
            word = text[start: end]
            if word in dic:
                IV += 1
            else:
                OOV += 1
        for (start, end) in A & B:
            word = text[start: end]
            if word in dic:
                IV_R += 1
            else:
                OOV_R += 1
    p, r = round(A_cap_B_size / B_size, digit_size), round(A_cap_B_size / A_size, digit_size)
    f1 = round(2 * p * r / (p + r), digit_size)
    OOV_R = round(OOV_R / OOV, digit_size)
    IV_R = round(IV_R / IV, digit_size)
    print(f"Precision: {p}\nRecall: {r}\nF1: {f1}")
    print(f"OOV_Recall: {OOV_R}\nIV_Recall: {IV_R}")
