def get_common_substr(str1, str2):
    """返回最大公共子串"""
    lstr1 = len(str1)
    lstr2 = len(str2)
    if lstr1 == 0 or lstr2 == 0:
        return ""
    p = 0
    maxNum = 0
    record = [[0 for j in range(lstr2+1)] for i in range(lstr1+1)]
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maxNum:
                    maxNum = record[i+1][j+1]
                    p = i + 1
    return str1[p-maxNum:p]
