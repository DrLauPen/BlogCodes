import numpy as np


def TFCalculate(OurDoc):
    wordsofbag = {}
    allwords = []  # 记录所有的词数
    # 首先统计所有的词数
    for doc in OurDoc:
        words = doc.split(' ')
        for word in words:
            if word not in wordsofbag.keys():
                wordsofbag[word] = 1
                allwords.append(word)
            else:
                wordsofbag[word] += 1

    tfvec = []  # 文档的tf(t,d)向量

    for doc in OurDoc:
        # 计算每个文档的tf值向量,tf即词文档出现频率
        words = doc.split(' ')
        newvec = [0 for _ in range(len(allwords))]
        for word in words:
            newvec[allwords.index(word)] += 1
        tfvec.append(newvec)

    return wordsofbag, tfvec, allwords


def IDFCalculate(tfvec, alllwords):
    """计算IDF的值,公式等于文档的<log(总数/(1+出现该词的文档数))>"""
    nd = len(tfvec)  # 文档数目
    idf = []  # 所有文档df的值,即出现该词的文档数
    for vec in tfvec:
        df = [0 for _ in range(len(allwords))]
        for i in range(len(df)):
            if vec[i]:
                df[i] += 1
        df = list(map(lambda x: np.log(nd / (x + 1)), df))  # +1防止分母为0
        idf.append(df)
    return idf


def TF_IDFCalculate(tfvec, idfvec):
    """计算最后的TF-IDF的向量表示"""
    TF_IDF = []
    for docnum in range(len(tfvec)):
        newvec = []
        for doccolumn in range(len(tfvec[0])):
            newvec.append(tfvec[docnum][doccolumn] * idfvec[docnum][doccolumn])
        TF_IDF.append(newvec)
    return TF_IDF


if __name__ == '__main__':
    # 定义文档，这里将每一句话视为一个文档
    OurDoc = ["the sun is shining",
              "the weather is sweet",
              "the sun is shining and the weather is sweet"]
    wordsofbag, tfvec, allwords = TFCalculate(OurDoc)
    idfvec = IDFCalculate(tfvec, allwords)
    TF_IDFVec = TF_IDFCalculate(tfvec, idfvec)
